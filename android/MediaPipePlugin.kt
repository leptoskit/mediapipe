package dev.leptoskit.mediapipe

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import dev.leptoskit.bridge.NativePlugin
import dev.leptoskit.bridge.PluginException
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Host-owned MediaPipe Tasks Vision plugin for LeptosKit Android apps.
 *
 * Implements the `"mediapipe"` binary wire protocol v1:
 * - method 0 (`STATUS`): capability flags (`pose | object | gesture`).
 * - method 1 (`START_SESSION`): loads verified model files from disk and opens a session.
 * - method 2 (`STOP_SESSION`): closes native MediaPipe tasks for the session.
 * - method 3 (`POLL_OBSERVATION`): pops the latest compact observation for Wasm guest overlays.
 *
 * Host camera adapters push frames directly via [analyzeFrame] without copying raw pixels
 * across Wasm linear memory.
 */
class MediaPipePlugin(
    context: Context,
) : NativePlugin {
    override val name: String = "mediapipe"
    override val methods: IntArray = intArrayOf(0, 1, 2, 3)

    private val appContext = context.applicationContext
    private val nextId = AtomicInteger(1)
    private val sessions = ConcurrentHashMap<Int, VisionSession>()
    @Volatile
    private var closed = false

    companion object {
        private const val WIRE_VERSION = 1
        private val activeInstance = AtomicReference<MediaPipePlugin?>(null)

        /**
         * Feed an upright-oriented camera [Bitmap] into every active vision session attached
         * to [cameraSessionId] (or `0` for standalone sessions).
         */
        @JvmStatic
        fun analyzeFrame(
            cameraSessionId: Int,
            bitmap: Bitmap,
            rotationDegrees: Int,
            mirrored: Boolean,
            timestampMs: Long,
        ) {
            activeInstance.get()?.dispatchFrame(
                cameraSessionId = cameraSessionId,
                bitmap = bitmap,
                rotationDegrees = rotationDegrees,
                mirrored = mirrored,
                timestampMs = timestampMs,
            )
        }
    }

    init {
        activeInstance.set(this)
    }

    private class VisionSession(
        val cameraSessionId: Int,
        val pose: PoseLandmarker?,
        val objects: ObjectDetector?,
        val gestures: GestureRecognizer?,
    ) {
        val latest = AtomicReference<ByteArray?>(null)
        private var lastTimestampMs = -1L

        @Synchronized
        fun nextTimestamp(candidateMs: Long): Long {
            val clamped = candidateMs.coerceAtLeast(0L)
            val emitted = if (clamped <= lastTimestampMs) lastTimestampMs + 1L else clamped
            lastTimestampMs = emitted
            return emitted
        }

        fun close() {
            pose?.close()
            objects?.close()
            gestures?.close()
        }
    }

    override suspend fun invoke(method: Int, payload: ByteArray): ByteArray {
        if (closed) throw PluginException.cancelled()
        return when (method) {
            0 -> handleStatus(payload)
            1 -> handleStartSession(payload)
            2 -> handleStopSession(payload)
            3 -> handlePollObservation(payload)
            else -> throw PluginException.unavailable()
        }
    }

    override fun close() {
        closed = true
        activeInstance.compareAndSet(this, null)
        for (session in sessions.values) {
            session.close()
        }
        sessions.clear()
    }

    private fun handleStatus(payload: ByteArray): ByteArray {
        if (payload.isNotEmpty()) throw PluginException.invalidArgument()
        val buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(WIRE_VERSION)
        buf.putInt(0b111) // pose + object + gesture available via MediaPipe Tasks Vision SDK
        return buf.array()
    }

    private fun handleStartSession(payload: ByteArray): ByteArray {
        val reader = WireReader(payload)
        val version = reader.readU32()
        if (version != WIRE_VERSION) throw PluginException.invalidArgument()
        val cameraSessionId = reader.readU32()
        val flags = reader.readU32()
        val poseModel = reader.readString()
        val objectModel = reader.readString()
        val gestureModel = reader.readString()
        val scoreThreshold = Float.fromBits(reader.readU32())
        val maxResults = reader.readU32()
        val allowLen = reader.readU32()
        if (allowLen < 0 || allowLen > 128) throw PluginException.invalidArgument()
        val allowlist = ArrayList<String>(allowLen)
        repeat(allowLen) { allowlist.add(reader.readString()) }
        reader.finish()

        val enablePose = (flags and 1) != 0
        val enableObjects = (flags and 2) != 0
        val enableGestures = (flags and 4) != 0

        val pose = if (enablePose && poseModel.isNotEmpty()) {
            requireModelFile(poseModel)
            val opts = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(poseModel).build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(1)
                .build()
            PoseLandmarker.createFromOptions(appContext, opts)
        } else {
            null
        }

        val objects = if (enableObjects && objectModel.isNotEmpty()) {
            requireModelFile(objectModel)
            val builder = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(objectModel).build())
                .setRunningMode(RunningMode.VIDEO)
                .setScoreThreshold(scoreThreshold)
                .setMaxResults(maxResults.coerceAtLeast(1))
            if (allowlist.isNotEmpty()) {
                builder.setCategoryAllowlist(allowlist)
            }
            ObjectDetector.createFromOptions(appContext, builder.build())
        } else {
            null
        }

        val gestures = if (enableGestures && gestureModel.isNotEmpty()) {
            requireModelFile(gestureModel)
            val opts = GestureRecognizer.GestureRecognizerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(gestureModel).build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .build()
            GestureRecognizer.createFromOptions(appContext, opts)
        } else {
            null
        }

        val id = nextId.getAndIncrement()
        sessions[id] = VisionSession(cameraSessionId, pose, objects, gestures)
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(id).array()
    }

    private fun handleStopSession(payload: ByteArray): ByteArray {
        val reader = WireReader(payload)
        val id = reader.readU32()
        reader.finish()
        val session = sessions.remove(id) ?: throw PluginException.invalidArgument()
        session.close()
        return ByteArray(0)
    }

    private fun handlePollObservation(payload: ByteArray): ByteArray {
        val reader = WireReader(payload)
        val id = reader.readU32()
        reader.finish()
        val session = sessions[id] ?: throw PluginException.invalidArgument()
        return session.latest.getAndSet(null)
            ?: ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(0).array()
    }

    private fun requireModelFile(path: String) {
        val file = File(path)
        if (!file.isFile || !file.canRead()) {
            throw PluginException.invalidArgument()
        }
    }

    private fun dispatchFrame(
        cameraSessionId: Int,
        bitmap: Bitmap,
        rotationDegrees: Int,
        mirrored: Boolean,
        timestampMs: Long,
    ) {
        if (closed || sessions.isEmpty()) return
        val mpImage = BitmapImageBuilder(bitmap).build()
        val procOpts = ImageProcessingOptions.builder()
            .setRotationDegrees(rotationDegrees)
            .build()
        val uprightWidth = if ((rotationDegrees / 90) % 2 != 0) bitmap.height else bitmap.width
        val uprightHeight = if ((rotationDegrees / 90) % 2 != 0) bitmap.width else bitmap.height

        for (session in sessions.values) {
            if (session.cameraSessionId != 0 && session.cameraSessionId != cameraSessionId) continue
            val ts = session.nextTimestamp(timestampMs)
            val poseResult = session.pose?.detectForVideo(mpImage, procOpts, ts)
            val objectResult = session.objects?.detectForVideo(mpImage, procOpts, ts)
            val gestureResult = session.gestures?.recognizeForVideo(mpImage, procOpts, ts)
            session.latest.set(
                encodeObservation(
                    timestampMs = ts,
                    frameWidth = uprightWidth,
                    frameHeight = uprightHeight,
                    rotationDegrees = rotationDegrees,
                    mirrored = mirrored,
                    pose = poseResult,
                    objects = objectResult,
                    gestures = gestureResult,
                ),
            )
        }
    }

    private fun encodeObservation(
        timestampMs: Long,
        frameWidth: Int,
        frameHeight: Int,
        rotationDegrees: Int,
        mirrored: Boolean,
        pose: PoseLandmarkerResult?,
        objects: ObjectDetectorResult?,
        gestures: GestureRecognizerResult?,
    ): ByteArray {
        val out = WireWriter()
        out.writeU32(1)
        out.writeU32((timestampMs and 0xFFFF_FFFFL).toInt())
        out.writeU32((timestampMs ushr 32).toInt())
        out.writeU32(frameWidth)
        out.writeU32(frameHeight)
        out.writeU32(rotationDegrees)
        out.writeU32(if (mirrored) 1 else 0)

        val firstPose = pose?.landmarks()?.firstOrNull()
        if (firstPose != null) {
            out.writeU32(firstPose.size)
            for (lm in firstPose) {
                out.writeU32(lm.x().toRawBits())
                out.writeU32(lm.y().toRawBits())
                out.writeU32(lm.z().toRawBits())
                out.writeU32(lm.visibility().orElse(-1.0f).toRawBits())
                out.writeU32(lm.presence().orElse(-1.0f).toRawBits())
            }
        } else {
            out.writeU32(0)
        }

        val dets = objects?.detections().orEmpty()
        out.writeU32(dets.size)
        for (det in dets) {
            val box = det.boundingBox()
            out.writeU32(box.left.toRawBits())
            out.writeU32(box.top.toRawBits())
            out.writeU32(box.width().toRawBits())
            out.writeU32(box.height().toRawBits())
            val cats = det.categories()
            out.writeU32(cats.size)
            for (cat in cats) {
                out.writeU32(cat.index())
                out.writeU32(cat.score().toRawBits())
                out.writeString(cat.categoryName().orEmpty())
            }
        }

        val gestureHands = gestures?.gestures().orEmpty()
        out.writeU32(gestureHands.size)
        for (hand in gestureHands) {
            out.writeU32(hand.size)
            for (cat in hand) {
                out.writeU32(cat.index())
                out.writeU32(cat.score().toRawBits())
                out.writeString(cat.categoryName().orEmpty())
            }
        }
        out.writeU32(gestures?.landmarks()?.size ?: 0)
        return out.toByteArray()
    }

    private class WireReader(private val bytes: ByteArray) {
        private var pos = 0

        fun readU32(): Int {
            if (pos + 4 > bytes.size) throw PluginException.invalidArgument()
            val v = ByteBuffer.wrap(bytes, pos, 4).order(ByteOrder.LITTLE_ENDIAN).int
            pos += 4
            return v
        }

        fun readString(): String {
            val len = readU32()
            if (len < 0 || pos + len > bytes.size) throw PluginException.invalidArgument()
            val s = String(bytes, pos, len, Charsets.UTF_8)
            pos += len
            return s
        }

        fun finish() {
            if (pos != bytes.size) throw PluginException.invalidArgument()
        }
    }

    private class WireWriter {
        private val stream = ByteArrayOutputStream(256)

        fun writeU32(value: Int) {
            val buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()
            stream.write(buf)
        }

        fun writeString(value: String) {
            val utf8 = value.toByteArray(Charsets.UTF_8)
            writeU32(utf8.size)
            stream.write(utf8)
        }

        fun toByteArray(): ByteArray = stream.toByteArray()
    }
}
