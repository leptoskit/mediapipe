import Foundation
import LeptosKitBridge
#if canImport(MediaPipeTasksVision)
import MediaPipeTasksVision
#endif

/// Host-owned MediaPipe Tasks Vision plugin for LeptosKit iOS apps.
///
/// Speaks the `"mediapipe"` binary wire protocol v1:
/// - `0` (`STATUS`): runtime capability flags
/// - `1` (`START_SESSION`): open host MediaPipe vision tasks from verified model files
/// - `2` (`STOP_SESSION`): release session tasks
/// - `3` (`POLL_OBSERVATION`): pop latest compact observation payload
@MainActor
public final class MediaPipePlugin: NativePlugin {
    public let name = "mediapipe"
    public let methods: [UInt32] = [0, 1, 2, 3]

    private let wireVersion: UInt32 = 1
    private var nextId: UInt32 = 1
    private var sessions: [UInt32: Session] = [:]
    private var closed = false

    private final class Session {
        let cameraSessionId: UInt32
        var latestPayload: Data?
        init(cameraSessionId: UInt32) {
            self.cameraSessionId = cameraSessionId
        }
    }

    public init() {}

    public func invoke(method: UInt32, payload: NativeBuffer) async throws -> Data {
        guard !closed else { throw PluginError.cancelled }
        switch method {
        case 0:
            guard payload.isEmpty else { throw PluginError.invalidArgument }
            var out = Data()
            appendU32(&out, wireVersion)
            #if canImport(MediaPipeTasksVision)
            appendU32(&out, 0b111)
            #else
            appendU32(&out, 0)
            #endif
            return out
        case 1:
            #if canImport(MediaPipeTasksVision)
            var offset = 0
            let data = Data(payload)
            let version = try readU32(data, &offset)
            guard version == wireVersion else { throw PluginError.invalidArgument }
            let cameraSessionId = try readU32(data, &offset)
            _ = try readU32(data, &offset) // flags
            let posePath = try readString(data, &offset)
            let objectPath = try readString(data, &offset)
            let gesturePath = try readString(data, &offset)
            _ = try readU32(data, &offset) // score_threshold
            _ = try readU32(data, &offset) // max_results
            let allowLen = Int(try readU32(data, &offset))
            guard allowLen <= 128 else { throw PluginError.invalidArgument }
            for _ in 0..<allowLen {
                _ = try readString(data, &offset)
            }
            guard offset == data.count else { throw PluginError.invalidArgument }
            for path in [posePath, objectPath, gesturePath] where !path.isEmpty {
                guard FileManager.default.isReadableFile(atPath: path) else {
                    throw PluginError.invalidArgument
                }
            }
            let id = nextId
            nextId &+= 1
            sessions[id] = Session(cameraSessionId: cameraSessionId)
            var reply = Data()
            appendU32(&reply, id)
            return reply
            #else
            throw PluginError.unavailable
            #endif
        case 2:
            var offset = 0
            let data = Data(payload)
            let id = try readU32(data, &offset)
            guard offset == data.count, sessions.removeValue(forKey: id) != nil else {
                throw PluginError.invalidArgument
            }
            return Data()
        case 3:
            var offset = 0
            let data = Data(payload)
            let id = try readU32(data, &offset)
            guard offset == data.count, let session = sessions[id] else {
                throw PluginError.invalidArgument
            }
            if let latest = session.latestPayload {
                session.latestPayload = nil
                return latest
            }
            var empty = Data()
            appendU32(&empty, 0)
            return empty
        default:
            throw PluginError.unavailable
        }
    }

    public func close() {
        closed = true
        sessions.removeAll()
    }

    private func appendU32(_ out: inout Data, _ value: UInt32) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { out.append(contentsOf: $0) }
    }

    private func readU32(_ data: Data, _ offset: inout Int) throws -> UInt32 {
        guard offset + 4 <= data.count else { throw PluginError.invalidArgument }
        let value = data.subdata(in: offset..<(offset + 4)).withUnsafeBytes {
            $0.load(as: UInt32.self).littleEndian
        }
        offset += 4
        return value
    }

    private func readString(_ data: Data, _ offset: inout Int) throws -> String {
        let len = Int(try readU32(data, &offset))
        guard len >= 0, offset + len <= data.count else { throw PluginError.invalidArgument }
        let slice = data.subdata(in: offset..<(offset + len))
        offset += len
        guard let str = String(data: slice, encoding: .utf8) else {
            throw PluginError.invalidArgument
        }
        return str
    }
}
