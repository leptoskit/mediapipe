#!/usr/bin/env bash
# Install the Rust toolchain a cloud agent needs, then fetch dependencies
# when this checkout is a Cargo package. Safe to run more than once.
set -euo pipefail

if [[ -d /usr/local/cargo/bin ]]; then
  export PATH="/usr/local/cargo/bin:${PATH}"
fi

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal
  install_dir="${CARGO_HOME:-${HOME}/.cargo}/bin"
  sudo ln -sfn "${install_dir}/rustup" /usr/local/bin/rustup
  sudo ln -sfn "${install_dir}/cargo" /usr/local/bin/cargo
  sudo ln -sfn "${install_dir}/rustc" /usr/local/bin/rustc
  export PATH="${install_dir}:${PATH}"
fi

# rust-toolchain.toml, when present, selects the compiler. Otherwise pin the
# same 1.83.0 release CI uses, so an image without Rust still matches.
if [[ -f rust-toolchain.toml ]]; then
  rustup show
else
  rustup toolchain install 1.83.0 --profile minimal --component rustfmt,clippy
  rustup default 1.83.0
fi

rustup component add rustfmt clippy

if [[ -f Cargo.toml ]]; then
  if [[ -f Cargo.lock ]]; then
    cargo fetch --locked
  else
    cargo fetch
  fi
fi
