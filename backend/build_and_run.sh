SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/encs_rust"
PYTHON_BIN_DEFAULT="/home/smayan/miniconda3/envs/data311/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi
if [[ -f "$SCRIPT_DIR/libittnotify_stub.so" ]]; then
    export LD_PRELOAD="$SCRIPT_DIR/libittnotify_stub.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
TMP_CACHE_ROOT="$(mktemp -d -t encs_build_cache.XXXXXXXX)"
cleanup() {
    if [[ "${KEEP_TEMP_CACHE:-0}" != "1" ]]; then
        rm -rf "$TMP_CACHE_ROOT"
    else
        echo "  ! KEEP_TEMP_CACHE=1 set; leaving temp cache at: $TMP_CACHE_ROOT" >&2
    fi
}
trap cleanup EXIT
export XDG_CACHE_HOME="$TMP_CACHE_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$TMP_CACHE_ROOT/xdg_config"
export XDG_DATA_HOME="$TMP_CACHE_ROOT/xdg_data"
export PIP_CACHE_DIR="$TMP_CACHE_ROOT/pip_cache"
export CARGO_HOME="$TMP_CACHE_ROOT/cargo_home"
export RUSTUP_HOME="$TMP_CACHE_ROOT/rustup_home"
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="$RUST_DIR/target"
if ! command -v rustc &> /dev/null; then
    echo "  ✗ Rust not found. Not installing automatically (avoids writing to HOME caches)." >&2
    echo "    Install Rust yourself, then re-run. Suggested: https://rustup.rs" >&2
    exit 1
else
    echo "  ✓ Rust $(rustc --version | awk '{print $2}')"
fi
if ! command -v maturin &> /dev/null; then
    echo "  Installing maturin..."
    "$PYTHON_BIN" -m pip install --no-cache-dir maturin
else
    echo "  ✓ maturin $(maturin --version | awk '{print $2}')"
fi
echo ""
echo "[2/4] Building Rust extension (encs_rust)..."
cd "$RUST_DIR"
"$PYTHON_BIN" -m maturin develop --release
cd "$SCRIPT_DIR"
echo ""
echo "[3/4] Verifying Python can import encs_rust..."
"$PYTHON_BIN" -c "
import encs_rust
print('  ✓ encs_rust imported successfully')
print('  Functions:', [x for x in dir(encs_rust) if not x.startswith('_')])
"
echo ""
echo "════════════════════════════════════════════════════════════"
if "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m streamlit run "$SCRIPT_DIR/dashboard.py"
else
    streamlit run "$SCRIPT_DIR/dashboard.py"
fi