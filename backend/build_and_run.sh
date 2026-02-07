set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/encs_rust"


echo "[1/4] Checking prerequisites..."

if ! command -v rustc &> /dev/null; then
    echo "  ✗ Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "  ✓ Rust $(rustc --version | awk '{print $2}')"
fi

if ! command -v maturin &> /dev/null; then
    echo "  Installing maturin..."
    pip install maturin
else
    echo "  ✓ maturin $(maturin --version | awk '{print $2}')"
fi

echo ""
echo "[2/4] Building Rust extension (encs_rust)..."
cd "$RUST_DIR"
maturin develop --release
cd "$SCRIPT_DIR"

echo ""
echo "[3/4] Verifying Python can import encs_rust..."
python -c "
import encs_rust
print('  ✓ encs_rust imported successfully')
print('  Functions:', [x for x in dir(encs_rust) if not x.startswith('_')])
"

echo ""
echo "[4/4] Launching Streamlit dashboard..."
echo "════════════════════════════════════════════════════════════"
streamlit run "$SCRIPT_DIR/dashboard.py"

