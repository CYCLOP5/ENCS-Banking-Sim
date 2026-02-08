# ── Stage 1: Build Rust extension ─────────────────────────────────────
FROM python:3.11-slim AS rust-builder

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy Rust extension source
COPY encs_rust/ ./encs_rust/

# Install maturin and build wheel
RUN pip install maturin && \
    cd encs_rust && \
    maturin build --release --interpreter python3.11

# ── Stage 2: Final image ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch and scipy
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
# Use CPU-only PyTorch to reduce image size by ~1.5GB
COPY requirements.txt .
RUN grep -v maturin requirements.txt | grep -v torch > requirements-runtime.txt && \
    pip install --no-cache-dir -r requirements-runtime.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch_geometric

# Install FastAPI and uvicorn explicitly
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy pre-built Rust wheel from builder stage and install
COPY --from=rust-builder /build/encs_rust/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Create user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app

# Copy app files as user
COPY --chown=user:user *.py ./
COPY --chown=user:user site_knowledge.json ./
COPY --chown=user:user data/output/adjacency_matrix.npz ./data/output/
COPY --chown=user:user data/output/country_aggregates.csv ./data/output/
COPY --chown=user:user data/output/master_nodes.csv ./data/output/
COPY --chown=user:user data/output/network_map.json ./data/output/
COPY --chown=user:user gnn_model.pth ./
COPY --chown=user:user norm_stats.pt ./
COPY --chown=user:user deg_histogram.pt ./

# Expose port
EXPOSE 7860

# Run with uvicorn
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
