# Multi-stage build for smaller image
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for encs_rust compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements FIRST (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build Rust extension (maturin)
# If encs_rust is a local folder, build it.
RUN if [ -d "encs_rust" ]; then \
    pip install maturin; \
    maturin build --release --manifest-path encs_rust/Cargo.toml --out dist; \
    pip install dist/*.whl; \
    fi

# -------------------------------------------------------
# Runtime Stage
# -------------------------------------------------------
FROM python:3.9-slim

WORKDIR /app

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=user:user . .

# Expose port (HF Spaces defaults to 7860)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
