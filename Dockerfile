# --- Base Image ---
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - libfontconfig1 and others required by Tectonic
# - wget for downloading tectonic
# - build-essential for compiling some python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libfontconfig1 \
    libgraphite2-3 \
    libharfbuzz0b \
    libicu-dev \
    libpng16-16 \
    libssl-dev \
    zlib1g \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Tectonic (LaTeX engine)
RUN wget https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
    && tar -xzf tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
    && mv tectonic /usr/local/bin/ \
    && rm tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz

# Pre-warm Tectonic (downloads standard bundles)
RUN tectonic --version

# Copy the rest of the application
COPY . .

# Install the package itself
RUN pip install --no-cache-dir .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/app/output
ENV CHECKPOINT_BACKEND=sqlite
ENV CHECKPOINT_DB_URL=/app/output/checkpoints.db
ENV BACKEND_URL=http://localhost:8000

# Make output directory
RUN mkdir -p /app/output

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose ports
# 8000: FastAPI
# 8501: Streamlit
EXPOSE 8000 8501

# Entrypoint to run both backend and frontend
CMD ["./entrypoint.sh"]
