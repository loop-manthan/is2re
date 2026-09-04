# IS2RE demo backend image.
#
# The ML stack (torch, fairchem-core, torch-geometric) is heavy, so this image is
# intentionally CPU-only and does NOT bundle the data artifacts. Mount them at
# runtime:
#
#   docker build -f app/Dockerfile -t is2re-demo .
#   docker run -p 8000:8000 \
#     -v $PWD/data:/workspace/data \
#     -v $PWD/runs:/workspace/runs \
#     -v $PWD/.cache:/workspace/.cache \
#     is2re-demo
#
# CPU inference on a single structure takes on the order of a second, which is fine
# for the demo. Note: this Dockerfile was written but not build-tested in CI.

FROM python:3.11-slim

WORKDIR /workspace

# Compiler toolchain: torch-scatter / torch-sparse no longer ship prebuilt wheels,
# so they are built from source against CPU-only torch (no CUDA, plain C++ build).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so fairchem-core (torch~=2.4.0) resolves against it.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir fairchem-core==1.10.0

# Try prebuilt wheels, fall back to building from source with the toolchain above.
RUN pip install --no-cache-dir torch-scatter==2.1.2 torch-sparse==0.6.18 \
    || pip install --no-cache-dir torch-scatter==2.1.2 torch-sparse==0.6.18 --no-binary :all:

RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Code only; large artifacts are mounted at runtime (see comment above).
COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/
COPY configs/ configs/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]