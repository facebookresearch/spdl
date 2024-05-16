# Usage: docker build --network=host -t test -f packaging/Dockerfile .

ARG CU_VERSION=12.1

FROM pytorch/manylinux2_28-builder:cuda${CU_VERSION} as builder

ARG CU_VERSION
ARG USE_CUDA=1
ARG USE_TRACING=1

ENV CUDACXX="/usr/local/cuda-${CU_VERSION}/bin/nvcc"
ENV SPDL_USE_CUDA="${USE_CUDA}"
ENV SPDL_USE_NVCODEC="${USE_CUDA}"
ENV SPDL_USE_TRACING="${USE_TRACING}"

# The default cmake is too old
RUN python3 -m pip uninstall -y cmake

WORKDIR /build

# Copy separately just because I want to copy audit_wheel.sh separately
# and the steps before that are cached.
COPY ./src/ ./src/
COPY ./cmake/ ./cmake/
COPY ./third_party/ ./third_party/
COPY ./setup.py ./CMakeLists.txt ./packaging/build_wheel.sh .

RUN ./build_wheel.sh cp310-cp310
RUN ./build_wheel.sh cp311-cp311
RUN ./build_wheel.sh cp312-cp312


FROM pytorch/manylinux2_28-builder:cuda${CU_VERSION}

WORKDIR /build

COPY --from=builder /build/dist /build/dist
COPY packaging/audit_wheel.sh .

RUN ./audit_wheel.sh
