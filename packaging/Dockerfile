ARG CU_VERSION=11.8

FROM pytorch/manylinux2_28-builder:cuda${CU_VERSION}

ARG CU_VERSION
ARG USE_CUDA=1
ARG USE_TRACING=1

ENV CUDACXX="/usr/local/cuda-${CU_VERSION}/bin/nvcc"
ENV SPDL_USE_CUDA="${USE_CUDA}"
ENV SPDL_USE_NVCODEC="${USE_CUDA}"
ENV SPDL_USE_TRACING="${USE_TRACING}"


# The default cmake is too old
RUN python3 -m pip uninstall -y cmake

RUN groupadd -g 1000 builder
RUN useradd -m -u 1000 -g builder -s /bin/bash builder
USER builder
ENV PATH="/home/builder/.local/bin/:${PATH}"

# Install newer cmake and ninja.
RUN /opt/python/cp310-cp310/bin/pip3 install -U cmake ninja

WORKDIR /build

COPY --chown=builder:builder . .

RUN ./packaging/build_wheel.sh cp310-cp310
RUN ./packaging/build_wheel.sh cp311-cp311
RUN ./packaging/build_wheel.sh cp312-cp312
RUN ./packaging/audit_wheel.sh
