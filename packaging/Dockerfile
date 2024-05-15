ARG platform=linux/amd64

FROM --platform=${platform} continuumio/miniconda3

WORKDIR /build
COPY . .

ARG USE_CUDA=1
ARG USE_TRACING=1
ARG python_version=3.10

RUN ./packaging/build_pypi.sh "${python_version}" "${USE_CUDA}" "${USE_TRACING}"
