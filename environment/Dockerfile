# hash:sha256:7272751484459e473e6521e6b34ba495efdc23790f22eead6e5e9b7a4e84e3e5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV POLARS_MAX_THREADS=1
ENV TOKIO_WORKER_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV RAYON_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema-models==0.5.7 \
    lazynwb==0.2.39 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@e1a5608c60a03e74da98807f4185fb2a9b21f40c#egg=dynamic-routing-analysis
