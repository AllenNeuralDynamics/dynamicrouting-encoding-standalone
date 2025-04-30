# hash:sha256:33f15c0087b85324eca77233c5acc365d5ccf7e247ef3ed9a6e535647d26cdfc
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
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@19343fed4deaf76181d124a9807b0291f3265442#egg=dynamic-routing-analysis
