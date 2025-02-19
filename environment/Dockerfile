# hash:sha256:59962f535c9c63a7994d0ca36fe20e3dbc038e0195f5e734040ee47786c57ba0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema-models==0.5.7 \
    code-ocean-aux-tools==1.1.3 \
    hdmf==3.14.6 \
    scikit-learn==1.5.2 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@5c065515d21f4d87910194b642469b865894b07b#egg=dynamic-routing-analysis
