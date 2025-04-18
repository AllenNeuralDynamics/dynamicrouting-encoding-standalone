# hash:sha256:9ecfa902b541854fa45651310727558474b2d30c369ed53bcfd1d14a823b5da0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

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
    pynwb==3.0.0 \
    scikit-learn==1.5.2 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@2fcda8fbf3f58886de65ffd7d100b8a6170fa68a#egg=dynamic-routing-analysis
