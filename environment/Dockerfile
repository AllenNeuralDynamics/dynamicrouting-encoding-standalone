# hash:sha256:9a264567606780190e6b42ad1cd4f13877c458141409542d26a713b3eb68395d
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
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@b5740eb24152c30dbd75e4d96d2651dc854571a5#egg=dynamic-routing-analysis
