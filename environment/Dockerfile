# hash:sha256:05d3773f8bd4eb79d2d5accc5f7cd7c13431949b6781254fe46f42bb5c10128e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

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
    lazynwb==0.2.22 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@c3de0db77289e009c385a74fbf3ae8e8d67e60b3#egg=dynamic-routing-analysis
