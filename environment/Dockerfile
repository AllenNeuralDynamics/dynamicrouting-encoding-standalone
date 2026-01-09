# hash:sha256:4f7cf9e6890006426cf2fe850c58f62ecd34963a41d7b2d497337390ebfa0fe1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-data-schema-models==0.5.7 \
    lazynwb==0.2.40 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@f6b0e1b437635c77ccba73ca8526529f8e6553f4#egg=dynamic-routing-analysis
