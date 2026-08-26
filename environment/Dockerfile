# hash:sha256:6dcac95ac9a3037245d259ba131c16c49aafa8ba1b999096b4ef705947a9c449
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-data-schema-models==0.5.7 \
    lazynwb==0.2.40 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@db7cd9ecf315b5683502c53bc53750a57cbc1afb#egg=dynamic-routing-analysis
