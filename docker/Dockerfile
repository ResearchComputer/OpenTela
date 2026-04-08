FROM ubuntu:22.04

ARG VERSION
ARG TARGETPLATFORM

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    curl && \
    case "$TARGETPLATFORM" in \
      "linux/arm64")  URL="https://github.com/eth-easl/OpenTela/releases/download/${VERSION}/opentela-arm64" ;; \
      *) URL="https://github.com/eth-easl/OpenTela/releases/download/${VERSION}/opentela-amd64" ;; \
    esac && \
    curl -L "$URL" > opentela && \
    chmod +x opentela

ENTRYPOINT ["/app/opentela"]