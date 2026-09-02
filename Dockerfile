# syntax=docker/dockerfile:1
# CNE engine image: cne_server (+ cne_prepare, cne_setup). Mount models/ at runtime.
#
# Build:
#   git submodule update --init --recursive
#   docker build -t cne-engine .
#
# Run (config + weights on a volume):
#   docker run --rm -v "$PWD/models:/models:ro" -p 8080:8080 cne-engine

ARG DEBIAN_VERSION=bookworm
FROM debian:${DEBIAN_VERSION}-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCNE_BUILD_SERVER=ON \
      -DCNE_BUILD_CLI=ON \
      -DCNE_BUILD_TOOLS=ON \
      -DCNE_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" \
      --target cne_server cne_prepare cne_setup \
    && mkdir -p /out/lib /out/bin \
    && find build/bin -maxdepth 1 -name 'lib*.so*' -exec cp -a {} /out/lib/ \; \
    && cp build/server/cne_server build/tools/cne_prepare build/cli/cne_setup /out/bin/

FROM debian:${DEBIAN_VERSION}-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/lib/ /usr/local/lib/
COPY --from=builder /out/bin/cne_server /usr/local/bin/cne_server
COPY --from=builder /out/bin/cne_prepare /usr/local/bin/cne_prepare
COPY --from=builder /out/bin/cne_setup /usr/local/bin/cne_setup
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh && ldconfig

WORKDIR /models
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:8080/health" || exit 1

# Bind all interfaces inside the container; override host/port via argv if needed.
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/models/server.json"]
