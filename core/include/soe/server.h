#pragma once
// HTTP/OpenAI-compatible server surface (soe-server). Kept separate from the
// core runtime: an API layer over an unstable runtime only masks the problem.
#include "soe/config.h"

namespace soe {

// Endpoints (V1):
//   GET  /v1/models
//   POST /v1/chat/completions   (SSE streaming)
//   POST /v1/completions
//   GET  /health
//   GET  /metrics
class Server {
public:
    explicit Server(Config cfg);
};

} // namespace soe
