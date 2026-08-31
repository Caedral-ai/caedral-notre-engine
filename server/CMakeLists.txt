# cne-server: OpenAI-compatible serving on top of the extracted runtime.
# Product core discipline: links cne_runtime (which brings llama + cne_core +
# llama-common for the MTP drafter); never modifies the llama.cpp tree.
add_executable(cne_server cne-server.cpp)
target_link_libraries(cne_server PRIVATE cne_runtime cpp-httplib vendor::nlohmann)
target_include_directories(cne_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/vendor/cpp-httplib)
