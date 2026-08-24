# GGUF parser + manifest validation tests.
add_executable(cne_test_gguf test_gguf_parse.cpp)
target_link_libraries(cne_test_gguf PRIVATE cne_core)
add_test(NAME gguf_parse COMMAND cne_test_gguf)
