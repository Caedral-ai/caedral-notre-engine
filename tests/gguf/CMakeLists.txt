# GGUF parser + manifest validation tests.
add_executable(soe_test_gguf test_gguf_parse.cpp)
target_link_libraries(soe_test_gguf PRIVATE soe_core)
add_test(NAME gguf_parse COMMAND soe_test_gguf)
