# DirectFile + SliceCache backend-mode tests.
add_executable(soe_test_direct_io test_direct_io.cpp)
target_link_libraries(soe_test_direct_io PRIVATE soe_core)
add_test(NAME direct_io COMMAND soe_test_direct_io)
