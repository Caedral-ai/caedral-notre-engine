add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE cne_core cne_adapter)
add_test(NAME runtime_prepare COMMAND test_runtime)
