add_executable(test_boot test_runtime.cpp)
target_link_libraries(test_boot PRIVATE cne_core cne_runtime)
add_test(NAME boot_prepare COMMAND test_boot)
