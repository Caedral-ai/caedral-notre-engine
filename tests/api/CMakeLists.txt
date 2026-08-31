add_executable(test_api test_api.cpp)
target_link_libraries(test_api PRIVATE cne_runtime)
add_test(NAME api_unit COMMAND test_api)
