# SliceCache unit tests.
add_executable(soe_test_cache test_cache.cpp)
target_link_libraries(soe_test_cache PRIVATE soe_core)
add_test(NAME cache_basic COMMAND soe_test_cache)
