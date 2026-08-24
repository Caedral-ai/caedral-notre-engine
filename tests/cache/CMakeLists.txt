# SliceCache unit tests.
add_executable(cne_test_cache test_cache.cpp)
target_link_libraries(cne_test_cache PRIVATE cne_core)
add_test(NAME cache_basic COMMAND cne_test_cache)
