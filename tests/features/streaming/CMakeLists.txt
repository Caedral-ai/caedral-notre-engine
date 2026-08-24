# SliceCache unit tests.
add_executable(cne_test_cache test_cache.cpp)
target_link_libraries(cne_test_cache PRIVATE cne_core)
add_test(NAME cache_basic COMMAND cne_test_cache)

add_executable(cne_test_direct_io test_direct_io.cpp)
target_link_libraries(cne_test_direct_io PRIVATE cne_core)
add_test(NAME direct_io COMMAND cne_test_direct_io)

add_executable(cne_test_io_scheduler test_io_scheduler.cpp)
target_link_libraries(cne_test_io_scheduler PRIVATE cne_core)
target_compile_features(cne_test_io_scheduler PRIVATE cxx_std_17)
add_test(NAME io_scheduler COMMAND cne_test_io_scheduler)
