# DirectFile, IoScheduler and SliceCache backend-mode tests.
add_executable(soe_test_direct_io test_direct_io.cpp)
target_link_libraries(soe_test_direct_io PRIVATE soe_core)
add_test(NAME direct_io COMMAND soe_test_direct_io)

add_executable(soe_test_io_scheduler test_io_scheduler.cpp)
target_link_libraries(soe_test_io_scheduler PRIVATE soe_core)
target_compile_features(soe_test_io_scheduler PRIVATE cxx_std_17)
add_test(NAME io_scheduler COMMAND soe_test_io_scheduler)
