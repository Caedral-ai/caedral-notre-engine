add_executable(test_session test_session.cpp)
target_link_libraries(test_session PRIVATE cne_runtime)
add_test(NAME session_lcp COMMAND test_session)

add_executable(test_session_kv test_session_kv.cpp)
target_link_libraries(test_session_kv PRIVATE cne_runtime)
add_test(NAME session_kv_live COMMAND test_session_kv)

add_executable(test_session_bigctx test_session_bigctx.cpp)
target_link_libraries(test_session_bigctx PRIVATE cne_runtime)
add_test(NAME session_bigctx_live COMMAND test_session_bigctx)
set_tests_properties(session_bigctx_live PROPERTIES
    LABELS "slow"
    TIMEOUT 1200)

# Run live session checks when the prepared LFM2 fixture is present.
set(CNE_SESSION_TEST_MODEL
    "${CMAKE_SOURCE_DIR}/models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf")
if(EXISTS "${CNE_SESSION_TEST_MODEL}")
    set_tests_properties(session_kv_live PROPERTIES
        ENVIRONMENT
        "CNE_TEST_MODEL=${CNE_SESSION_TEST_MODEL};CNE_STREAM=0;CNE_CTX=2048;CNE_CACHE_GIB=4;CNE_THREADS=4;CNE_SESSION_MAX=2")
    set_tests_properties(session_bigctx_live PROPERTIES
        ENVIRONMENT
        "CNE_TEST_MODEL=${CNE_SESSION_TEST_MODEL};CNE_STREAM=0;CNE_CTX=8192;CNE_CACHE_GIB=4;CNE_THREADS=4;CNE_SESSION_MAX=1;CNE_TEST_CTX=8192;CNE_TEST_PROMPT_TOK=7000")
endif()
