add_executable(test_session test_session.cpp)
target_link_libraries(test_session PRIVATE cne_runtime)
add_test(NAME session_lcp COMMAND test_session)

add_executable(test_session_kv test_session_kv.cpp)
target_link_libraries(test_session_kv PRIVATE cne_runtime cne_e2e_config)
add_test(NAME session_kv_live COMMAND test_session_kv)
add_test(NAME session_kv_qwen_live COMMAND test_session_kv)

add_executable(test_session_bigctx test_session_bigctx.cpp)
target_link_libraries(test_session_bigctx PRIVATE cne_runtime cne_e2e_config)
add_test(NAME session_bigctx_live COMMAND test_session_bigctx)
set_tests_properties(session_bigctx_live PROPERTIES
    LABELS "slow"
    TIMEOUT 1200)

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/session_kv_live.json")
    set_tests_properties(session_kv_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/session_kv_live.json")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/session_kv_qwen_live.json")
    set_tests_properties(session_kv_qwen_live PROPERTIES
        LABELS "slow"
        TIMEOUT 1200
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/session_kv_qwen_live.json")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/session_bigctx_live.json")
    set_tests_properties(session_bigctx_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/session_bigctx_live.json")
endif()
