add_executable(test_server_live test_server_live.cpp)
target_compile_definitions(test_server_live PRIVATE
    "CNE_SERVER_EXE=\"$<TARGET_FILE:cne_server>\"")
target_link_libraries(test_server_live PRIVATE cne_e2e_config cpp-httplib
    vendor::nlohmann)
target_include_directories(test_server_live PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)

# Same binary; per-model JSON via CNE_E2E_CONFIG.
add_test(NAME server_e2e_live COMMAND test_server_live)
add_test(NAME server_e2e_qwen_live COMMAND test_server_live)
add_test(NAME server_e2e_lfm25_live COMMAND test_server_live)
set_tests_properties(server_e2e_live server_e2e_qwen_live server_e2e_lfm25_live PROPERTIES
    LABELS "slow"
    TIMEOUT 900)

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_e2e_live.json")
    set_tests_properties(server_e2e_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_e2e_live.json")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_e2e_qwen_live.json")
    set_tests_properties(server_e2e_qwen_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_e2e_qwen_live.json"
        TIMEOUT 1200)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_e2e_lfm25_live.json")
    set_tests_properties(server_e2e_lfm25_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_e2e_lfm25_live.json"
        TIMEOUT 600)
endif()

add_executable(test_server_api_live test_server_api_live.cpp)
target_compile_definitions(test_server_api_live PRIVATE
    "CNE_SERVER_EXE=\"$<TARGET_FILE:cne_server>\"")
target_link_libraries(test_server_api_live PRIVATE cne_e2e_config cpp-httplib
    vendor::nlohmann)
target_include_directories(test_server_api_live PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)

add_test(NAME server_api_live COMMAND test_server_api_live)
set_tests_properties(server_api_live PROPERTIES LABELS "slow" TIMEOUT 900)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_api_live.json")
    set_tests_properties(server_api_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_api_live.json")
endif()

add_executable(test_server_api_per_user_live test_server_api_per_user_live.cpp)
target_compile_definitions(test_server_api_per_user_live PRIVATE
    "CNE_SERVER_EXE=\"$<TARGET_FILE:cne_server>\""
    "CNE_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_server_api_per_user_live PRIVATE cne_e2e_config
    cpp-httplib vendor::nlohmann)
target_include_directories(test_server_api_per_user_live PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)

add_test(NAME server_api_per_user_live COMMAND test_server_api_per_user_live)
set_tests_properties(server_api_per_user_live PROPERTIES LABELS "slow" TIMEOUT 300)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_api_per_user_live.json")
    set_tests_properties(server_api_per_user_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_api_per_user_live.json")
endif()

add_executable(test_server_gateway_live test_server_gateway_live.cpp)
target_compile_definitions(test_server_gateway_live PRIVATE
    "CNE_SERVER_EXE=\"$<TARGET_FILE:cne_server>\""
    "CNE_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_server_gateway_live PRIVATE cne_e2e_config cpp-httplib
    vendor::nlohmann)
target_include_directories(test_server_gateway_live PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)

add_test(NAME server_gateway_live COMMAND test_server_gateway_live)
set_tests_properties(server_gateway_live PROPERTIES LABELS "slow" TIMEOUT 1200)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_gateway_live.json")
    set_tests_properties(server_gateway_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_gateway_live.json")
endif()

add_executable(test_server_gateway_policy_live test_server_gateway_policy_live.cpp)
target_compile_definitions(test_server_gateway_policy_live PRIVATE
    "CNE_SERVER_EXE=\"$<TARGET_FILE:cne_server>\""
    "CNE_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(test_server_gateway_policy_live PRIVATE cne_e2e_config
    cpp-httplib vendor::nlohmann)
target_include_directories(test_server_gateway_policy_live PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)

add_test(NAME server_gateway_policy_live COMMAND test_server_gateway_policy_live)
set_tests_properties(server_gateway_policy_live PROPERTIES LABELS "slow" TIMEOUT 1200)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/e2e/server_gateway_policy_live.json")
    set_tests_properties(server_gateway_policy_live PROPERTIES
        ENVIRONMENT
        "CNE_E2E_CONFIG=${CMAKE_SOURCE_DIR}/tests/e2e/server_gateway_policy_live.json")
endif()
