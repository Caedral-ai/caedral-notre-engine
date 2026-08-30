cmake_minimum_required(VERSION 3.16)

if(CNE_BUILD_TESTS AND NOT CMAKE_VERSION VERSION_LESS 3.10)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/gateway/tests/test_gateway.py")
        add_test(
            NAME gateway_unit
            COMMAND ${Python3_EXECUTABLE} -m pytest -q tests
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/gateway"
        )
        set_tests_properties(gateway_unit PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/gateway")
    endif()
    add_subdirectory(e2e)
    add_subdirectory(gguf)
    add_subdirectory(features/streaming)
    add_subdirectory(memory)
    add_subdirectory(boot)
    add_subdirectory(session)
    add_subdirectory(api)
    add_subdirectory(perplexity)
    if(TARGET cne_server)
        add_subdirectory(server)
    endif()
endif()
