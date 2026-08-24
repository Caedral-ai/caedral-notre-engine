cmake_minimum_required(VERSION 3.16)
project(caedral-notre-engine VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SOE_BUILD_TESTS "Build test suite" ON)
option(SOE_BUILD_SERVER "Build soe-server" OFF)
option(SOE_BUILD_CLI "Build CLI tools" OFF)
option(SOE_BUILD_TOOLS "Build developer tools (graph probe)" ON)
option(SOE_WITH_LLAMA "Build pinned llama.cpp and the adapter seam" ON)

add_library(soe INTERFACE)
target_include_directories(soe INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

add_subdirectory(core/src)

if(SOE_WITH_LLAMA)
    # Pinned upstream (see third_party/llama.cpp). Build only libllama:
    # no examples, tests, server or UI. Fork discipline: product logic
    # never goes inside this subproject.
    set(LLAMA_BUILD_EXAMPLES OFF)
    set(LLAMA_BUILD_TESTS    OFF)
    set(LLAMA_BUILD_TOOLS    OFF)
    set(LLAMA_BUILD_SERVER   OFF)
    set(LLAMA_BUILD_APP      OFF)
    set(LLAMA_BUILD_UI       OFF)
    # `common` is needed by tools that wire upstream speculative decoding
    # (common_speculative draft-mtp). Library-only discipline for the product
    # core is unchanged: soe_core/soe_adapter never link it.
    set(LLAMA_BUILD_COMMON   ON)
    add_subdirectory(third_party/llama.cpp)
    add_subdirectory(adapters)
endif()

if(SOE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SOE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(SOE_BUILD_SERVER)
    add_subdirectory(server)
endif()

if(SOE_BUILD_CLI)
    add_subdirectory(cli)
endif()
