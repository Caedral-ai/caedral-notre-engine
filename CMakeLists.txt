cmake_minimum_required(VERSION 3.16)
project(caedral-notre-engine VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CNE_BUILD_TESTS "Build test suite" ON)
option(CNE_BUILD_SERVER "Build cne-server" OFF)
option(CNE_BUILD_CLI "Build CLI tools" OFF)
option(CNE_BUILD_TOOLS "Build developer tools (graph probe)" ON)
option(CNE_WITH_LLAMA "Build pinned llama.cpp and the adapter seam" ON)

add_library(cne INTERFACE)
target_include_directories(cne INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

add_subdirectory(core/src)

if(CNE_WITH_LLAMA)
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
    # core is unchanged: cne_core/cne_adapter never link it.
    set(LLAMA_BUILD_COMMON   ON)
    add_subdirectory(third_party/llama.cpp)
    add_subdirectory(adapters)
endif()

if(CNE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CNE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(CNE_BUILD_SERVER)
    add_subdirectory(server)
endif()

if(CNE_BUILD_CLI)
    add_subdirectory(cli)
endif()
