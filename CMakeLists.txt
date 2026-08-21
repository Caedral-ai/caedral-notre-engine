cmake_minimum_required(VERSION 3.16)
project(caedral-notre-engine VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SOE_BUILD_TESTS "Build test suite" ON)
option(SOE_BUILD_SERVER "Build soe-server" OFF)
option(SOE_BUILD_CLI "Build CLI tools" OFF)

add_library(soe INTERFACE)
target_include_directories(soe INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

if(SOE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SOE_BUILD_SERVER)
    add_subdirectory(server)
endif()

if(SOE_BUILD_CLI)
    add_subdirectory(cli)
endif()
