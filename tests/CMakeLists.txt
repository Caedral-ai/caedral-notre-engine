cmake_minimum_required(VERSION 3.16)

if(SOE_BUILD_TESTS AND NOT CMAKE_VERSION VERSION_LESS 3.10)
    enable_testing()
    add_subdirectory(gguf)
    add_subdirectory(cache)
    add_subdirectory(io)
    add_subdirectory(memory)
endif()
