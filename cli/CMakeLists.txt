# cne-setup: first-run configuration CLI. Core-only by design: hardware and
# regime detection must work on a fresh machine before any model is loaded.
# nlohmann lives in the llama.cpp vendor tree; pull it in directly when the
# llama seam is disabled so CNE_BUILD_CLI stays standalone-buildable.
if(NOT TARGET vendor::nlohmann)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/vendor/nlohmann
                     ${CMAKE_BINARY_DIR}/vendor-nlohmann-standalone)
endif()
add_executable(cne_setup cne-setup.cpp)
target_link_libraries(cne_setup PRIVATE cne_core vendor::nlohmann)
