# Integration glue between cne_core and pinned llama.cpp — no product logic
# inside the fork beyond ggml-cpu compute hooks.
add_library(cne_runtime STATIC
    cne_runtime.cpp
    cne_api.cpp
    cne_session.cpp
    cne_stream_spec.cpp
    seam/cne_adapter.cpp
    cne_stream_cb.cpp)
target_include_directories(cne_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)
target_link_libraries(cne_runtime PUBLIC llama cne_core llama-common)
target_compile_features(cne_runtime PUBLIC cxx_std_17)
if(CNE_AUDIT)
    target_compile_definitions(cne_runtime PUBLIC CNE_AUDIT)
endif()
