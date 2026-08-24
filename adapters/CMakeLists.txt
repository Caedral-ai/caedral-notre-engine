# Contains only integration glue — no product logic inside llama.cpp.
add_library(cne_adapter STATIC
    llama.cpp/cne_adapter.cpp
    stream_cb.cpp)
target_include_directories(cne_adapter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)
target_link_libraries(cne_adapter PUBLIC llama cne_core)
target_compile_features(cne_adapter PUBLIC cxx_std_17)
