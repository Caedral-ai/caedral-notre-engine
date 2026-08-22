# Contains only integration glue — no product logic inside llama.cpp.
add_library(soe_adapter STATIC llama.cpp/soe_adapter.cpp)
target_include_directories(soe_adapter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(soe_adapter PUBLIC llama)
target_compile_features(soe_adapter PUBLIC cxx_std_17)
