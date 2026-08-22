add_library(soe_core STATIC
    gguf/gguf.cpp
    gguf/classify.cpp
    gguf/model_registry.cpp
    cache/cache.cpp
)
target_include_directories(soe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_features(soe_core PUBLIC cxx_std_17)
