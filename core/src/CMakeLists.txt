add_library(cne_core STATIC
    gguf/gguf.cpp
    gguf/classify.cpp
    gguf/model_registry.cpp
    cache/cache.cpp
    io/direct_io.cpp
    io/io_scheduler.cpp
    memory/memory_budget.cpp
)
target_include_directories(cne_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_features(cne_core PUBLIC cxx_std_17)
