add_library(cne_e2e_config STATIC e2e_config.cpp)
target_include_directories(cne_e2e_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cne_e2e_config PUBLIC vendor::nlohmann)
target_compile_definitions(cne_e2e_config PUBLIC
    CNE_PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
