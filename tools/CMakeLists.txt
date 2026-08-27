add_executable(cne_graph_probe graph-probe.cpp)
target_link_libraries(cne_graph_probe PRIVATE cne_core cne_runtime)

add_executable(cne_rebind_probe rebind-probe.cpp)
target_link_libraries(cne_rebind_probe PRIVATE cne_core cne_runtime)

add_executable(cne_identity_gate identity-gate.cpp)
target_link_libraries(cne_identity_gate PRIVATE cne_core cne_runtime)

add_executable(cne_touch_recorder touch-recorder.cpp)
target_link_libraries(cne_touch_recorder PRIVATE cne_core cne_runtime)

add_executable(cne_lru_sim lru-sim.cpp)
target_compile_features(cne_lru_sim PRIVATE cxx_std_17)

add_executable(cne_cb_semantics_probe cb-semantics-probe.cpp)
target_link_libraries(cne_cb_semantics_probe PRIVATE cne_core cne_runtime)

add_executable(cne_bench cne-bench.cpp)
target_link_libraries(cne_bench PRIVATE cne_core cne_runtime llama-common)

add_executable(cne_prepare cne-prepare.cpp)
target_link_libraries(cne_prepare PRIVATE cne_core)
