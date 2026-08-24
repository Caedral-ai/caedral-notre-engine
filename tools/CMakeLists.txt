add_executable(soe_graph_probe graph-probe.cpp)
target_link_libraries(soe_graph_probe PRIVATE soe_core soe_adapter)

add_executable(soe_rebind_probe rebind-probe.cpp)
target_link_libraries(soe_rebind_probe PRIVATE soe_core soe_adapter)

add_executable(soe_identity_gate identity-gate.cpp)
target_link_libraries(soe_identity_gate PRIVATE soe_core soe_adapter)

add_executable(soe_touch_recorder touch-recorder.cpp)
target_link_libraries(soe_touch_recorder PRIVATE soe_core soe_adapter)

add_executable(soe_lru_sim lru-sim.cpp)
target_compile_features(soe_lru_sim PRIVATE cxx_std_17)

add_executable(soe_cb_semantics_probe cb-semantics-probe.cpp)
target_link_libraries(soe_cb_semantics_probe PRIVATE soe_core soe_adapter)

add_executable(soe_streaming_bench streaming-bench.cpp)
target_link_libraries(soe_streaming_bench PRIVATE soe_core soe_adapter llama-common)

add_executable(soe_prepare soe-prepare.cpp)
target_link_libraries(soe_prepare PRIVATE soe_core)
