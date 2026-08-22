add_executable(soe_graph_probe graph-probe.cpp)
target_link_libraries(soe_graph_probe PRIVATE soe_core soe_adapter)

add_executable(soe_rebind_probe rebind-probe.cpp)
target_link_libraries(soe_rebind_probe PRIVATE soe_core soe_adapter)

add_executable(soe_identity_gate identity-gate.cpp)
target_link_libraries(soe_identity_gate PRIVATE soe_core soe_adapter)
