use std::collections::BTreeSet;

use pretraining_term_compiler::{
    compile_process, generate_processes, CompiledProcess, ProcessFamily, ProcessNodeKind,
    ProcessVisibility,
};

#[test]
fn compiler_emits_all_legal_structural_processes_through_one_schema() {
    let programs = generate_processes(0xA11CE, 16).expect("mixed process corpus compiles");
    assert_eq!(programs.len(), 16);
    let families: BTreeSet<_> = programs.iter().map(|item| item.process.family).collect();
    assert_eq!(families.len(), 8);
    for item in programs {
        item.process
            .validate()
            .expect("compiled process remains valid");
        assert_eq!(
            item.process.public_schema.schema,
            "trajectory-public-events-v2"
        );
        assert!(item.process.nodes.iter().all(|node| {
            !node.outputs.iter().any(|port| {
                port.visibility == ProcessVisibility::Public
                    && !matches!(
                        node.kind,
                        ProcessNodeKind::Input { .. } | ProcessNodeKind::Output { .. }
                    )
            })
        }));
    }
}

#[test]
fn generator_covers_each_legal_operator_composition() {
    let programs = generate_processes(7, 8).unwrap();
    let signatures: BTreeSet<_> = programs
        .iter()
        .map(|item| {
            (
                item.process.runtime_private.lag_alpha.is_some(),
                item.process.runtime_private.goal_switch.is_some(),
                item.process.runtime_private.disturbance.is_some(),
            )
        })
        .collect();
    assert_eq!(signatures.len(), 8);
    assert!(signatures.contains(&(false, false, false)));
    assert!(signatures.contains(&(true, true, true)));
}

#[test]
fn lag_and_composed_families_add_real_wiring_and_runtime_semantics() {
    let reaching = compile_process(ProcessFamily::Reaching, "r", 8, 2, 12).unwrap();
    let lag = compile_process(ProcessFamily::ActuatorLag, "l", 8, 2, 12).unwrap();
    let composed =
        compile_process(ProcessFamily::LagGoalSwitchDisturbance, "c", 10, 4, 18).unwrap();
    assert!(lag.nodes.len() > reaching.nodes.len());
    assert!(composed.nodes.len() > lag.nodes.len());
    assert!(lag.runtime_private.lag_alpha.is_some());
    assert!(composed.runtime_private.goal_switch.is_some());
    assert!(composed.runtime_private.disturbance.is_some());
    assert!(composed.wiring.iter().any(|wire| wire.from == "goal.out"));
    assert!(composed
        .wiring
        .iter()
        .any(|wire| wire.from == "disturbance.out"));
}

#[test]
fn lowering_reads_lag_node_and_disconnected_lag_is_rejected() {
    let mut process = compile_process(ProcessFamily::ActuatorLag, "lag", 8, 2, 12).unwrap();
    assert_eq!(
        process.lowered_runtime_private().unwrap().lag_alpha,
        Some(0.75)
    );
    process
        .nodes
        .iter_mut()
        .find(|node| node.id == "actuator_lag")
        .unwrap()
        .kind = ProcessNodeKind::ActuatorLag { alpha: 0.5 };
    let error = process
        .validate()
        .expect_err("serialized runtime must be regenerated after graph edits");
    assert!(error.to_string().contains("runtime_private does not match"));

    let mut disconnected = compile_process(ProcessFamily::ActuatorLag, "broken", 8, 2, 12).unwrap();
    disconnected
        .wiring
        .retain(|wire| wire.from != "actuator_lag.out");
    let error = disconnected
        .validate()
        .expect_err("lag output must feed the plant");
    assert!(error.to_string().contains("disconnected") || error.to_string().contains("input"));
}

#[test]
fn process_json_round_trip_keeps_runtime_private_separate_from_public_schema() {
    let process = compile_process(
        ProcessFamily::LagGoalSwitchDisturbance,
        "round-trip",
        10,
        4,
        18,
    )
    .unwrap();
    let json = serde_json::to_string(&process).unwrap();
    let decoded: CompiledProcess = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, process);
    assert!(!decoded
        .public_schema
        .channels
        .iter()
        .any(|name| name.contains("seed")));
    assert!(!decoded
        .public_schema
        .event_kinds
        .iter()
        .any(|name| name.contains("private")));
    assert!(json.contains("runtime_private"));
    assert!(!json.contains("generator_seed"));
}

#[test]
fn validator_rejects_private_to_public_wiring() {
    let mut process = compile_process(ProcessFamily::Reaching, "bad", 8, 2, 12).unwrap();
    process
        .nodes
        .iter_mut()
        .find(|node| node.id == "plant")
        .unwrap()
        .outputs[0]
        .visibility = ProcessVisibility::Public;
    let error = process
        .validate()
        .expect_err("private state cannot flow to a public input");
    assert!(error.to_string().contains("private process output leaks"));
}
