fn main() {
    let audit = pretraining_goal_conditioned_world::reference_audit();
    println!(
        "{}",
        serde_json::to_string_pretty(&audit).expect("reference audit is serializable")
    );
}
