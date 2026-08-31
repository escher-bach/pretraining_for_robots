fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&pretraining_card04a_goal_use::audit_report())
            .expect("audit JSON")
    );
}
