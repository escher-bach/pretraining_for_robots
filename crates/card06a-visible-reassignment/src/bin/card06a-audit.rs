fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&pretraining_card06a_visible_reassignment::audit_report())
            .expect("audit JSON")
    );
}
