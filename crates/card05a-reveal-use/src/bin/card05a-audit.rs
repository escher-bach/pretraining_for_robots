fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&pretraining_card05a_reveal_use::audit_report())
            .expect("audit JSON")
    );
}
