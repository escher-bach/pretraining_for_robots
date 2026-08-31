fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&pretraining_card03a_body_identification::audit_report())
            .expect("audit JSON")
    );
}
