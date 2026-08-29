fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&pretraining_card06_perceptual_organization::audit_report())
            .expect("audit JSON")
    );
}
