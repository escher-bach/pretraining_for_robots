//! Emit card 02's exact audit as JSON on standard output.

fn main() {
    let report = pretraining_card02_predictive_state::audit_report();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("the audit report serializes")
    );
}
