//! Emit card 03's exact audit as JSON on standard output.

fn main() {
    let report = pretraining_card03_affordance::audit_report();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("the audit report serializes")
    );
}
