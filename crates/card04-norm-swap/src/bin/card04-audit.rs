//! Emit the exact audit for Card 04. No learner is constructed or run.

use pretraining_card04_norm_swap::audit_report;

fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&audit_report()).expect("the report serializes")
    );
}
