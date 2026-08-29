use std::env;
use std::error::Error;
use std::fs;

use pretraining_goal_conditioned_world::{
    classify_progress, CheckpointEvidence, ProgressThresholds,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if !(args.len() == 3 || args.len() == 4) {
        return Err(
            "usage: classify-progress <previous-evidence.json> <candidate-evidence.json> [thresholds.json]"
                .into(),
        );
    }

    let previous: CheckpointEvidence = read_json(&args[1])?;
    let candidate: CheckpointEvidence = read_json(&args[2])?;
    let thresholds = if args.len() == 4 {
        read_json(&args[3])?
    } else {
        ProgressThresholds::default()
    };
    let decision = classify_progress(&previous, &candidate, &thresholds);
    println!("{}", serde_json::to_string_pretty(&decision)?);
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}
