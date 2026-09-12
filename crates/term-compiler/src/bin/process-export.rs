//! Emit the stable process-program JSON consumed by the trajectory world.
//!
//! Usage: `cargo run -p pretraining-term-compiler --bin process-export --
//! --seed 123 --count 12 [--pretty]`

use std::env;

use pretraining_term_compiler::generate_processes;

fn main() {
    let mut seed = 0_u64;
    let mut count = 3_usize;
    let mut pretty = false;
    let args: Vec<String> = env::args().collect();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--seed" => {
                index += 1;
                seed = args
                    .get(index)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage("--seed needs an integer"));
            }
            "--count" => {
                index += 1;
                count = args
                    .get(index)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage("--count needs an integer"));
            }
            "--pretty" => pretty = true,
            "--help" | "-h" => usage(""),
            other => usage(&format!("unknown argument `{other}`")),
        }
        index += 1;
    }
    let programs = generate_processes(seed, count).unwrap_or_else(|error| {
        eprintln!("process generation failed: {error}");
        std::process::exit(1);
    });
    let output = if pretty {
        serde_json::to_string_pretty(&programs)
    } else {
        serde_json::to_string(&programs)
    }
    .expect("generated process programs serialize");
    println!("{output}");
}

fn usage(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("{message}");
    }
    eprintln!("usage: process-export [--seed INTEGER] [--count INTEGER] [--pretty]");
    std::process::exit(if message.is_empty() { 0 } else { 2 });
}
