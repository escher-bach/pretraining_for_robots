//! Generate a replayable JSON artifact of finite G0 worlds.
//!
//! The binary is intentionally a thin adapter around [`generate_corpus`].
//! Generation, semantic admission, and rendering remain library behaviour so
//! callers get the same checks whether they use the CLI or Rust directly.

use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use pretraining_g0_corpus::{generate_corpus, CorpusArtifact};
use pretraining_term_compiler::{
    G0CorpusSpec, G0FeatureSchedule, G0Topology, PlanLengthDistribution,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TopologyArg {
    Chain,
    Star,
    Tree,
    Dag,
    Grid,
    Lattice,
    Ring,
    Graph,
    Mixed,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FeatureScheduleArg {
    Plain,
    Coverage,
}

impl From<FeatureScheduleArg> for G0FeatureSchedule {
    fn from(schedule: FeatureScheduleArg) -> Self {
        match schedule {
            FeatureScheduleArg::Plain => Self::Plain,
            FeatureScheduleArg::Coverage => Self::Coverage,
        }
    }
}

impl From<TopologyArg> for G0Topology {
    fn from(topology: TopologyArg) -> Self {
        match topology {
            TopologyArg::Chain => Self::Chain,
            TopologyArg::Star => Self::Star,
            TopologyArg::Tree => Self::Tree,
            TopologyArg::Dag => Self::Dag,
            TopologyArg::Grid => Self::Grid,
            TopologyArg::Lattice => Self::Lattice,
            TopologyArg::Ring => Self::Ring,
            TopologyArg::Graph => Self::Graph,
            TopologyArg::Mixed => Self::Mixed,
        }
    }
}

fn parse_plan_length_weights(value: &str) -> Result<PlanLengthDistribution, String> {
    let weights = value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<u64>()
                .map_err(|error| format!("invalid plan-length weight {item:?}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    PlanLengthDistribution::new(weights).map_err(|error| error.to_string())
}

#[derive(Debug, Parser)]
#[command(
    name = "g0-corpus",
    about = "Generate a solver-verified finite G0 corpus artifact"
)]
struct Args {
    /// Destination JSON file.
    #[arg(short, long, default_value = "g0-corpus.json", value_name = "PATH")]
    output: PathBuf,

    /// Seed for generated training worlds.
    #[arg(long, default_value_t = 0)]
    train_seed: u64,

    /// Seed for generated held-out worlds.
    #[arg(long, default_value_t = 1)]
    held_out_seed: u64,

    /// Number of training worlds to generate.
    #[arg(long, default_value_t = 16, value_name = "COUNT")]
    train_count: usize,

    /// Number of held-out worlds to generate.
    #[arg(long, default_value_t = 16, value_name = "COUNT")]
    held_out_count: usize,

    /// Topology distribution shared by both splits.
    #[arg(long, value_enum, default_value_t = TopologyArg::Mixed)]
    topology: TopologyArg,

    /// Feature arm schedule for generated worlds.
    #[arg(long, value_enum, default_value_t = FeatureScheduleArg::Coverage)]
    feature_schedule: FeatureScheduleArg,

    /// Nonnegative weights for exact plan lengths 1..=N, as a comma-delimited list.
    /// At least one weight must be positive; zero permits sparse support.
    #[arg(
        long,
        value_parser = parse_plan_length_weights,
        default_value = "1,1,1,1,1,1",
        value_name = "W1,W2,..."
    )]
    plan_length_weights: PlanLengthDistribution,

    /// Inclusive lower bound for world state counts, shared by both splits.
    #[arg(long, default_value_t = 4, value_name = "COUNT")]
    min_states: usize,

    /// Inclusive upper bound for world state counts, shared by both splits.
    #[arg(long, default_value_t = 8, value_name = "COUNT")]
    max_states: usize,

    /// Inclusive lower bound for episode horizons, shared by both splits.
    #[arg(long, default_value_t = 2, value_name = "STEPS")]
    min_horizon: usize,

    /// Inclusive upper bound for episode horizons, shared by both splits.
    #[arg(long, default_value_t = 6, value_name = "STEPS")]
    max_horizon: usize,
}

impl Args {
    fn specs(&self) -> (G0CorpusSpec, G0CorpusSpec) {
        let common = || G0CorpusSpec {
            seed: 0,
            count: 0,
            topology: self.topology.into(),
            feature_schedule: self.feature_schedule.into(),
            plan_length_distribution: self.plan_length_weights.clone(),
            min_states: self.min_states,
            max_states: self.max_states,
            min_horizon: self.min_horizon,
            max_horizon: self.max_horizon,
        };
        let mut train = common();
        train.seed = self.train_seed;
        train.count = self.train_count;
        let mut held_out = common();
        held_out.seed = self.held_out_seed;
        held_out.count = self.held_out_count;
        (train, held_out)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Summary {
    train_worlds: usize,
    held_out_worlds: usize,
    examples: usize,
    tokens: usize,
    distinct_hashes: usize,
    distinct_fingerprints: usize,
    plan_lengths: std::collections::BTreeMap<usize, usize>,
}

impl Summary {
    fn from_artifact(artifact: &CorpusArtifact) -> Self {
        let mut hashes = std::collections::BTreeSet::new();
        let mut fingerprints = std::collections::BTreeSet::new();
        let mut plan_lengths = std::collections::BTreeMap::new();
        let mut tokens = 0;
        for example in &artifact.examples {
            hashes.insert(&example.semantic_hash);
            fingerprints.insert(example.fingerprint);
            *plan_lengths.entry(example.plan_length).or_insert(0) += 1;
            tokens += example.tokens.len();
        }
        Self {
            train_worlds: artifact.train().count(),
            held_out_worlds: artifact.held_out().count(),
            examples: artifact.examples.len(),
            tokens,
            distinct_hashes: hashes.len(),
            distinct_fingerprints: fingerprints.len(),
            plan_lengths,
        }
    }

    fn print(&self, output: &Path) {
        let plan_lengths = self
            .plan_lengths
            .iter()
            .map(|(length, count)| format!("{length}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "wrote {} | train_worlds={} held_out_worlds={} examples={} tokens={} distinct_hashes={} distinct_fingerprints={} plan_lengths={}",
            output.display(),
            self.train_worlds,
            self.held_out_worlds,
            self.examples,
            self.tokens,
            self.distinct_hashes,
            self.distinct_fingerprints,
            plan_lengths,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let (train_spec, held_out_spec) = args.specs();
    let artifact = generate_corpus(train_spec, held_out_spec)?;
    // Serialize before opening the destination. A serialization error thus
    // cannot truncate or leave a partial existing artifact.
    let bytes = serde_json::to_vec_pretty(&artifact)?;
    write_atomically(&args.output, &bytes)?;
    Summary::from_artifact(&artifact).print(&args.output);
    Ok(())
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output path has no file name: {}", path.display()))?
        .to_string_lossy();

    // create_new keeps concurrent invocations from sharing a temporary file.
    // A bounded suffix loop also handles a stale temp file from an interrupted
    // prior invocation without touching it.
    let mut temp = None;
    for suffix in 0..1000u32 {
        let candidate = parent.join(format!(".{file_name}.tmp-{}-{suffix}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
                    drop(file);
                    let _ = fs::remove_file(&candidate);
                    return Err(error.into());
                }
                temp = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let (temp_path, file) = temp.ok_or_else(|| {
        format!(
            "could not allocate a temporary output beside {}",
            path.display()
        )
    })?;
    drop(file);

    if let Err(error) = fs::rename(&temp_path, path) {
        // Unix rename replaces an existing file; Windows requires the target
        // to be removed first. The serialized bytes are already durable in the
        // sibling temp file, so a failed rename cannot expose a partial file.
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            fs::remove_file(path)?;
            fs::rename(&temp_path, path)?;
        } else {
            let _ = fs::remove_file(&temp_path);
            return Err(error.into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn argument_defaults_map_to_two_specs() {
        let args = Args::try_parse_from(["g0-corpus"]).expect("defaults");
        let (train, held_out) = args.specs();
        assert_eq!(train, G0CorpusSpec::default());
        assert_eq!(held_out.seed, 1);
        assert_eq!(held_out.count, 16);
        assert_eq!(held_out.topology, G0Topology::Mixed);
        assert_eq!(held_out.feature_schedule, G0FeatureSchedule::Coverage);
        assert_eq!(
            held_out.plan_length_distribution,
            PlanLengthDistribution {
                weights: vec![1; 6]
            }
        );
    }

    #[test]
    fn argument_ranges_and_split_values_map_without_crossing() {
        let args = Args::try_parse_from([
            "g0-corpus",
            "--train-seed",
            "7",
            "--held-out-seed",
            "9",
            "--train-count",
            "3",
            "--held-out-count",
            "5",
            "--topology",
            "graph",
            "--feature-schedule",
            "plain",
            "--plan-length-weights",
            "1,2,3",
            "--min-states",
            "5",
            "--max-states",
            "9",
            "--min-horizon",
            "2",
            "--max-horizon",
            "7",
        ])
        .expect("arguments");
        let (train, held_out) = args.specs();
        assert_eq!(train.seed, 7);
        assert_eq!(held_out.seed, 9);
        assert_eq!(train.count, 3);
        assert_eq!(held_out.count, 5);
        assert_eq!(train.topology, G0Topology::Graph);
        assert_eq!(held_out.topology, G0Topology::Graph);
        assert_eq!(train.feature_schedule, G0FeatureSchedule::Plain);
        assert_eq!(
            train.plan_length_distribution,
            PlanLengthDistribution {
                weights: vec![1, 2, 3]
            }
        );
        assert_eq!((train.min_states, train.max_states), (5, 9));
        assert_eq!((held_out.min_horizon, held_out.max_horizon), (2, 7));
    }

    #[test]
    fn writer_replaces_target_only_after_serialized_bytes_are_ready() {
        let root = std::env::temp_dir().join(format!(
            "g0-corpus-cli-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("temp dir");
        let output = root.join("corpus.json");
        fs::write(&output, b"old").expect("old artifact");
        write_atomically(&output, br#"{"schema_version":1}"#).expect("write");
        assert_eq!(
            fs::read(&output).expect("artifact"),
            br#"{"schema_version":1}"#
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn every_topology_name_maps_to_the_generator_enum() {
        let expected = [
            ("chain", G0Topology::Chain),
            ("star", G0Topology::Star),
            ("tree", G0Topology::Tree),
            ("dag", G0Topology::Dag),
            ("grid", G0Topology::Grid),
            ("lattice", G0Topology::Lattice),
            ("ring", G0Topology::Ring),
            ("graph", G0Topology::Graph),
            ("mixed", G0Topology::Mixed),
        ];
        for (name, topology) in expected {
            let args = Args::try_parse_from(["g0-corpus", "--topology", name]).expect("topology");
            assert_eq!(args.specs().0.topology, topology);
        }
    }

    #[test]
    fn plan_length_weights_allow_sparse_support_but_not_zero_mass() {
        let args = Args::try_parse_from(["g0-corpus", "--plan-length-weights", "1,0,2"])
            .expect("sparse weights");
        assert_eq!(
            args.specs().0.plan_length_distribution.weights,
            vec![1, 0, 2]
        );
        assert!(Args::try_parse_from(["g0-corpus", "--plan-length-weights", "0,0"]).is_err());
    }
}
