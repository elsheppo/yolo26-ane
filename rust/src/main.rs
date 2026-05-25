//! yolo26-ane: Rust wrapper for YOLO26 on Apple Neural Engine
//!
//! Calls the Swift CoreML CLI (yolo26-coreml) via subprocess.
//!
//! Architecture:
//!   Rust binary → subprocess(yolo26-coreml) → CoreML → ANE → JSON → Rust parses
//!
//! For benchmark results, see the repository README and results artifacts.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// Detection result from YOLO26
#[derive(Debug, Serialize, Deserialize)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub conf: f32,
    #[serde(rename = "class")]
    pub class_id: i32,
}

/// Benchmark result
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchResult {
    pub fps: f32,
    pub mean_ms: f32,
    pub std_ms: f32,
    pub median_ms: f32,
    pub p50_ms: f32,
    pub p99_ms: f32,
    pub min_ms: f32,
    pub max_ms: f32,
    pub warmup: i32,
    pub runs: i32,
    pub compute_units: String,
    pub cache_bust: bool,
    pub preprocess: String,
}

#[derive(Parser)]
#[command(name = "yolo26-ane")]
#[command(about = "YOLO26 on Apple Neural Engine. Zero Python.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to yolo26-coreml Swift binary
    #[arg(long, env = "YOLO26_COREML_BIN")]
    swift_bin: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run detection on an image
    Predict {
        /// Path to .mlpackage model
        model: PathBuf,
        /// Path to input image
        image: PathBuf,
        /// Compute units: ane, all, gpu, cpu
        #[arg(long, default_value = "ane")]
        compute_units: String,
        /// Confidence threshold
        #[arg(long, default_value = "0.25")]
        conf: f32,
    },
    /// Benchmark inference
    Bench {
        /// Path to .mlpackage model
        model: PathBuf,
        /// Path to input image
        image: PathBuf,
        /// Number of inference runs
        #[arg(long, default_value = "300")]
        runs: u32,
        /// Number of warmup runs
        #[arg(long, default_value = "20")]
        warmup: u32,
        /// Compute units: ane, all, gpu, cpu
        #[arg(long, default_value = "ane")]
        compute_units: String,
        /// Mutate the input pixel buffer between runs to rule out identical-input caching
        #[arg(long)]
        cache_bust: bool,
        /// Preprocessing mode: stretch or letterbox
        #[arg(long, default_value = "stretch")]
        preprocess: String,
    },
}

/// Find the yolo26-coreml Swift binary
fn find_swift_binary(explicit: Option<PathBuf>) -> Result<PathBuf> {
    // 1. Explicit path
    if let Some(p) = explicit {
        if p.exists() {
            return Ok(p);
        }
    }

    // 2. Sibling of current executable
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.parent().unwrap().join("yolo26-coreml");
        if sibling.exists() {
            return Ok(sibling);
        }
    }

    // 3. Known build path (development)
    let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(".build/release/yolo26-coreml");
    if dev_path.exists() {
        return Ok(dev_path);
    }

    // 4. PATH lookup
    let which = Command::new("which")
        .arg("yolo26-coreml")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()))
            } else {
                None
            }
        });
    if let Some(p) = which {
        return Ok(p);
    }

    anyhow::bail!(
        "Cannot find yolo26-coreml binary. Set YOLO26_COREML_BIN or place it next to this binary."
    )
}

/// Run prediction via Swift subprocess
fn predict(
    swift_bin: &PathBuf,
    model: &PathBuf,
    image: &PathBuf,
    compute_units: &str,
    conf: f32,
) -> Result<Vec<Detection>> {
    let output = Command::new(swift_bin)
        .args([
            "predict",
            model.to_str().unwrap(),
            image.to_str().unwrap(),
            "--compute-units",
            compute_units,
            "--conf",
            &conf.to_string(),
        ])
        .output()
        .context("Failed to spawn yolo26-coreml")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("yolo26-coreml failed: {}", stderr);
    }

    // Parse stdout as JSON
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detections: Vec<Detection> =
        serde_json::from_str(&stdout).context("Failed to parse detection JSON")?;

    // Print stderr (status messages) to our stderr
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{}", stderr);
    }

    Ok(detections)
}

/// Run benchmark via Swift subprocess
fn bench(
    swift_bin: &PathBuf,
    model: &PathBuf,
    image: &PathBuf,
    runs: u32,
    warmup: u32,
    compute_units: &str,
    cache_bust: bool,
    preprocess: &str,
) -> Result<BenchResult> {
    let runs_arg = runs.to_string();
    let warmup_arg = warmup.to_string();
    let mut args = vec![
        "bench",
        model.to_str().unwrap(),
        image.to_str().unwrap(),
        "--runs",
        &runs_arg,
        "--warmup",
        &warmup_arg,
        "--compute-units",
        compute_units,
        "--preprocess",
        preprocess,
    ];
    if cache_bust {
        args.push("--cache-bust");
    }

    let output = Command::new(swift_bin)
        .args(args)
        .output()
        .context("Failed to spawn yolo26-coreml")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("yolo26-coreml failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: BenchResult =
        serde_json::from_str(&stdout).context("Failed to parse bench JSON")?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{}", stderr);
    }

    Ok(result)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let swift_bin = find_swift_binary(cli.swift_bin)?;

    eprintln!("Using Swift binary: {}", swift_bin.display());

    match cli.command {
        Commands::Predict {
            model,
            image,
            compute_units,
            conf,
        } => {
            let t0 = Instant::now();
            let detections = predict(&swift_bin, &model, &image, &compute_units, conf)?;
            let elapsed = t0.elapsed();

            // Output JSON to stdout
            let json = serde_json::to_string_pretty(&detections)?;
            println!("{}", json);

            eprintln!(
                "--- {} detections in {:.1}ms (includes subprocess overhead) ---",
                detections.len(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
        Commands::Bench {
            model,
            image,
            runs,
            warmup,
            compute_units,
            cache_bust,
            preprocess,
        } => {
            let result = bench(
                &swift_bin,
                &model,
                &image,
                runs,
                warmup,
                &compute_units,
                cache_bust,
                &preprocess,
            )?;

            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);

            eprintln!(
                "--- {:.1} FPS | {:.2}ms median | {:.2}ms p99 ({} warmup, {} runs, {}, cache_bust={}, preprocess={}) ---",
                result.fps,
                result.median_ms,
                result.p99_ms,
                result.warmup,
                result.runs,
                result.compute_units,
                result.cache_bust,
                result.preprocess
            );
        }
    }

    Ok(())
}
