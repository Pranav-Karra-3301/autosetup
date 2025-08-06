pub mod init;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "autosetup",
    version,
    about = "Rust CLI for Reproducible ML Fine-Tuning Projects",
    long_about = "A fast, reliable CLI tool that bootstraps professional, reproducible fine-tuning codebases in seconds."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Initialize a new ML fine-tuning project")]
    Init(init::InitArgs),
}