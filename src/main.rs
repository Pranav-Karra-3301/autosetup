mod cli;
mod compute;
mod templates;
mod utils;

use anyhow::Result;
use clap::Parser;
use colored::*;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("autosetup=info".parse()?))
        .init();

    let cli = cli::Cli::parse();
    
    match cli.command {
        cli::Commands::Init(args) => {
            cli::init::run(args).await?;
        }
    }

    Ok(())
}