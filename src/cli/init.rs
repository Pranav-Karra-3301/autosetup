use anyhow::Result;
use clap::Parser;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use inquire::{Confirm, Select, Text};
use std::env;
use std::time::Duration;

use crate::compute::ComputeBackend;
use crate::templates::{ProjectConfig, TemplateEngine};
use crate::utils::{create_directory_structure, setup_git, setup_uv_environment};

#[derive(Parser, Debug)]
pub struct InitArgs {
    #[arg(long, help = "Enable advanced configuration mode")]
    pub advanced: bool,

    #[arg(long, help = "Use current directory as project root")]
    pub pwd: bool,

    #[arg(long, help = "Skip git initialization")]
    pub nogit: bool,

    #[arg(help = "Project name (optional, will prompt if not provided)")]
    pub name: Option<String>,
}

pub async fn run(args: InitArgs) -> Result<()> {
    println!("{}", "🚀 Welcome to Autosetup!".green().bold());
    println!("{}", "Let's create your ML fine-tuning project.\n".cyan());

    let config = if args.advanced {
        advanced_setup(args).await?
    } else {
        simple_setup(args).await?
    };

    let spinner = create_spinner("Creating project structure...");
    create_project(&config).await?;
    spinner.finish_with_message(format!("✅ Project structure created"));

    if !config.nogit {
        let spinner = create_spinner("Initializing git repository...");
        setup_git(&config.project_path)?;
        spinner.finish_with_message(format!("✅ Git repository initialized"));
    }

    let spinner = create_spinner("Setting up Python environment...");
    setup_uv_environment(&config.project_path)?;
    spinner.finish_with_message(format!("✅ Python environment created"));

    print_success_message(&config);

    Ok(())
}

async fn simple_setup(args: InitArgs) -> Result<ProjectConfig> {
    let project_name = match args.name {
        Some(name) => name,
        None => Text::new("Project name:")
            .with_default("my-finetune-project")
            .prompt()?,
    };

    let project_path = if args.pwd {
        env::current_dir()?
    } else {
        env::current_dir()?.join(&project_name)
    };

    let task_type = Select::new(
        "Task type:",
        vec![
            "Text Classification",
            "Language Modeling",
            "Question Answering",
            "Summarization",
            "Custom",
        ],
    )
    .prompt()?;

    let model_name = Text::new("Model name (HuggingFace):")
        .with_default("bert-base-uncased")
        .prompt()?;

    let dataset = Text::new("Dataset (HuggingFace name or local path):")
        .with_default("imdb")
        .prompt()?;

    let use_wandb = Confirm::new("Use Weights & Biases for logging?")
        .with_default(true)
        .prompt()?;

    let wandb_key = if use_wandb {
        Some(
            Text::new("W&B API key (press Enter to skip):")
                .prompt()
                .unwrap_or_default(),
        )
    } else {
        None
    };

    let hf_token = Text::new("HuggingFace token (press Enter to skip):")
        .prompt()
        .unwrap_or_default();

    Ok(ProjectConfig {
        project_name: project_name.clone(),
        project_path,
        task_type: task_type.to_string(),
        model_name,
        dataset,
        compute_backend: ComputeBackend::Local,
        use_lora: false,
        use_qlora: false,
        use_sweep: false,
        use_wandb,
        wandb_key,
        hf_token: if hf_token.is_empty() {
            None
        } else {
            Some(hf_token)
        },
        nogit: args.nogit,
        learning_rate: 2e-5,
        batch_size: 16,
        num_epochs: 3,
        max_length: 512,
        warmup_steps: 500,
        gpu_ids: vec![],
        use_ddp: false,
        modal_gpu_type: None,
        modal_num_gpus: None,
    })
}

async fn advanced_setup(args: InitArgs) -> Result<ProjectConfig> {
    let mut config = simple_setup(args).await?;

    println!("\n{}", "⚙️ Advanced Configuration".yellow().bold());

    config.use_lora = Confirm::new("Enable LoRA (Low-Rank Adaptation)?")
        .with_default(false)
        .prompt()?;

    if config.use_lora {
        config.use_qlora = Confirm::new("Enable QLoRA (Quantized LoRA)?")
            .with_default(false)
            .prompt()?;
    }

    config.use_sweep = Confirm::new("Add hyperparameter sweep configuration?")
        .with_default(false)
        .prompt()?;

    let compute_choice = Select::new(
        "Compute backend:",
        vec!["Local (GPU/CPU)", "Modal (Serverless Cloud)"],
    )
    .prompt()?;

    match compute_choice {
        "Local (GPU/CPU)" => {
            config.compute_backend = ComputeBackend::Local;
            let gpu_info = crate::utils::detect_gpus()?;
            
            if !gpu_info.is_empty() {
                println!("Detected GPUs:");
                for (i, gpu) in gpu_info.iter().enumerate() {
                    println!("  [{}] {}", i, gpu);
                }

                let gpu_selection = Text::new("Select GPU IDs (comma-separated, or 'all'):")
                    .with_default("0")
                    .prompt()?;

                if gpu_selection == "all" {
                    config.gpu_ids = (0..gpu_info.len()).collect();
                } else {
                    config.gpu_ids = gpu_selection
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .collect();
                }

                if config.gpu_ids.len() > 1 {
                    config.use_ddp = Confirm::new("Use Distributed Data Parallel (DDP)?")
                        .with_default(true)
                        .prompt()?;
                }
            }
        }
        "Modal (Serverless Cloud)" => {
            config.compute_backend = ComputeBackend::Modal;
            
            let gpu_type = Select::new(
                "Modal GPU type:",
                vec!["A10G", "A100-40GB", "A100-80GB", "H100"],
            )
            .prompt()?;
            config.modal_gpu_type = Some(gpu_type.to_string());

            let num_gpus = Text::new("Number of GPUs:")
                .with_default("1")
                .prompt()?
                .parse::<u32>()
                .unwrap_or(1);
            config.modal_num_gpus = Some(num_gpus);
        }
        _ => {}
    }

    println!("\n{}", "📊 Hyperparameters".yellow().bold());

    config.learning_rate = Text::new("Learning rate:")
        .with_default("2e-5")
        .prompt()?
        .parse::<f64>()
        .unwrap_or(2e-5);

    config.batch_size = Text::new("Batch size:")
        .with_default("16")
        .prompt()?
        .parse::<u32>()
        .unwrap_or(16);

    config.num_epochs = Text::new("Number of epochs:")
        .with_default("3")
        .prompt()?
        .parse::<u32>()
        .unwrap_or(3);

    config.max_length = Text::new("Max sequence length:")
        .with_default("512")
        .prompt()?
        .parse::<u32>()
        .unwrap_or(512);

    Ok(config)
}

async fn create_project(config: &ProjectConfig) -> Result<()> {
    create_directory_structure(&config.project_path)?;

    let engine = TemplateEngine::new()?;
    engine.render_all(config).await?;

    Ok(())
}

fn create_spinner(msg: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );
    spinner.set_message(msg.to_string());
    spinner.enable_steady_tick(Duration::from_millis(100));
    spinner
}

fn print_success_message(config: &ProjectConfig) {
    println!("\n{}", "✨ Project created successfully!".green().bold());
    println!("\n{}", "Next steps:".cyan().bold());
    
    let project_name = config.project_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&config.project_name);

    if !config.project_path.eq(&env::current_dir().unwrap()) {
        println!("  1. cd {}", project_name);
        println!("  2. source .venv/bin/activate");
        println!("  3. python scripts/train.py");
    } else {
        println!("  1. source .venv/bin/activate");
        println!("  2. python scripts/train.py");
    }

    println!("\n{}", "📚 Documentation:".cyan());
    println!("  - README.md: Project overview and setup");
    println!("  - instructions.md: Detailed usage instructions");
    
    if config.use_wandb {
        println!("\n{}", "🔑 Remember to set your W&B API key:".yellow());
        println!("  export WANDB_API_KEY=your_key_here");
    }

    if config.compute_backend == ComputeBackend::Modal {
        println!("\n{}", "☁️ Modal setup:".yellow());
        println!("  1. pip install modal");
        println!("  2. modal token new");
        println!("  3. modal run scripts/train_modal.py");
    }
}