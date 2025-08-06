use anyhow::{Context, Result};
use regex::Regex;
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn create_directory_structure(project_path: &Path) -> Result<()> {
    let dirs = vec![
        "configs",
        "data/raw",
        "data/processed",
        "data/splits",
        "scripts",
        "notebooks",
        "src/dataset",
        "src/models",
        "src/training",
        "src/utils",
        "src/eval",
        "models/checkpoints",
        "models/logs",
    ];

    for dir in dirs {
        let path = project_path.join(dir);
        fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create directory: {:?}", path))?;
    }

    Ok(())
}

pub fn detect_gpus() -> Result<Vec<String>> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout
                .lines()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .collect())
        }
        Err(_) => Ok(vec![]),
    }
}

pub fn setup_uv_environment(project_path: &Path) -> Result<()> {
    let uv_available = which::which("uv").is_ok();

    if !uv_available {
        println!("⚠️ uv not found. Installing via pip...");
        Command::new("pip")
            .args(&["install", "uv"])
            .status()
            .context("Failed to install uv")?;
    }

    std::env::set_current_dir(project_path)?;

    Command::new("uv")
        .args(&["venv", ".venv"])
        .status()
        .context("Failed to create virtual environment")?;

    let requirements_path = project_path.join("requirements.txt");
    if requirements_path.exists() {
        let activate_cmd = if cfg!(windows) {
            ".venv\\Scripts\\activate"
        } else {
            "source .venv/bin/activate"
        };

        Command::new("uv")
            .args(&["pip", "install", "-r", "requirements.txt"])
            .status()
            .context("Failed to install requirements")?;
    }

    Ok(())
}

pub fn setup_git(project_path: &Path) -> Result<()> {
    std::env::set_current_dir(project_path)?;

    Command::new("git")
        .arg("init")
        .status()
        .context("Failed to initialize git repository")?;

    Command::new("git")
        .args(&["add", "."])
        .status()
        .context("Failed to add files to git")?;

    Command::new("git")
        .args(&["commit", "-m", "Initial commit from autosetup"])
        .status()
        .context("Failed to create initial commit")?;

    Ok(())
}

pub fn run_command(cmd: &str, args: &[&str], cwd: Option<&Path>) -> Result<String> {
    let mut command = Command::new(cmd);
    command.args(args);

    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    let output = command
        .output()
        .with_context(|| format!("Failed to run command: {} {:?}", cmd, args))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Command failed: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn validate_project_name(name: &str) -> Result<()> {
    let re = Regex::new(r"^[a-zA-Z][a-zA-Z0-9-_]*$")?;
    if !re.is_match(name) {
        anyhow::bail!(
            "Invalid project name. Must start with a letter and contain only letters, numbers, hyphens, and underscores."
        );
    }
    Ok(())
}