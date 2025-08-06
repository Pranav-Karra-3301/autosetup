use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_init_command_help() {
    let mut cmd = Command::cargo_bin("autosetup").unwrap();
    cmd.arg("init")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Initialize a new ML fine-tuning project"));
}

#[test]
fn test_version() {
    let mut cmd = Command::cargo_bin("autosetup").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("autosetup"));
}

#[test]
fn test_help() {
    let mut cmd = Command::cargo_bin("autosetup").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Rust CLI for Reproducible ML Fine-Tuning Projects"));
}

#[test]
#[ignore] // This test requires user interaction
fn test_init_creates_project_structure() {
    let temp_dir = TempDir::new().unwrap();
    let project_name = "test_project";
    
    let mut cmd = Command::cargo_bin("autosetup").unwrap();
    cmd.arg("init")
        .arg(project_name)
        .arg("--nogit")
        .current_dir(&temp_dir)
        .assert()
        .success();
    
    let project_path = temp_dir.path().join(project_name);
    
    // Check directory structure
    assert!(project_path.join("configs").exists());
    assert!(project_path.join("data").exists());
    assert!(project_path.join("scripts").exists());
    assert!(project_path.join("notebooks").exists());
    assert!(project_path.join("src").exists());
    assert!(project_path.join("models").exists());
    
    // Check files
    assert!(project_path.join("requirements.txt").exists());
    assert!(project_path.join("README.md").exists());
    assert!(project_path.join("instructions.md").exists());
    assert!(project_path.join(".env").exists());
    assert!(project_path.join("run.sh").exists());
    
    // Check config files
    assert!(project_path.join("configs/base.yaml").exists());
    assert!(project_path.join("configs/dataset.yaml").exists());
    assert!(project_path.join("configs/model.yaml").exists());
    assert!(project_path.join("configs/training.yaml").exists());
    
    // Check scripts
    assert!(project_path.join("scripts/train.py").exists());
    assert!(project_path.join("scripts/evaluate.py").exists());
    assert!(project_path.join("scripts/prepare_dataset.py").exists());
    assert!(project_path.join("scripts/inference.py").exists());
    assert!(project_path.join("scripts/export_model.py").exists());
}