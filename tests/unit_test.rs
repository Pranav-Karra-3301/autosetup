use autosetup::utils::{validate_project_name, detect_gpus};
use autosetup::compute::{LocalCompute, ModalCompute};
use autosetup::templates::{ProjectConfig, TemplateEngine};
use std::path::PathBuf;

#[test]
fn test_validate_project_name() {
    assert!(validate_project_name("my-project").is_ok());
    assert!(validate_project_name("my_project").is_ok());
    assert!(validate_project_name("MyProject123").is_ok());
    
    assert!(validate_project_name("123project").is_err());
    assert!(validate_project_name("my project").is_err());
    assert!(validate_project_name("my@project").is_err());
    assert!(validate_project_name("").is_err());
}

#[test]
fn test_local_compute_config() {
    let compute = LocalCompute::new(vec![0, 1], true);
    let config = compute.generate_config();
    
    assert!(config.contains("provider: local"));
    assert!(config.contains("gpus: [0, 1]"));
    assert!(config.contains("use_ddp: true"));
    assert!(config.contains("mixed_precision: fp16"));
}

#[test]
fn test_modal_compute_config() {
    let compute = ModalCompute::new("A100".to_string(), 2);
    let config = compute.generate_config();
    
    assert!(config.contains("provider: modal"));
    assert!(config.contains("gpu_type: A100"));
    assert!(config.contains("num_gpus: 2"));
    assert!(config.contains("volume: autosetup-data"));
}

#[test]
fn test_modal_app_generation() {
    let compute = ModalCompute::new("A10G".to_string(), 1);
    let app = compute.generate_modal_app();
    
    assert!(app.contains("import modal"));
    assert!(app.contains("gpu.A10G(count=1)"));
    assert!(app.contains("Volume.from_name(\"autosetup-data\""));
    assert!(app.contains("def train():"));
    assert!(app.contains("@app.local_entrypoint()"));
}

#[test]
fn test_template_engine_creation() {
    let engine = TemplateEngine::new();
    assert!(engine.is_ok());
}

#[test]
fn test_project_config_creation() {
    let config = ProjectConfig {
        project_name: "test_project".to_string(),
        project_path: PathBuf::from("/tmp/test_project"),
        task_type: "Text Classification".to_string(),
        model_name: "bert-base-uncased".to_string(),
        dataset: "imdb".to_string(),
        compute_backend: autosetup::compute::ComputeBackend::Local,
        use_lora: false,
        use_qlora: false,
        use_sweep: false,
        use_wandb: true,
        wandb_key: Some("test_key".to_string()),
        hf_token: None,
        nogit: false,
        learning_rate: 2e-5,
        batch_size: 16,
        num_epochs: 3,
        max_length: 512,
        warmup_steps: 500,
        gpu_ids: vec![0],
        use_ddp: false,
        modal_gpu_type: None,
        modal_num_gpus: None,
    };
    
    assert_eq!(config.project_name, "test_project");
    assert_eq!(config.batch_size, 16);
    assert_eq!(config.learning_rate, 2e-5);
}

#[cfg(test)]
mod template_tests {
    use super::*;
    use tera::{Context, Tera};

    #[test]
    fn test_template_rendering() {
        let mut tera = Tera::default();
        tera.add_raw_template("test", "Project: {{ project_name }}").unwrap();
        
        let mut context = Context::new();
        context.insert("project_name", "my_project");
        
        let result = tera.render("test", &context).unwrap();
        assert_eq!(result, "Project: my_project");
    }

    #[test]
    fn test_conditional_template_rendering() {
        let mut tera = Tera::default();
        tera.add_raw_template(
            "test",
            "{% if use_wandb %}wandb enabled{% else %}wandb disabled{% endif %}"
        ).unwrap();
        
        let mut context = Context::new();
        context.insert("use_wandb", &true);
        let result = tera.render("test", &context).unwrap();
        assert_eq!(result, "wandb enabled");
        
        context.insert("use_wandb", &false);
        let result = tera.render("test", &context).unwrap();
        assert_eq!(result, "wandb disabled");
    }
}