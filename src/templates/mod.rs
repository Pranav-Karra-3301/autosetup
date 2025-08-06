mod python_scripts;
mod config_templates;
mod notebook_templates;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tera::Tera;

use crate::compute::ComputeBackend;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project_name: String,
    pub project_path: PathBuf,
    pub task_type: String,
    pub model_name: String,
    pub dataset: String,
    pub compute_backend: ComputeBackend,
    pub use_lora: bool,
    pub use_qlora: bool,
    pub use_sweep: bool,
    pub use_wandb: bool,
    pub wandb_key: Option<String>,
    pub hf_token: Option<String>,
    pub nogit: bool,
    pub learning_rate: f64,
    pub batch_size: u32,
    pub num_epochs: u32,
    pub max_length: u32,
    pub warmup_steps: u32,
    pub gpu_ids: Vec<usize>,
    pub use_ddp: bool,
    pub modal_gpu_type: Option<String>,
    pub modal_num_gpus: Option<u32>,
}

pub struct TemplateEngine {
    tera: Tera,
}

impl TemplateEngine {
    pub fn new() -> Result<Self> {
        let mut tera = Tera::default();
        
        tera.add_raw_templates(vec![
            ("train.py", python_scripts::TRAIN_SCRIPT),
            ("evaluate.py", python_scripts::EVALUATE_SCRIPT),
            ("prepare_dataset.py", python_scripts::PREPARE_DATASET_SCRIPT),
            ("inference.py", python_scripts::INFERENCE_SCRIPT),
            ("export_model.py", python_scripts::EXPORT_MODEL_SCRIPT),
            ("sweep.py", python_scripts::SWEEP_SCRIPT),
            ("base.yaml", config_templates::BASE_CONFIG),
            ("dataset.yaml", config_templates::DATASET_CONFIG),
            ("model.yaml", config_templates::MODEL_CONFIG),
            ("training.yaml", config_templates::TRAINING_CONFIG),
            ("wandb.yaml", config_templates::WANDB_CONFIG),
            ("requirements.txt", config_templates::REQUIREMENTS),
            ("README.md", config_templates::README_TEMPLATE),
            ("instructions.md", config_templates::INSTRUCTIONS_TEMPLATE),
            (".env", config_templates::ENV_TEMPLATE),
            ("run.sh", config_templates::RUN_SCRIPT),
            ("exploration.ipynb", notebook_templates::EXPLORATION_NOTEBOOK),
            ("results.ipynb", notebook_templates::RESULTS_NOTEBOOK),
            ("dataset.py", python_scripts::DATASET_MODULE),
            ("model.py", python_scripts::MODEL_MODULE),
            ("trainer.py", python_scripts::TRAINER_MODULE),
            ("utils.py", python_scripts::UTILS_MODULE),
            ("evaluator.py", python_scripts::EVALUATOR_MODULE),
        ])?;

        Ok(Self { tera })
    }

    pub async fn render_all(&self, config: &ProjectConfig) -> Result<()> {
        let context = self.create_context(config)?;

        self.render_scripts(config, &context)?;
        self.render_configs(config, &context)?;
        self.render_src_modules(config, &context)?;
        self.render_notebooks(config, &context)?;
        self.render_project_files(config, &context)?;

        if config.compute_backend == ComputeBackend::Modal {
            self.render_modal_files(config, &context)?;
        }

        if config.use_sweep {
            self.render_sweep_files(config, &context)?;
        }

        Ok(())
    }

    fn create_context(&self, config: &ProjectConfig) -> Result<tera::Context> {
        let mut context = tera::Context::new();
        
        context.insert("project_name", &config.project_name);
        context.insert("task_type", &config.task_type);
        context.insert("model_name", &config.model_name);
        context.insert("dataset", &config.dataset);
        context.insert("use_lora", &config.use_lora);
        context.insert("use_qlora", &config.use_qlora);
        context.insert("use_wandb", &config.use_wandb);
        context.insert("learning_rate", &config.learning_rate);
        context.insert("batch_size", &config.batch_size);
        context.insert("num_epochs", &config.num_epochs);
        context.insert("max_length", &config.max_length);
        context.insert("warmup_steps", &config.warmup_steps);
        context.insert("use_ddp", &config.use_ddp);
        
        let gpu_list = if config.gpu_ids.is_empty() {
            "cpu".to_string()
        } else {
            config.gpu_ids.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        context.insert("gpu_ids", &gpu_list);

        if let Some(gpu_type) = &config.modal_gpu_type {
            context.insert("modal_gpu_type", gpu_type);
        }
        if let Some(num_gpus) = &config.modal_num_gpus {
            context.insert("modal_num_gpus", num_gpus);
        }

        Ok(context)
    }

    fn render_scripts(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let scripts = vec![
            ("train.py", "scripts/train.py"),
            ("evaluate.py", "scripts/evaluate.py"),
            ("prepare_dataset.py", "scripts/prepare_dataset.py"),
            ("inference.py", "scripts/inference.py"),
            ("export_model.py", "scripts/export_model.py"),
        ];

        for (template, path) in scripts {
            let content = self.tera.render(template, context)?;
            let full_path = config.project_path.join(path);
            fs::write(&full_path, content)
                .with_context(|| format!("Failed to write {}", path))?;
        }

        Ok(())
    }

    fn render_configs(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let configs = vec![
            ("base.yaml", "configs/base.yaml"),
            ("dataset.yaml", "configs/dataset.yaml"),
            ("model.yaml", "configs/model.yaml"),
            ("training.yaml", "configs/training.yaml"),
        ];

        for (template, path) in configs {
            let content = self.tera.render(template, context)?;
            let full_path = config.project_path.join(path);
            fs::write(&full_path, content)
                .with_context(|| format!("Failed to write {}", path))?;
        }

        if config.use_wandb {
            let content = self.tera.render("wandb.yaml", context)?;
            let full_path = config.project_path.join("configs/wandb.yaml");
            fs::write(&full_path, content)?;
        }

        Ok(())
    }

    fn render_src_modules(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let modules = vec![
            ("dataset.py", "src/dataset/dataset.py"),
            ("model.py", "src/models/model.py"),
            ("trainer.py", "src/training/trainer.py"),
            ("utils.py", "src/utils/utils.py"),
            ("evaluator.py", "src/eval/evaluator.py"),
        ];

        for (template, path) in modules {
            let content = self.tera.render(template, context)?;
            let full_path = config.project_path.join(path);
            fs::write(&full_path, content)
                .with_context(|| format!("Failed to write {}", path))?;
            
            let init_path = full_path.parent().unwrap().join("__init__.py");
            if !init_path.exists() {
                fs::write(&init_path, "")?;
            }
        }

        Ok(())
    }

    fn render_notebooks(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let notebooks = vec![
            ("exploration.ipynb", "notebooks/exploration.py"),
            ("results.ipynb", "notebooks/results.py"),
        ];

        for (template, path) in notebooks {
            let content = self.tera.render(template, context)?;
            let full_path = config.project_path.join(path);
            fs::write(&full_path, content)
                .with_context(|| format!("Failed to write {}", path))?;
        }

        Ok(())
    }

    fn render_project_files(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let files = vec![
            ("requirements.txt", "requirements.txt"),
            ("README.md", "README.md"),
            ("instructions.md", "instructions.md"),
            (".env", ".env"),
            ("run.sh", "run.sh"),
        ];

        for (template, path) in files {
            let content = self.tera.render(template, context)?;
            let full_path = config.project_path.join(path);
            fs::write(&full_path, content)
                .with_context(|| format!("Failed to write {}", path))?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let run_sh_path = config.project_path.join("run.sh");
            let mut perms = fs::metadata(&run_sh_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&run_sh_path, perms)?;
        }

        Ok(())
    }

    fn render_modal_files(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let modal_app = crate::compute::ModalCompute::new(
            config.modal_gpu_type.clone().unwrap_or_else(|| "A10G".to_string()),
            config.modal_num_gpus.unwrap_or(1),
        ).generate_modal_app();

        let modal_path = config.project_path.join("scripts/train_modal.py");
        fs::write(&modal_path, modal_app)?;

        Ok(())
    }

    fn render_sweep_files(&self, config: &ProjectConfig, context: &tera::Context) -> Result<()> {
        let sweep_config = format!(
            r#"program: scripts/train.py
method: bayes
metric:
  name: eval/loss
  goal: minimize
parameters:
  learning_rate:
    min: 1e-6
    max: 1e-3
    distribution: log_uniform
  batch_size:
    values: [8, 16, 32]
  warmup_steps:
    min: 0
    max: 1000
  weight_decay:
    min: 0.0
    max: 0.1
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
"#
        );

        let sweep_path = config.project_path.join("configs/sweep.yaml");
        fs::write(&sweep_path, sweep_config)?;

        let sweep_script = self.tera.render("sweep.py", context)?;
        let script_path = config.project_path.join("scripts/sweep.py");
        fs::write(&script_path, sweep_script)?;

        Ok(())
    }
}