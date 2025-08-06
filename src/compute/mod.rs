use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComputeBackend {
    Local,
    Modal,
}

pub struct LocalCompute {
    pub gpu_ids: Vec<usize>,
    pub use_ddp: bool,
}

pub struct ModalCompute {
    pub gpu_type: String,
    pub num_gpus: u32,
    pub volume_name: String,
}

impl LocalCompute {
    pub fn new(gpu_ids: Vec<usize>, use_ddp: bool) -> Self {
        Self { gpu_ids, use_ddp }
    }

    pub fn generate_config(&self) -> String {
        format!(
            r#"compute:
  provider: local
  gpus: {:?}
  use_ddp: {}
  mixed_precision: fp16"#,
            self.gpu_ids, self.use_ddp
        )
    }
}

impl ModalCompute {
    pub fn new(gpu_type: String, num_gpus: u32) -> Self {
        Self {
            gpu_type,
            num_gpus,
            volume_name: "autosetup-data".to_string(),
        }
    }

    pub fn generate_config(&self) -> String {
        format!(
            r#"compute:
  provider: modal
  gpu_type: {}
  num_gpus: {}
  volume: {}
  image: autosetup/image:latest"#,
            self.gpu_type, self.num_gpus, self.volume_name
        )
    }

    pub fn generate_modal_app(&self) -> String {
        format!(
            r#"import modal
from modal import Image, gpu, Volume

app = modal.App("autosetup-finetune")

volume = Volume.from_name("{}", create_if_missing=True)

image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .run_commands("apt-get update && apt-get install -y git wget")
)

@app.function(
    image=image,
    gpu=gpu.{}(count={}),
    volumes={{"/data": volume}},
    timeout=86400,
)
def train():
    import sys
    sys.path.append("/root")
    from scripts.train import main
    main()

@app.local_entrypoint()
def main():
    train.remote()
"#,
            self.volume_name,
            self.gpu_type.replace("-", "_"),
            self.num_gpus
        )
    }
}