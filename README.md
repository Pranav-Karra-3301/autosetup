# autosetup: Rust CLI for Reproducible ML Fine‑Tuning Projects

Welcome to **autosetup**, a Rust‑based command‑line tool that bootstraps a professional, reproducible fine‑tuning codebase in seconds. Designed for both beginners and power users, autosetup generates a complete folder structure, config files, scripts, environment setup, and compute integration (local GPUs or Modal). It even ships with `uv`‑style animations and progress bars.

---

## 🚀 Key Features

- **Rust CLI**: Fast, reliable, installable via **cargo**, **curl**, and **brew**.
- **Two Modes**:
    - **Simple**: Answer a few prompts—project name, model, dataset, task—and go.
    - **Advanced**: Full control over configs, compute backends, LoRA, hyperparameter sweeps, and more.
- **Industry‑Standard Structure**:
    - Modular `configs/`, `src/`, `scripts/`, `notebooks/`, `data/`, and `models/` folders.
    - Config‑driven: Hydra/OmegaConf style YAML files.
- **Environment Automation**:
    - Uses `uv venv` to create and activate Python virtual environments.
    - Installs `requirements.txt` automatically.
- **Compute Backend Integration**:
    - **Local**: Auto‑detect GPUs via `nvidia‑smi`, choose DDP or single GPU.
    - **Modal**: Serverless cloud compute templates (`A10G`, `A100`, CPU).
- **Polished UX**:
    - `indicatif` spinners, progress bars, colored output, and emojis.
    - Fast interactive prompts via `inquire`.
- **Templating**:
    - `tera` templates for Python scripts, README, and configs.
- **Extensible**:
    - Add custom templates, override defaults, plug in new compute providers.

---

## 📦 Installation

### Cargo (Recommended for Rust developers)

```bash
cargo install autosetup
```

### Homebrew (macOS/Linux)

```bash
brew tap Pranav-Karra-3301/autosetup
brew install autosetup
```

### Curl + sh (Universal)

```bash
curl -fsSL https://raw.githubusercontent.com/Pranav-Karra-3301/autosetup/main/install.sh | sh
```

### From Source

```bash
git clone https://github.com/Pranav-Karra-3301/autosetup
cd autosetup
cargo build --release
sudo mv target/release/autosetup /usr/local/bin/
```

---

## 🛠 Usage

### Simple Mode

```bash
autosetup init
```

1. **Project name**: e.g., `my-bert-project`
2. **Task type**: Text classification, LM, QA, Summarization, Custom
3. **Model**: e.g., `bert-base-uncased`
4. **Dataset**: HF dataset name or local path
5. **Hyperparameters**: Accept defaults or customize
6. **HuggingFace login**: Optional HF token input
7. **W&B integration**: Optional API key
8. **Git initialization**: Automatic setup for version control

Autosetup then scaffolds the project and runs:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Advanced Mode

```bash
autosetup init --advanced
```

- **LoRA/QLoRA support**: Enable parameter-efficient fine-tuning
- **Hyperparameter sweep**: Add `sweep.yaml` and `scripts/sweep.py`
- **Compute backend**:
    - Local GPUs (auto‑detect & choose)
    - Modal (choose instance type, volume caching)
- **Notebook templates**: exploration, results, embedding analysis

### Additional Options

```bash
# Use current directory as project root
autosetup init --pwd

# Skip git initialization
autosetup init --nogit

# Provide project name directly
autosetup init my-project
```

---

## 📁 Generated Structure

```
finetune-project/
├── configs/
│   ├── base.yaml       # project metadata & paths
│   ├── dataset.yaml    # data locations & preprocessing
│   ├── model.yaml      # model name & architecture overrides
│   ├── training.yaml   # hyperparameters & compute settings
│   └── wandb.yaml      # logging/experiment tracking
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── export_model.py
│   └── sweep.py
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── results.ipynb
│   └── embedding_analysis.ipynb
│
├── src/
│   ├── dataset/         # loader & preprocessing modules
│   ├── models/          # model & tokenizer wrappers
│   ├── training/        # custom Trainer, callbacks, scheduler
│   ├── utils/           # metrics, logging, config loader
│   └── eval/            # evaluator & visualization code
│
├── models/
│   ├── checkpoints/
│   └── logs/            # wandb or tensorboard
│
├── .env                 # environment variables & HF token
├── .gitignore
├── requirements.txt     # pip dependencies
├── README.md            # project documentation
├── instructions.md      # detailed usage guide
└── run.sh               # one-click training script
```

---

## ⚙️ Compute Configuration

Autosetup writes `configs/training.yaml` with:

```yaml
compute:
  provider: local        # or "modal"
  gpus: [0, 1]           # for local
  use_ddp: true
  mixed_precision: fp16

  # Modal backend options:
  gpu_type: A100
  num_gpus: 1
  volume: autosetup-data
  image: autosetup/image:latest
```

In local mode, it runs `nvidia-smi` under the hood and prompts you to select among detected GPUs or CPU.

---

## 🎯 Example Workflows

### Fine-tune BERT on IMDB

```bash
autosetup init
# Project name: bert-imdb
# Task: Text Classification
# Model: bert-base-uncased
# Dataset: imdb

cd bert-imdb
source .venv/bin/activate
python scripts/train.py
```

### Fine-tune Llama with LoRA

```bash
autosetup init --advanced
# Enable LoRA: Yes
# Model: meta-llama/Llama-2-7b-hf
# Use QLoRA: Yes (for 4-bit training)

cd llama-project
source .venv/bin/activate
python scripts/train.py
```

### Cloud Training with Modal

```bash
autosetup init --advanced
# Compute backend: Modal
# GPU type: A100-40GB
# Number of GPUs: 2

cd modal-project
modal run scripts/train_modal.py
```

---

## 🧪 Development

### Building from Source

```bash
git clone https://github.com/Pranav-Karra-3301/autosetup
cd autosetup
cargo build --release
```

### Running Tests

```bash
cargo test
cargo test --release
```

### Code Quality

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
```

---

## 📚 Further Reading & Contribution

- [Rust Documentation](https://www.rust-lang.org/learn)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Modal Documentation](https://modal.com/docs)
- [Weights & Biases](https://docs.wandb.ai/)

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Rust](https://www.rust-lang.org/)
- UI powered by [inquire](https://github.com/mikaelmello/inquire) and [indicatif](https://github.com/console-rs/indicatif)
- Templates via [Tera](https://tera.netlify.app/)
- ML frameworks: [Transformers](https://huggingface.co/transformers), [PyTorch](https://pytorch.org/)

---

*Made with ❤️ for the ML community*