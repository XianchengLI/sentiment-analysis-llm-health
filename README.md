# Sentiment Analysis with Large Language Models in Digital Health

This repository contains the implementation for the research paper "The Promise of Large Language Models in Digital Health: Evidence from Sentiment Analysis in Online Health Communities."

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/XianchengLI/sentiment-analysis-llm-health.git
cd sentiment-analysis-llm-health

# Or download ZIP from: https://github.com/XianchengLI/sentiment-analysis-llm-health

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Setup

You have multiple options to set up API keys:

#### Option A: Set in Jupyter Notebook (Recommended for testing)
```python
# In your notebook
OPENAI_API_KEY = 'your_openai_key_here'    # Replace with your actual key
LLAMA_API_KEY = 'your_llama_key_here'      # Leave empty if not using
DEEPSEEK_API_KEY = 'your_deepseek_key_here' # Leave empty if not using

# Apply to environment
import os
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# ... (similar for other keys)
```

#### Option B: Permanent Setup (Recommended for production)
```bash
# Windows Command Prompt
setx OPENAI_API_KEY "your_key_here"
setx LLAMA_API_KEY "your_llama_key_here"

# Restart Jupyter after setting keys
```

#### Option C: Using .env file
```bash
# Edit .env file in project root
OPENAI_API_KEY=your_openai_key_here
LLAMA_API_KEY=your_llama_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### 3. Run Your First Experiment

```python
# In notebooks/test_experiment.ipynb
from run_experiments import run_experiment_with_custom_data

results = run_experiment_with_custom_data(
    data_path="../data/your_dataset.csv",
    models=["gpt-4o-mini", "llama3.1-70b"],
    post_id_col="PostId",           # Your post ID column
    content_col="Body",             # Your text content column  
    expert_label_col="Label"        # Your ground truth label column
)
```

## 🎯 Key Features

- 🤖 **Multi-LLM Support**: OpenAI GPT, LLaMA, DeepSeek with unified interface
- 📊 **Flexible Data Input**: Custom column names, multiple file formats
- 🏥 **Health-Domain Optimized**: Expert-derived codebook for health community analysis
- 🔄 **Zero-shot & Few-shot**: Both learning approaches implemented
- 🔒 **Privacy-Aware**: Works with synthetic data for demonstration

## 📁 Project Structure

```
sentiment-analysis-llm-health/
├── src/                           # Core modules
│   ├── data/                     # Data processing
│   │   ├── prompt_manager.py     # Prompt template management
│   │   └── sample_generator.py   # Synthetic data generation
│   ├── models/                   # LLM interfaces
│   │   └── llm_client.py        # Multi-provider LLM client
│   ├── evaluation/               # Performance evaluation
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/                    # Utility functions
├── data/                         # Data directory
│   ├── prompts/                 # Prompt templates
│   │   ├── zero_shot_prompt.txt # Zero-shot learning prompt
│   │   └── few_shot_prompt.txt  # Few-shot with examples
│   ├── sample_data/             # Generated sample datasets
│   └── real_data/               # Real datasets
├── experiments/                  # Experiment scripts
│   ├── run_experiments.py       # Main experiment runner
├── notebooks/                   # User-friendly notebooks
│   └── test_experiment.ipynb    # Example usage notebook
├── config/                      # Configuration files
├── results/                     # Experiment outputs
└── README.md
```

## 🔬 Supported Models

### OpenAI Models
- `gpt-4.1` 
- `gpt-4.1-mini` 
- `o3` 
- `o3-mini` 

### LLaMA Models 
- `llama3.1-70b` 
- `llama3.1-405b` 

### DeepSeek Models
- `deepseek-chat` 
- `deepseek-reasoner` 

### Other Models
Other models may work with the structure but not extensively tested

## 📖 Usage Examples

### Basic Usage

```python
from experiments.run_experiments import run_experiment_with_custom_data

# Simple experiment with custom data
results = run_experiment_with_custom_data(
    data_path="path/to/your/data.csv",
    models=["gpt-4o-mini", "o3"],
    post_id_col="ID",
    content_col="PostText", 
    expert_label_col="TrueLabel"
)
```

### Advanced Usage with Multiple Models

```python
# Compare multiple LLM providers
results = run_experiment_with_custom_data(
    data_path="../data/health_posts.csv",
    models=[
        "gpt-4o-mini",      # OpenAI
        "llama3.1-70b",     # LLaMA
        "deepseek-chat"     # DeepSeek
    ],
    prompt_templates=["zero_shot_prompt", "few_shot_prompt"],
    verbose=True,           # Show detailed progress
    output_dir="../results/multi_model_comparison/"
)

# Results automatically saved with performance metrics
print(f"Accuracy comparison:")
for exp_name, result in results.items():
    if 'metrics' in result:
        acc = result['metrics'].get('accuracy', 0)
        print(f"{exp_name}: {acc:.3f}")
```

### Silent Mode for Batch Processing

```python
# Run experiments without verbose output
results = run_experiment_with_custom_data(
    data_path="../data/large_dataset.csv",
    models=["gpt-4o-mini", "llama3.1-70b"],
    verbose=False  # Minimal output
)
```

## 🎛️ Configuration

### Custom Prompt Templates

Create your own prompts in `data/prompts/`:

```
data/prompts/
├── zero_shot_prompt.txt      # Your zero-shot template
├── few_shot_prompt.txt       # Your few-shot template
└── custom_domain_prompt.txt  # Domain-specific template
```

### Model Configuration

Edit `config/models.yaml` to adjust model parameters, only apply for gpt models that support these parameters:

```yaml
models:
  gpt-4o-mini:
    temperature: 0.0
    max_tokens: 50
```

## 📊 Data Format Requirements

Your dataset should be a CSV or Excel file. **Column names are completely flexible** - you can specify them when running experiments:

### Example Dataset Structure:

| Your Column Name | Description | Example Values |
|------------------|-------------|----------------|
| Any ID column | Unique identifier | POST_001, ID_123, message_id |
| Any text column | Text to analyze | "I feel much better after...", "症状改善了" |
| Any label column | Expert annotations | Positive/Negative/Neutral, 1/0/-1, Good/Bad/OK |

### Usage with Custom Columns:

```python
# Your data can have ANY column names
results = run_experiment_with_custom_data(
    data_path="your_data.csv",
    models=["gpt-4o-mini"],
    post_id_col="ID",              # ← Your actual ID column name
    content_col="PostContent",     # ← Your actual text column name  
    expert_label_col="TrueLabel"   # ← Your actual label column name
)
```

**Supported formats:**
- Flexible column names: Specify any column names in your dataset
- Multiple file formats: CSV (auto-encoding detection), Excel (.xlsx)
- International text: Automatic encoding detection (UTF-8, GBK, GB2312, etc.)

## 🔧 Troubleshooting

### Common Issues

#### 1. Encoding Errors
```python
# The system automatically tries multiple encodings:
# UTF-8, GBK, GB2312, Latin-1, CP1252
# No manual intervention needed
```

#### 2. API Key Issues
```python
# Check if keys are properly set
import os
print("OpenAI key set:", bool(os.getenv('OPENAI_API_KEY')))
print("LLaMA key set:", bool(os.getenv('LLAMA_API_KEY')))
```

#### 3. Model Not Available
```python
# Check available models based on your API keys
from src.models.llm_client import LLMClient
client = LLMClient()
available = client.get_available_models()
print(f"Available models: {available}")
```

#### 4. HTTP Request Logs
```python
# HTTP request logs are disabled by default
# To ENABLE detailed HTTP logging (for debugging), uncomment these lines:
# import logging
# logging.getLogger("httpx").setLevel(logging.INFO)
# logging.getLogger("openai").setLevel(logging.INFO)

# To ensure logs are disabled (default behavior):
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
```

## 📈 Research Methodology

### Expert Knowledge Integration

Our approach uses a structured codebook that encodes expert-derived interpretation guidelines. **You can customize these rules for your own domain by editing the prompt templates.**

#### Complete Rule Set:

1. **Improvement and Self-management**: Health improvement or effective symptom control → **Positive**
2. **Uncertainty**: Vague, unclear, or non-personal experiences → **Neutral**
3. **Objective Information**: General facts without personal reference → **Neutral**
4. **Polarized Sentiment from Emphasis**: Strong emphasis ("really helped", "so much worse") → **Positive/Negative**
5. **Helpful Advice or Resources**: Sharing tools, tips, or resources → **Positive**
6. **Tone Sensitivity**: Negative experiences ending with hope/support → **Positive**
7. **Punctuation Sensitivity**: Exclamation marks amplify emotion, question marks suggest uncertainty
8. **Health Struggles**: Pain, treatment failure, emotional hardship → **Negative**
9. **Prioritize Polarized Sentiment**: Emotional content overrides neutral elements

#### Customizing for Your Domain:

**Edit Existing Prompts**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Make your changes with English comments
4. Add tests for new functionality
5. Run the test suite (`pytest tests/`)
6. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite our paper (to publish):

```bibtex

```

## 📧 Contact

- **Corresponding Author**: Xiancheng Li (x.l.li@qmul.ac.uk)
- **Issues**: Please use the GitHub issue tracker for questions and bug reports
- **Discussions**: GitHub Discussions for research questions and methodology

**Note**: This implementation uses synthetic sample data for demonstration purposes. The original research data cannot be shared due to privacy constraints, but the methodology and code structure remain faithful to the published research.