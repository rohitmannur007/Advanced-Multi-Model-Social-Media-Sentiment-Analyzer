# Advanced Multi-Model Social Media Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-purple)](https://mlflow.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Management-green)](https://python-poetry.org/)

## Overview

The **Advanced Multi-Model Social Media Sentiment Analyzer** is a comprehensive Python-based toolkit for performing sentiment analysis on social media content, with a focus on tweets. This project leverages the [TweetEval dataset](https://huggingface.co/datasets/tweet_eval) from Hugging Face to classify tweets as **positive**, **negative**, or **neutral**. It goes beyond basic classification by comparing multiple model architectures—including traditional machine learning (ML) models, encoder-only transformers (e.g., BERT variants), and decoder-only large language models (LLMs) like Mistral and Gemma—to evaluate their performance, data efficiency, and computational requirements.

Key goals include:
- **Model Comparison**: Assess accuracy, training time, and parameter efficiency across model types.
- **Exploratory Data Analysis (EDA)**: Uncover insights like trending hashtags, semantic clusters, and network graphs of tweet interactions.
- **Scalability and Optimization**: Use techniques like LoRA fine-tuning for LLMs, few-shot learning, and hyperparameter optimization with Optuna.

This project is ideal for researchers, data scientists, or developers interested in NLP for social media monitoring, trend detection, or opinion mining.

## Features

- **Multi-Modal Support**: Handles traditional ML (Logistic Regression, Random Forest, VADER), encoder-only transformers, and decoder-only LLMs.
- **Preprocessing Pipelines**: Custom tokenization, cleaning, and prompt engineering tailored to each model type.
- **Fine-Tuning & Evaluation**: LoRA for efficient LLM tuning, few-shot prompting, and metrics like accuracy, F1-score, and confusion matrices.
- **Experiment Tracking**: Integrated with MLflow for logging runs, parameters, and artifacts; Optuna for automated hyperparameter tuning.
- **Advanced Insights**: 
  - Hashtag semantic clustering using embeddings.
  - Graph-based analysis of tweet networks.
  - Similarity computations for trend detection.
- **Visualization**: Jupyter notebooks for EDA, plus MLflow/Optuna dashboards for interactive model comparison.
- **Modular Design**: Easy to extend with new models or datasets.

## Technologies & Dependencies

- **Core**: Python 3.8+, PyTorch, Hugging Face Transformers & Datasets.
- **ML/Optimization**: Scikit-learn, Optuna, VADERSentiment.
- **Tracking & Viz**: MLflow, Matplotlib, NetworkX, Seaborn.
- **Package Management**: Poetry (see `pyproject.toml` for full list).
- **Other**: Flash-Attention for efficient training, SQLite for local DB.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/rohitmannur007/Advanced-Multi-Model-Social-Media-Sentiment-Analyzer.git
   cd Advanced-Multi-Model-Social-Media-Sentiment-Analyzer
   ```

2. **Set Up Virtual Environment with Poetry**:
   ```
   poetry install
   poetry shell  # Activate the environment
   ```

3. **Install Additional Requirements** (if needed for GPUs or extras):
   - For CUDA support: Ensure PyTorch is installed with CUDA via Poetry.
   - Run `poetry add torch torchvision torchaudio --extras "cuda"` if using GPUs.

4. **Initialize MLflow** (optional, for tracking):
   ```
   mlflow ui  # Starts the UI at http://localhost:5000
   ```

5. **Download Dataset**:
   The TweetEval dataset is auto-loaded via Hugging Face in scripts, but you can preload it:
   ```
   python -c "from datasets import load_dataset; load_dataset('tweet_eval', 'sentiment')"
   ```

**Note**: Ensure you have sufficient RAM/GPU for LLMs (e.g., 16GB+ for Mistral). Use `device='cpu'` in scripts for testing.

## Quick Start / Usage

### 1. Run Exploratory Data Analysis
Launch the EDA notebook to visualize the dataset:
```
poetry run jupyter notebook notebooks/eda.ipynb
```
- This covers data loading, distribution plots, and initial preprocessing.

### 2. Train and Evaluate Models
- **Machine Learning Models** (e.g., Logistic Regression):
  ```
  poetry run python src/social_media_nlp/experiments/ml/tune.py --model logistic_regression
  ```
- **Encoder-Only Transformers** (e.g., BERT):
  ```
  poetry run python src/social_media_nlp/experiments/seq_lm/train.py --model bert-base-uncased
  poetry run python src/social_media_nlp/experiments/seq_lm/evaluation.py --model bert-base-uncased
  ```
- **Decoder-Only LLMs** (e.g., Mistral with LoRA):
  ```
  poetry run python src/social_media_nlp/experiments/llm/train.py --model mistral-7b --epochs 3
  poetry run python src/social_media_nlp/experiments/llm/merge_models.py --model mistral-7b
  poetry run python src/social_media_nlp/experiments/llm/evaluate_few_shot.py --model mistral-7b --shots 5
  ```

### 3. Generate Insights
- Hashtag Clustering:
  ```
  poetry run python src/social_media_nlp/visualization/hashtag_clustering.py
  ```
- View Results in MLflow:
  Open `http://localhost:5000` to compare experiments.

### 4. Inference on New Data
Load a trained model for predictions:
```
poetry run python src/social_media_nlp/models/transformers/inference.py --model bert-base-uncased --input_text "I love this product! #Happy"
```

For full usage details, refer to docstrings in each script or run `poetry run python <script> --help`.

## Project Structure

This repository is organized modularly for clarity and extensibility. Below is a detailed breakdown of every folder and file, including their purpose, key contents, and how they fit into the workflow. Paths are relative to the root.

### Top-Level Files
- **`mlflow.db`**: SQLite database used by Optuna for storing hyperparameter tuning trials and results. Auto-generated during tuning experiments (e.g., via `hyperparameter_tuning.py`). Do not edit manually—it's for backend storage.
- **`poetry.lock`**: Lockfile generated by Poetry, pinning exact versions of all dependencies (e.g., `transformers==4.35.0`). Ensures reproducible builds across environments.
- **`pyproject.toml`**: Poetry configuration file defining project metadata (name, version, authors), dependencies (e.g., `torch`, `datasets`), and scripts. Use this to add/remove packages with `poetry add <pkg>`.
- **`README.md`**: The main documentation file (this one!). Covers setup, usage, and structure.

### Top-Level Folders
- **`mlruns/`**: Directory for MLflow experiment artifacts. Contains subfolders for each run (e.g., `mlruns/0/<run-id>/`), storing metrics, parameters, models, and logs. Generated automatically during training/evaluation. Use `mlflow ui` to browse.
- **`models/`**: Stores saved model checkpoints, predictions, and outputs from experiments. 
  - Subfolders like `<model-name>/` hold `.bin`/`.pt` files (PyTorch weights), `predictions.json` (evaluation results), and config files. Populated by training scripts (e.g., `train.py`).
- **`notebooks/`**: Jupyter notebooks for interactive analysis.
  - **`eda.ipynb`**: Core EDA notebook. Loads the TweetEval dataset, performs statistical summaries (e.g., label distribution, text length histograms), visualizes trends (e.g., word clouds), and preprocesses samples. Run this first to understand the data.

### `src/social_media_nlp/` (Core Source Code)
The heart of the project—modular Python package for data, models, and experiments. Import as `from social_media_nlp import ...`.

#### `src/social_media_nlp/data/`
Handles dataset ingestion and cleaning.
- **`cleaning.py`**: Text normalization script. Functions like `clean_tweet(text)` remove URLs, mentions, emojis, and stopwords; applies lemmatization and lowercase conversion. Used in preprocessing pipelines for ML and transformer models.
- **`preprocessing.py`**: Model-agnostic pipeline. Includes tokenizers (e.g., BERTTokenizer), padding, and dataset splitting. Supports custom prompts for LLMs. Returns PyTorch DataLoaders.

#### `src/social_media_nlp/experiments/`
Experiment runners, grouped by model category. Each subfolder has dedicated train/eval scripts.

- **`llm/`** (Decoder-Only LLMs, e.g., Mistral/Gemma):
  - **`evaluate_few_shot.py`**: Runs few-shot inference (0-10 shots) on LLMs. Computes metrics like ROUGE for prompt-based classification. Ideal for low-data scenarios.
  - **`evaluate.py`**: General evaluation loop for fine-tuned LLMs. Loads models, runs on test set, logs to MLflow (accuracy, precision/recall).
  - **`merge_models.py`**: Merges LoRA adapters back into the base LLM weights post-fine-tuning. Outputs a standalone model file for deployment.
  - **`train.py`**: Fine-tunes LLMs with LoRA/PEFT. Supports prompt engineering from `prompts.py`; tracks with MLflow.

- **`ml/`** (Traditional Machine Learning):
  - **`tune.py`**: Hyperparameter tuner using Optuna. Fits models like LogisticRegression or RandomForest on TF-IDF features; optimizes via cross-validation. Logs best params to MLflow.

- **`seq_lm/`** (Encoder-Only Transformers, e.g., BERT):
  - **`evaluation.py`**: Post-training evaluator. Computes classification report, confusion matrix, and visualizations. Compares against baselines.
  - **`train.py`**: Trainer for sequence models using Hugging Face's Trainer API. Handles tokenization, fine-tuning, and saving to `models/`.

#### `src/social_media_nlp/models/`
Model utilities, metrics, and implementations.

- **`evaluation.py`**: Shared metrics module. Defines functions for F1-score, accuracy, and custom LLM-specific metrics (e.g., perplexity). Used across all experiment evaluators.
- **`hyperparameter_tuning.py`**: Optuna integration. Defines search spaces (e.g., learning rate, batch size) and objective functions for ML/transformer tuning. Stores trials in `mlflow.db`.
- **`ml/`** (ML Model Wrappers):
  - **`vader.py`**: Integrates VADER (Valence Aware Dictionary for sEntiment Reasoning). Rule-based scorer for tweets; handles slang/emojis. Used as a zero-shot baseline.
- **`transformers/`** (Transformer Utilities):
  - **`inference.py`**: Standalone inference script. Loads fine-tuned models for batch/single predictions; supports CPU/GPU.
  - **`train.py`**: Low-level training logic for transformers (complements `experiments/seq_lm/train.py`). Includes early stopping and gradient accumulation.
  - **`utils.py`**: Helpers like model loading, device mapping, and flash attention setup for efficiency.
- **`utils.py`**: General utilities (e.g., seed setting, logging config, data loaders). Imported by most modules.

#### Other Files in `src/social_media_nlp/`
- **`prompts.py`**: Prompt templates for LLMs (e.g., "Classify this tweet as positive/negative/neutral: {text}"). Supports zero/few-shot formatting.
- **`visualization/`** (Insight Generation):
  - **`features.py`**: Extracts NLP features (e.g., TF-IDF vectors, embeddings via SentenceTransformers) for downstream analysis.
  - **`graph.py`**: Builds and visualizes tweet interaction graphs using NetworkX (e.g., retweet networks, centrality measures for influencers).
  - **`hashtag_clustering.py`**: Clusters hashtags semantically using K-Means on embeddings. Outputs dendrograms and top clusters for trend spotting.
  - **`similarity.py`**: Computes cosine similarity between tweets/hashtags via embeddings. Useful for duplicate detection or topic modeling.

## Contributing

1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Commit changes (`git commit -m 'Add some amazing feature'`).
3. Push to the branch (`git push origin feature/amazing-feature`).
4. Open a Pull Request.

Please add tests for new features and ensure code passes `poetry run black .` for formatting.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (add one if missing).

## Acknowledgments

- Built with [Hugging Face](https://huggingface.co/) datasets and models.
- Thanks to the TweetEval contributors for the dataset.

For issues or questions, open a GitHub issue! 🚀
