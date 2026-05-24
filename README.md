# Multimodal Retail Decision Intelligence Using Graph Neural Networks and Large Language Models

## Overview

This repository contains the implementation, experiments, and research artifacts for the master's thesis:

**"Multimodal Retail Decision Intelligence Using Graph Neural Networks and Large Language Models"**

The project investigates how multimodal retail data can be integrated using:

- Graph Neural Networks (GNNs)
- Large Language Models (LLMs)
- Multimodal embeddings
- Explanation-oriented evaluation techniques

The system combines:

- transactional data
- product metadata
- textual reviews
- product images
- graph relationships

to support retail decision intelligence tasks such as:

- recommendation systems
- demand prediction
- customer behavior analysis
- retail analytics evaluation

---

# Research Objectives

The research focuses on the following objectives:

1. Construct multimodal retail knowledge representations
2. Learn graph-based relationships between retail entities
3. Integrate LLM-assisted semantic understanding
4. Improve recommendation and predictive performance
5. Provide interpretable retail intelligence outputs

---

# Repository Structure

```text
multimodal-retail-decision-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── RQ0_data_preparation.ipynb
│   ├── RQ1_multimodal_embeddings.ipynb
│   ├── RQ2_graph_construction.ipynb
│   ├── RQ3_gnn_modeling.ipynb
│   ├── RQ4_llm_reasoning.ipynb
│   ├── RQ5_explainability.ipynb
│   └── RQ6_evaluation.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── models/
│
├── docs/
│   └── architecture/
│
├── figures/
├── references/
│
├── requirements.txt
├── environment.yml
├── DATASETS.md
├── LICENSE
└── README.md
```

---

# Experimental Workflow

This repository follows a notebook-based research workflow.

Each notebook corresponds to a distinct research question (RQ) or experimental stage in the thesis pipeline. The notebooks are designed to be executed sequentially for full reproducibility of the reported experiments and outputs.

The implementation prioritizes transparency and research reproducibility over production-oriented software modularization.

---

# Datasets

This project uses the following public datasets:

| Dataset | Purpose | Link |
|---|---|---|
| Amazon Reviews 2023 | Reviews, metadata, multimodal product understanding | https://amazon-reviews-2023.github.io/ |
| RetailRocket Ecommerce Dataset | User interaction modeling | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset |
| Instacart Online Grocery Basket Analysis | Basket and purchasing behavior analysis | https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset |

Dataset download instructions are also provided in:

```text
DATASETS.md
```

Due to dataset licensing and storage constraints, raw datasets are not distributed directly in this repository.

Downloaded datasets should be placed inside:

```text
data/raw/
```

---

# Methodology

The workflow includes:

1. Data preprocessing
2. Feature engineering
3. Multimodal embedding generation
4. Graph construction
5. Graph Neural Network training
6. LLM-assisted semantic analysis and explanation generation
7. Explainability and evaluation

---

# Technologies Used

## Programming

- Python 3.11+

## Machine Learning / Deep Learning

- PyTorch
- PyTorch Geometric
- Scikit-learn
- XGBoost

## NLP / LLMs

- Transformers
- SentenceTransformers
- Hugging Face

## Data Processing

- Pandas
- NumPy
- Dask

## Visualization

- Matplotlib
- Plotly

---

# Installation

## Clone Repository

```bash
git clone https://github.com/BhanuTejaMalineni/multimodal-retail-decision-intelligence.git
cd multimodal-retail-decision-intelligence
```

---

## Create Environment

### Using Conda

```bash
conda env create -f environment.yml
conda activate thesis-env
```

### Using pip

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Dataset Preparation

Run:

```text
RQ0_data_preparation.ipynb
```

This notebook:

- cleans datasets
- aligns schemas
- generates processed outputs

---

## Step 2 — Run Research Question Experiments

Each research question has an independent notebook:

| Notebook | Purpose |
|---|---|
| RQ1_multimodal_embeddings.ipynb | Multimodal feature generation |
| RQ2_graph_construction.ipynb | Retail graph creation |
| RQ3_gnn_modeling.ipynb | Graph neural network experiments |
| RQ4_llm_reasoning.ipynb | LLM-assisted semantic analysis |
| RQ5_explainability.ipynb | Explanation evaluation |
| RQ6_evaluation.ipynb | Performance and robustness evaluation |

For full reproducibility, notebooks should be executed sequentially.

---

# Outputs

Generated outputs include:

- intermediate multimodal embeddings
- graph representations
- trained experimental models
- evaluation summaries
- figures and tables
- experiment logs

Stored in:

```text
outputs/
```

---

# Hardware Used

Development environment:

- Apple Mac Mini M4
- 24GB RAM
- macOS

---

# Reproducibility

To improve experimental reproducibility:

- fixed random seeds are used where applicable
- notebook execution order is documented
- intermediate outputs are stored in the `outputs/` directory
- all experiments are based on publicly available datasets
- dependency versions are specified in `requirements.txt` and `environment.yml`

For full reproduction of results, execute notebooks sequentially from:

```text
RQ0_data_preparation.ipynb
```

through:

```text
RQ6_evaluation.ipynb
```

---

# Research Scope

This repository represents a research-oriented prototype developed for academic experimentation and evaluation. It is not intended as a production retail deployment system.

---

# Citation

If using this work in academic research, please cite:

```bibtex
@mastersthesis{malineni2026,
  title={Multimodal Retail Decision Intelligence Using Graph Neural Networks and Large Language Models},
  author={Malineni, Bhanu Teja},
  year={2026},
  school={University of Europe for Applied Sciences}
}
```

---

# License

This project is intended for academic and research purposes.

---

# Author

Bhanu Teja Malineni  
M.Sc. Software Engineering  
University of Europe for Applied Sciences