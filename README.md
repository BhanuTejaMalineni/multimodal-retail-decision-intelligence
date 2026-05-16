# Multimodal Retail Decision Intelligence Using Graph Neural Networks and Large Language Models

## Overview

This repository contains the implementation, experiments, and research artifacts for the master's thesis:

**"Multimodal Retail Decision Intelligence Using Graph Neural Networks and Large Language Models"**

The project investigates how multimodal retail data can be integrated using:
- Graph Neural Networks (GNNs)
- Large Language Models (LLMs)
- Multimodal embeddings
- Explainable AI techniques

The system combines:
- transactional data
- product metadata
- textual reviews
- product images
- graph relationships

to improve retail decision intelligence tasks such as:
- recommendation systems
- demand prediction
- customer behavior analysis
- explainable retail analytics

---

# Research Objectives

The research focuses on the following objectives:

1. Construct multimodal retail knowledge representations
2. Learn graph-based relationships between retail entities
3. Integrate LLM-driven semantic understanding
4. Improve recommendation and predictive performance
5. Provide interpretable retail intelligence outputs

---

# Repository Structure

```text
Thesis/
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
├── src/
│   ├── data/
│   ├── embeddings/
│   ├── graphs/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── models/
│
├── figures/
├── references/
│
├── requirements.txt
├── environment.yml
├── DATASETS.md
├── LICENSE
├── .gitignore
└── README.md
```

---

# Datasets

This project uses the following public datasets:

| Dataset | Purpose |
|---|---|
| RetailRocket | User interaction modeling |
| Amazon Product Data | Reviews, metadata, multimodal product understanding |
| Instacart Market Basket | Basket and purchasing behavior analysis |

Dataset download instructions are provided in:

`DATASETS.md`

---

# Methodology

The workflow includes:

1. Data preprocessing
2. Feature engineering
3. Multimodal embedding generation
4. Graph construction
5. Graph Neural Network training
6. LLM-based semantic reasoning
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
- Seaborn
- Plotly

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd Thesis
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

```bash
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
| RQ4_llm_reasoning.ipynb | LLM-enhanced retail reasoning |
| RQ5_explainability.ipynb | Explainability analysis |
| RQ6_evaluation.ipynb | Performance evaluation |

---

# Outputs

Generated outputs include:
- trained models
- figures
- evaluation tables
- embeddings
- logs

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

To ensure reproducibility:
- fixed random seeds are used
- notebook execution order is modular
- outputs are version-controlled where feasible

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

This project is for academic and research purposes.

---

# Author

Bhanu Teja Malineni  
M.Sc. Software Engineering  
University of Europe for Applied Sciences
