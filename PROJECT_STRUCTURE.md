# Project Structure

This document explains the organization of the thesis repository.

---

# Root Directory

```text
Thesis/
```

Contains all project files.

---

# data/

Stores datasets.

## raw/
Original downloaded datasets.

## processed/
Preprocessed datasets and graph-ready outputs.

---

# notebooks/

Contains research-question-wise Jupyter notebooks.

| Notebook | Purpose |
|---|---|
| RQ0_data_preparation.ipynb | Dataset preprocessing |
| RQ1_overall_framework.ipynb | Overall framework evaluation |
| RQ2_graph_learning.ipynb | Graph learning experiments |
| RQ3_causal_modeling.ipynb | Causal inference |
| RQ4_counterfactual.ipynb | Counterfactual analysis |
| RQ5_llm_explanation.ipynb | LLM explainability |
| RQ6_robustness.ipynb | Robustness evaluation |
| RQ7_end_to_end.ipynb | End-to-end evaluation |
| RQ8_sensitivity.ipynb | Sensitivity analysis |

---

# outputs/

Stores generated experiment outputs.

## figures/
Generated plots and visualizations.

## tables/
Generated tables and CSV outputs.

## prepared/
Processed intermediate outputs.

---

# figures/

Contains thesis-ready figures used in the manuscript.

---

# src/

Contains architecture diagrams and supporting assets.

---

# README.md

Project overview and setup instructions.

---

# DATASETS.md

Dataset descriptions and download instructions.

---

# requirements.txt

Python dependencies.

---

# environment.yml

Conda environment configuration.

---

# LICENSE

Repository license.