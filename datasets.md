# DATASETS

This document describes all datasets used in the thesis project:

**"Multimodal Retail Decision Intelligence Using Graph Neural Networks and Large Language Models"**

---

# Overview

The project combines multiple public retail datasets to support:
- multimodal learning
- graph construction
- recommendation systems
- customer behavior analysis
- retail intelligence

The datasets used are:

1. Amazon Reviews 2023
2. RetailRocket E-Commerce Dataset
3. Instacart Online Grocery Basket Analysis Dataset

---

# Directory Structure

Store datasets using the following structure:

```text
data/
│
├── raw/
│   ├── amazon/
│   ├── retailrocket/
│   └── instacart/
│
└── processed/
```

---

# 1. Amazon Reviews 2023 Dataset

## Description

Amazon Reviews 2023 is a large-scale multimodal e-commerce dataset containing:
- product reviews
- ratings
- product metadata
- textual descriptions
- images
- customer interactions

This dataset is used for:
- multimodal embedding generation
- semantic product understanding
- LLM-driven retail reasoning
- recommendation modeling

---

## Dataset Features

| Component | Description |
|---|---|
| Reviews | Customer textual reviews |
| Ratings | Product rating scores |
| Metadata | Product information |
| Images | Product images |
| Categories | Product category labels |

---

## Official Source

:contentReference[oaicite:0]{index=0}

---

## Storage Location

```text
data/raw/amazon/
```

---

## Usage in Thesis

Used for:
- text embedding generation
- image embedding generation
- multimodal representation learning
- LLM-enhanced semantic reasoning

---

# 2. RetailRocket E-Commerce Dataset

## Description

RetailRocket is an e-commerce interaction dataset containing:
- user browsing behavior
- product views
- add-to-cart events
- purchase transactions
- category metadata

This dataset is mainly used for:
- recommendation systems
- graph construction
- interaction modeling
- customer behavior analysis

---

## Dataset Features

| File | Description |
|---|---|
| events.csv | User interaction events |
| item_properties.csv | Product metadata |
| category_tree.csv | Category hierarchy |

---

## Official Source

:contentReference[oaicite:1]{index=1}

---

## Storage Location

```text
data/raw/retailrocket/
```

---

## Usage in Thesis

Used for:
- user-item graph generation
- session-based recommendation
- graph neural network experiments
- retail interaction analysis

---

# 3. Instacart Online Grocery Basket Analysis Dataset

## Description

The Instacart dataset contains grocery purchasing behavior data including:
- customer orders
- reordered products
- basket interactions
- aisle and department metadata

This dataset supports:
- basket prediction
- demand forecasting
- graph-based retail intelligence
- customer purchasing analysis

---

## Dataset Features

| File | Description |
|---|---|
| orders.csv | Customer order information |
| products.csv | Product catalog |
| order_products.csv | Basket-product mapping |
| aisles.csv | Product aisles |
| departments.csv | Department metadata |

---

## Official Source

:contentReference[oaicite:2]{index=2}

---

## Storage Location

```text
data/raw/instacart/
```

---

## Usage in Thesis

Used for:
- customer-product interaction graphs
- demand prediction
- sequential purchase modeling
- recommendation experiments

---

# Dataset Preparation Workflow

The preprocessing pipeline performs:

1. Data cleaning
2. Missing value handling
3. Feature normalization
4. Text preprocessing
5. Image preprocessing
6. Graph edge construction
7. Multimodal feature generation
8. Train-validation-test split generation

Processed outputs are stored in:

```text
data/processed/
```

---

# Storage Requirements

| Dataset | Approximate Size |
|---|---|
| Amazon Reviews 2023 | 10GB – 100GB+ |
| RetailRocket | ~500MB |
| Instacart | ~700MB |

---

# Hardware Environment

Experiments were conducted using:
- Apple Mac Mini M4
- 24GB RAM
- macOS

---

# Data Preprocessing

Run the preprocessing notebook before experiments:

```text
RQ0_data_preparation.ipynb
```

This notebook:
- prepares datasets
- aligns schemas
- generates multimodal embeddings
- creates graph-ready data

---

# Notes

- Large raw datasets are intentionally excluded from this repository.
- Only processed outputs, experiment summaries, and figures are included.
- Download datasets manually using the provided official links.

---

# Ethical Considerations

All datasets used are:
- publicly available
- anonymized
- intended for academic research purposes

No personally identifiable information (PII) is used.

---

# Recommended Local Directory Layout

```text
data/
│
├── raw/
│   ├── amazon/
│   ├── retailrocket/
│   └── instacart/
│
└── processed/
```