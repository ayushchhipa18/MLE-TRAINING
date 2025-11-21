# Diabetes Health Indicator Prediction — End-to-End Full-Stack MLOps Project

A complete production-grade project for building, tracking, deploying, and scaling a multi-class classifier that predicts a person's **Diabetes Status** (Healthy, Pre-Diabetic, Diabetic) using the CDC Health Indicators [dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). 

This project covers ML engineering, MLOps, DevOps, backend, frontend, CI/CD, containerization, and cloud deployment on **AWS ECS Fargate** and **AWS EKS Fargate**.

## 1\. Project Overview

### Goals

  * Build a high-quality multi-class classification model addressing class imbalance (using metrics like Weighted F1-score).
  * Set up reproducible environments using **uv**.
  * Track experiments using **MLflow**.
  * Build a scalable prediction API using **FastAPI**.
  * Develop a **Streamlit** interface for end-users.
  * Automate CI/CD with **GitHub Actions** (including ECR push).
  * Deploy serverlessly to **AWS ECS Fargate** and **AWS EKS Fargate**

### Branching Strategy

We use a simplified Gitflow model for structured development:

  * Create features on `feature/*` branches (e.g., `feature/mlflow-tracking`).
  * Merge to `main` only through **Pull Requests (PRs)** after CI checks pass and a review is completed.

-----

## 2\. Weekly Assignment Plan & Deliverables

This project is structured into four distinct weekly assignments, ensuring an incremental build from core ML to production deployment.

### ➡️ Week 1: Foundation, Data Science, and Reproducibility

| Task | Deliverable | Branch Name Example |
| :--- | :--- | :--- |
| **A1.1: Environment & Repo** | `pyproject.toml`, `uv.lock`, project directory structure. | `feature/setup-env` |
| **A1.2: EDA & Baseline Model Notebooks** | `notebooks/01_eda.ipynb` (Imbalance check) and `notebooks/02_model_dev.ipynb`. | `feature/eda-notebooks` |
| **A1.3: Data Preparation Script** | `src/data_prep.py` (Cleaning, multi-class encoding) and saved `preprocessor.pkl`. | `feature/data-prep-script` |
| **A1.4: Baseline Training Script** | `src/train.py` (Initial training) and saved `model.pkl`. | `feature/baseline-model` |

### ➡️ Week 2: MLOps Tools, Testing, and Documentation

| Task | Deliverable | Branch Name Example |
| :--- | :--- | :--- |
| **A2.1: Inference Pipeline** | `src/predict.py` class ready for API consumption. | `feature/inference-module` |
| **A2.2: Unit and Inference Tests** | `tests/test_data_prep.py` and `tests/test_inference.py` (Must pass `uv run pytest`). | `feature/core-testing` |
| **A2.3: MLflow Integration** | **Updated** `src/train.py` logging **Weighted F1-score** and **Confusion Matrix image** as artifacts. | `feature/mlflow-tracking` |
| **A2.4: Sphinx Documentation** | `docs/` initialized; API reference generated; Rationale for imbalance and metrics documented. | `feature/sphinx-docs` |

### ➡️ Week 3: Full-Stack Application & CI/CD Automation

| Task | Deliverable | Branch Name Example |
| :--- | :--- | :--- |
| **A3.1: FastAPI Production Backend** | `app/api.py` with `/predict` endpoint and Pydantic validation. | `feature/fastapi-backend` |
| **A3.2: Streamlit Interface** | `app/frontend.py` with input form and display of multi-class predictions/probabilities. | `feature/streamlit-frontend` |
| **A3.3: API Integration Test** | `tests/test_integration.py` hitting the `/predict` endpoint. | `feature/api-integration-test`|
| **A3.4: Optimized Dockerfile** | **`Dockerfile`** using a multi-stage build, leveraging `uv` for deterministic builds. | `feature/dockerization` |
| **A3.5: GitHub Actions CI** | `.github/workflows/ci.yml` (Must run linting, tests, and `docker build` on every PR). | `feature/github-ci` |

### ➡️ Week 4: Cloud Deployment and Scalability

| Task | Deliverable | Branch Name Example |
| :--- | :--- | :--- |
| **A4.1: ECR Push CI/CD Step** | **Updated** `ci.yml` (Pushes tagged image to ECR on merge to `main`). | `release/v1.0-ecr` |
| **A4.2: AWS ECS Fargate Setup** | IaC/Documentation in `infra/ecs/` for ECS Cluster, Task Definition (Fargate), Service, and ALB. | `deployment/ecs-fargate` |
| **A4.3: AWS EKS Fargate Setup** | IaC/Documentation for provisioning EKS Cluster and Fargate Profiles. | `deployment/eks-config` |
| **A4.4: Kubernetes Manifests** | YAML files in `infra/eks/` for Deployment, Service, and Ingress. | `deployment/eks-final` |
| **A4.5: Final Repository Polish** | Completed `Makefile` and final `README.md` review. | `release/v1.0-final` |

-----

## 3\. Repository Structure

```
project/
│
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_dev.ipynb
│
├── src/
│   ├── data_prep.py    # Data cleaning and feature engineering
│   ├── train.py        # Model training and MLflow tracking
│   ├── predict.py      # Inference Pipeline (loads artifacts)
│   └── utils.py
│
├── app/
│   ├── api.py          # FastAPI backend (A3.1)
│   └── frontend.py     # Streamlit UI (A3.2)
│
├── tests/
│   ├── test_data_prep.py
│   ├── test_inference.py
│   └── test_integration.py
│
├── docs/
│   └── sphinx/...      # Sphinx documentation (A2.4)
│
├── infra/
│   ├── ecs/            # AWS ECS Fargate IaC/Config (A4.2)
│   └── eks/            # AWS EKS Fargate K8s Manifests (A4.4)
│
├── Dockerfile          # Optimized multi-stage build (A3.4)
├── .github/workflows/ci.yml # GitHub Actions CI/CD (A3.5, A4.1)
├── pyproject.toml
├── uv.lock             # Locked dependencies for reproducibility
└── README.md
```

-----

## 4\. Environment Setup (using `uv`)

This project uses `uv` for deterministic, fast environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | bash

# Create virtual environment
uv venv

# Install locked dependencies
uv sync

# Run scripts (example)
uv run python src/train.py
uv run pytest
uv run uvicorn app.api:app --reload
```
