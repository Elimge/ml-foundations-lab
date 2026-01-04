# 🧠 ML Foundations Lab | Training

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Llama 3](https://img.shields.io/badge/GenAI-Llama_3-0467DF?style=for-the-badge&logo=meta&logoColor=white)

**An experimentation workbench designed to bridge the gap between Classical Machine Learning, Deep Learning, and Generative AI.**

This repository serves as a training ground for the Belabs engineering team, focusing on the fundamentals of neural networks, data processing pipelines, and the integration of Local LLMs (Large Language Models) for semantic reasoning.

---

## 📂 Project Structure

The project follows a modular Data Science cookiecutter structure, optimized for containerized development.

```bash
ml-foundations-lab/
├── .devcontainer/       # Configuration for VS Code Remote Containers
│   └── devcontainer.json
├── data/                # Data storage (Ignored by Git)
│   ├── processed/       # Cleaned data ready for training
│   └── raw/             # Immutable original data
├── docker/              # Container infrastructure
│   └── Dockerfile       # Image definition with Python, FFmpeg, Git
├── notebooks/           # Jupyter Notebooks (Sequential Labs)
│   ├── 01_data_generation.ipynb
│   ├── 02_training_classic.ipynb
│   ├── 03_genai_classification.ipynb
│   └── 04_neural_network_mnist.ipynb
├── src/                 # Source code and artifacts
│   ├── models/          # Serialized models (.pkl)
│   └── utils/           # Helper functions
├── docker-compose.yml   # Orchestration for Jupyter & Volumes
├── requirements.txt     # Python dependencies pinned versions
└── README.md            # Project documentation
```

---

## 🎯 Learning Modules

This laboratory is divided into four sequential stages:

### 1. Data Engineering & Generation
*   **Goal:** Understand the importance of "Ground Truth" and data distribution.
*   **Tech:** `Pandas`, `NumPy`.
*   **Outcome:** Synthetic generation of a dataset (Avocados vs. Mangoes) with controlled noise and outliers.

### 2. Classical Machine Learning
*   **Goal:** Solve a binary classification problem using statistical methods.
*   **Tech:** `Scikit-Learn`, `Logistic Regression`.
*   **Outcome:** A trained model achieving **~97% Accuracy** with explainable feature weights.

### 3. Generative AI (The "Reasoning" Layer)
*   **Goal:** Use an LLM to handle edge cases where mathematical boundaries fail, providing semantic explanations.
*   **Tech:** `Llama 3`, `Ollama API`.
*   **Outcome:** A hybrid system where the AI explains *why* a specific fruit was classified based on descriptive text.

### 4. Deep Learning (Computer Vision)
*   **Goal:** Introduction to Neural Networks, Layers, and Backpropagation.
*   **Tech:** `PyTorch`, `torchvision`, `MNIST Dataset`.
*   **Outcome:** A Multi-Layer Perceptron (MLP) capable of recognizing handwritten digits with **~96% Accuracy**.

---

## 🛠️ Architecture & Stack

We use a **Cloud-Native approach** for local development to ensure reproducibility across the team (35+ engineers).

*   **Docker & Dev Containers:** Eliminates "it works on my machine" issues. The entire environment (OS, Libraries, Tools) is defined as code.
*   **Ollama (Host):** Runs the Llama 3 inference engine on the host machine (Windows/Mac/Linux) to leverage hardware acceleration (Metal/CUDA/AVX).
*   **Jupyter Lab (Container):** connects to Ollama via the internal Docker network gateway (`host.docker.internal`).

---

## 🚀 Getting Started

### Prerequisites
1.  **Docker Desktop** installed and running.
2.  **VS Code** with the *Dev Containers* extension installed.
3.  **Ollama** installed on your host machine.
    *   Run: `ollama run llama3` in your terminal to download the model.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/ml-foundations-lab.git
    cd ml-foundations-lab
    ```

2.  **Launch the Environment:**
    *   Open the folder in **VS Code**.
    *   You will see a prompt: *"Folder contains a Dev Container configuration file. Reopen to folder to develop in a container"*.
    *   Click **Reopen in Container**.

    *> VS Code will build the Docker image, install all dependencies defined in `requirements.txt`, and set up the Python kernel automatically.*

3.  **Run the Labs:**
    Open the `notebooks/` folder and execute the notebooks in order (01 to 04).

---

> **Engineer Miguel Canedo**  
> *Building the future of behavioral analysis with AI.*