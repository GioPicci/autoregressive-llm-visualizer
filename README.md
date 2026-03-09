# Autoregressive LLM Explorer

An interactive web application built with **Streamlit** to explore "under the hood" how an autoregressive Large Language Model (LLM) works (based on the GPT-2 architecture).  

This project is educational: it allows step-by-step visualization of what happens inside the model during text generation, from the initial prompt to the selection of the next word.

## Features (The 5 Phases)

The interface guides the user through key LLM concepts:  
1. **Tokenization:** Visualizes how text is split into subwords and converted into numerical IDs.  
2. **Embeddings:** Interactive 3D projection of token vectors and raw data visualization.  
3. **Self-Attention:** Heatmaps to explore how tokens communicate with each other, navigable by *Layer* and *Head*.  
4. **Probabilities and Sampling:** Dynamic bar chart of the next word probabilities. Allows experimentation with parameters such as *Temperature* and *Top-P*.  
5. **Iterative Generation:** Step-by-step token generation or "Autopilot" mode to complete sentences automatically.

## Technologies Used
* **Language:** Python 3.10  
* **Package Manager:** [uv](https://github.com/astral-sh/uv) (Extremely fast and modern)  
* **Frontend:** Streamlit  
* **Core ML:** PyTorch & HuggingFace Transformers  
* **Visualization:** Plotly & NumPy

## Installation and Running

This project uses `uv` for fast and efficient dependency management and virtual environment setup (`pyproject.toml` and `uv.lock`).

### 1. Prerequisites
Make sure **Python 3.10** and **uv** are installed on your system.  
*(If you don’t have `uv`, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` on Mac/Linux or via `pip install uv`).*

### 2. Setup
Clone or download this repository, open a terminal in the project root folder (`AutoregressiveLLMBasicsCourse`) and run:

```bash
# Create the virtual environment and sync dependencies from uv.lock
uv sync
```

### 3. Run the Application

Once dependencies are installed, you can launch the Streamlit interface with a single command:

```bash
# Run the Streamlit app
uv run streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

### 4. Project Structure

- app.py: Entry-point for the Streamlit app; contains the UI and interaction logic.
- src/:
  - llm_engine.py / base_engine.py: Handle model loading (GPT-2), tokenization, and forward pass.
  - visualizer.py: Contains Plotly functions to generate 3D plots, heatmaps, and bar charts.
- pyproject.toml & uv.lock: uv configuration files for project dependencies.
