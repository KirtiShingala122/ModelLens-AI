# ModelLens AI

## Project Overview
ModelLens AI is an intelligent, open-source LLM discovery and benchmarking platform. Independently implemented from scratch, it evaluates real-time, live model and benchmark data to recommend the perfect large language model for your specific use cases.

## Problem and Solution
**Problem:** The rapid proliferation of open-source LLMs has made it incredibly difficult for developers and organizations to keep track of the latest models, interpret disparate benchmark scores, and confidently choose the right model for their specific tasks.

**Solution:** ModelLens AI cuts through the noise. By aggregating live data from the Hugging Face Open LLM Leaderboard (v2), it applies a dynamic filtering and deterministic scoring engine to evaluate models against user-defined constraints—such as use case, size limits, and required languages. It delivers a personalized, ranked list of models with transparent insights into exactly *why* a model is recommended.

## Features
- **Live Data Ingestion:** Automatically fetches the latest leaderboard benchmarks dynamically (IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO).
- **Smart Model Filtering:** Prunes thousands of models based on parameter count bounds, language support, and task alignment.
- **AI-Powered Recommender Engine:** A deterministic scoring algorithm that weights different benchmarks dynamically depending on the selected task (e.g., Coding & Math, General Purpose, Creative Writing).
- **Transparent Insights:** Auto-generates "Why this model?" explanations alongside categorized strengths and trade-offs.
- **Rich Visualizations:** Interactive Plotly comparison charts highlighting relative benchmark performance.
- **Clean UI:** A modern, responsive, emoji-free Streamlit interface built with premium styling aesthetics.

## Architecture
1. **UI Layer (`src/ui/`)**: A modular Streamlit interface that captures user constraints and visualizes the results seamlessly.
2. **Data Ingestion (`src/data/`)**: Uses `requests` to fetch and parse live data from the Hugging Face Datasets API, with robust schema enforcement and caching.
3. **Filtering Engine (`src/models/filter.py`)**: A multi-stage pipeline that drops irrelevant models based on size limits, language compatibility, and use-case applicability.
4. **Scoring Engine (`src/models/scorer.py`)**: Calculates 0-100 normalized scores for every candidate by weighting specific benchmark columns according to the task preset.
5. **Recommendation Engine (`src/models/recommender.py`)**: Orchestrates the backend by sorting the highest scorers and generating human-readable insights.

## Technology Stack
- **Python 3.11+**
- **Streamlit** (Web framework & UI)
- **Pandas** (Data manipulation & schema enforcement)
- **Plotly** (Interactive data visualization)
- **Requests** (API communication)

## Screenshots
> *(Placeholder: Add screenshots of the Recommender Dashboard, Sidebar, and Comparison Charts here)*

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ModelLens_AI.git
   cd ModelLens_AI
   ```

## Environment Setup
It is recommended to use a virtual Python environment.
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```
Install the core dependencies:
```bash
pip install -r requirements.txt
```

## Run Command
Start the ModelLens AI platform with the following command:
```bash
streamlit run app.py
```
*(By default, this will launch the application on `http://localhost:8501`)*

## Example Workflow
1. Navigate to the **Recommender** via the sidebar.
2. Select your **Use Case** (e.g., *Coding & Math*).
3. Set your **Model Size Preference** (e.g., *Medium (7–14B)*).
4. Enter your **Required Languages** (e.g., *en, zh*).
5. Adjust the **Performance Priority** threshold if you require a strict minimum baseline.
6. Click **Find Models**. The engine will crunch the live datasets, filter out incompatible models, score the survivors, and present you with the Top 5 contenders alongside comparative graphs.

## Limitations
- **API Rate Limits:** Because the app queries the live Hugging Face Datasets Server dynamically, heavy traffic can sometimes lead to temporary API timeouts or blank datasets.
- **Dataset Splitting:** To balance speed and API constraints, the app fetches a batch of up to 500 models at a time. Exhaustive searching of all 3,000+ open models would require a persistent offline local cache.
- **Schema Evolution:** The tool depends on the underlying structure of the Open LLM Leaderboard (v2). Changes to their upstream column names or benchmarks may require corresponding mapping updates in `src/utils/constants.py`.

## Future Scope
- **Integration with LMSYS Chatbot Arena:** Incorporate blind human preference ELO scores alongside static benchmarks for a more holistic evaluation.
- **Hardware Calculators:** Provide VRAM estimation and quantization tracking (GGUF/AWQ) to determine if a recommended model can comfortably run on the user's specific local hardware.
- **Background Syncing:** Implement an offline daemon to pull the full Hugging Face leaderboard daily into a local SQLite/DuckDB database to bypass API limits and support instant querying across the entire dataset.
