# Credit-Probability-Model

## Reproducible Environment

Follow these steps to reproduce the project environment locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/binia01/Credit-Probability-Model.git
   cd Credit-Probability-Model
   ```

2. **Check your Python version (3.10+ recommended)**
   ```bash
   python3 --version
   ```

3. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   ```

4. **Activate the virtual environment**
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\.venv\Scripts\activate
     ```

5. **Upgrade pip and install dependencies**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Verify the environment (optional)**
   ```bash
   pip check
   ```

---

## Running the Project

### 1. Exploratory Data Analysis (EDA)

Start by exploring the data with the EDA notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

Or use JupyterLab:
```bash
jupyter lab notebooks/EDA.ipynb
```

This step helps you understand the data structure, distributions, and patterns before preprocessing.

---

### 2. Data Processing

Process raw data and prepare it for model training:

```bash
python -m src.data_processing
```

This runs the data pipeline to create `data/processed/model_ready_data.csv` from the raw transaction data.

---

### 3. Model Training

Train all models (Logistic Regression, Random Forest, and RFM) with MLflow tracking:

```bash
python -m src.train
```

This will:
- Train and tune models using GridSearchCV
- Log parameters, metrics, and artifacts to MLflow
- Register models in the MLflow Model Registry
- Save models locally to the `models/` directory

**View MLflow UI:**
```bash
 mlflow ui --host 0.0.0.0 --port 5050 --serve-artifacts --default-artifact-root ./mlruns --allowed-hosts "*";
```
Then open http://localhost:5000 in your browser.

---

### 4. FastAPI Server

Start the REST API for credit scoring predictions:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Get credit risk prediction

**API Documentation:** Open http://localhost:8000/docs after starting the server.

---

### 5. Streamlit Dashboard

Launch the interactive dashboard for credit risk scoring:

```bash
streamlit run src/dashboard.py
```

> **Note:** The API server must be running for the dashboard to work. Start the API first (see step 3).

The dashboard will be available at http://localhost:8501.

---

## Project Structure

```
Credit-Probability-Model/
├── data/
│   ├── raw/                    # Raw transaction data
│   └── processed/              # Processed model-ready data
├── models/                     # Saved model artifacts
├── mlruns/                     # MLflow tracking data
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── plots/                      # Generated visualizations
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI application
│   │   └── pydantic_models.py # Request/Response schemas
│   ├── assets/                # Static assets (images)
│   ├── dashboard.py           # Streamlit dashboard
│   ├── data_processing.py     # Data pipeline
│   ├── predict.py             # Prediction/scoring engine
│   └── train.py               # Model training script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image definition
├── docker-compose.yml         # Multi-service orchestration
└── README.md
```

---

## Docker Deployment

### Prerequisites

- Docker and Docker Compose installed
- Models trained and saved in `models/` directory

### Build and Run API

```bash
# Build and start the API service
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### Service Available

| Service | URL                   | Description             |
|---------|-----------------------|-------------------------|
| API     | http://localhost:8000 | Credit Scoring REST API |

**API Endpoints:**
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /predict` - Get credit risk prediction

### Stop Service

```bash
docker-compose down
```

### Build Image Only

```bash
docker build -t credit-scoring-api .
```

---

## Quick Start (Full Pipeline)

Run the entire pipeline in order:

```bash
# 1. Explore data (EDA)
jupyter notebook notebooks/EDA.ipynb

# 2. Process data
python -m src.data_processing

# 3. Train models
python -m src.train

# 4. Start API (in a new terminal)
uvicorn src.api.main:app --reload --port 8000

# 5. Start dashboard (in another terminal)
streamlit run src/dashboard.py
```

---

## Quick Start (Docker)

```bash
# 1. Explore data (EDA)
jupyter notebook notebooks/EDA.ipynb

# 2. Train models first (local)
python -m src.data_processing
python -m src.train

# 3. Start API with Docker
docker-compose up --build -d

# 4. Access the API
# http://localhost:8000/docs
```