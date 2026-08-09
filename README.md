# Shoppers Intent MLOps

A production-grade end-to-end ML Classification project that predicts whether an online shopper will make a purchase or not.

## Live Demo
- **Frontend**: https://shoppers-intent-mlops-vxo7.onrender.com/
- **Swagger UI**: https://shoppers-intent-mlops.onrender.com/docs

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Source DB | MySQL |
| Data Warehouse | PostgreSQL |
| ETL | Python (Extract → Transform → Load) |
| Data Validation | Pandas |
| ML Models | Random Forest, XGBoost, LightGBM |
| Hyperparameter Tuning | GridSearchCV |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API | FastAPI |
| Monitoring | Evidently AI |
| Scheduler | APScheduler |
| Logging | Python logging module |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Registry | Docker Hub |
| Deployment | Render Free Tier |

## Architecture


## Project Structure

```text
shoppers-intent-mlops/
├── api/
│   ├── templates/index.html     # frontend html
│   ├── static/style.css         # frontend css
│   ├── static/script.js         # frontend javascript
│   └── main.py                  # fastapi app
├── data/
│   ├── online_shoppers_intention.csv  # raw dataset
│   └── reload_data.py           # csv to mysql loader
├── etl/
│   ├── extract.py               # extract from mysql
│   ├── transform.py             # clean, validate, encode
│   └── load.py                  # load to postgresql
├── ml/
│   ├── preprocess.py            # scaling, selection, smote
│   ├── train.py                 # train rf+xgb+lgbm with tuning
│   ├── evaluate.py              # evaluate saved model
│   ├── predict.py               # make predictions
│   └── mlflow_tracker.py        # mlflow tracking + registry
├── monitoring/
│   └── drift_report.py          # evidently drift report
├── scheduler/
│   └── retrain_job.py           # apscheduler auto retrain
├── tests/
│   └── test_api.py              # pytest tests
├── utils/
│   └── logger.py                # common logging module
├── logs/                        # log files
├── .github/workflows/
│   └── ci_cd.yml                # github actions ci/cd
├── .env.example                 # env variables template
├── .gitignore                   # git ignore rules
├── .dockerignore                # docker ignore rules
├── Dockerfile                   # docker configuration
├── render.yaml                  # render deployment config
└── requirements.txt             # python dependencies


## Dataset
- **Source**: Online Shoppers Purchasing Intention Dataset
- **Rows**: 12,330
- **Target**: Revenue (Will Purchase or Not)
- **Features**: 17 behavioral and session features
- **Class Imbalance**: Fixed using SMOTE

## ML Pipeline
1. **Extract** — Raw data from MySQL
2. **Validate** — Schema, nulls, value ranges checked
3. **Transform** — Encode, clean, deduplicate
4. **Load** — Push to PostgreSQL warehouse
5. **Preprocess** — StandardScaler + SelectKBest + SMOTE
6. **Train** — RF + XGBoost + LightGBM with GridSearchCV
7. **Select** — Best model by ROC AUC auto-selected
8. **Track** — MLflow experiment tracking + model registry
9. **Predict** — FastAPI prediction endpoint
10. **Monitor** — Evidently data drift report
11. **Retrain** — APScheduler daily 2 AM auto retrain

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 91.82% |
| Precision | 89.96% |
| Recall | 94.33% |
| F1 Score | 92.09% |
| ROC AUC | 97.32% |

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Frontend UI |
| GET | /health | Health check |
| POST | /predict | Make prediction |
| GET | /docs | Swagger UI |

## Setup and Run Locally

### Prerequisites
- Python 3.11
- MySQL 8.0
- PostgreSQL 15+
- Docker

## Logging
All modules log to:
- **Console** — real time output
- **`logs/app.log`** — persistent log file

## Author
Nihar Gudadhe
