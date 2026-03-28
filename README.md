# Wholesale Customer Clustering — Airflow Pipeline

This project uses Apache Airflow to orchestrate a simple ML pipeline that clusters wholesale customers based on their spending patterns. The whole thing runs in Docker, so there's nothing to install locally beyond Docker Desktop.

The dataset comes from the [UCI Wholesale Customers dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers), which tracks annual spending across categories like Fresh, Milk, Grocery, etc. for 440 clients of a wholesale distributor.

---

## What This Lab Covers

- Setting up Apache Airflow with Docker Compose
- Writing a DAG that chains together four Python tasks
- Passing data between tasks using XCom (with base64-encoded pickle serialization)
- Running DBSCAN clustering with silhouette score evaluation
- Saving and loading ML models inside an Airflow pipeline

---

## The Pipeline

The DAG (`Wholesale_Clustering_Pipeline`) runs four tasks in sequence:

**load_data** → **data_preprocessing** → **build_save_model** → **load_model_predict**

Here's what each task actually does:

**load_data** — Reads `wholesale.csv`, serializes the dataframe, and pushes it to XCom so the next task can pick it up.

**data_preprocessing** — Pulls the dataframe from XCom, drops nulls, selects three features (Fresh, Milk, Grocery), and scales them with StandardScaler. Pushes the scaled array back to XCom.

**build_save_model** — Runs DBSCAN across a range of `eps` values (0.3 to 1.7), calculates silhouette scores for each, keeps the best model, and saves it as a pickle file. Returns the scores dict via XCom.

**load_model_predict** — Loads the saved model, finds the best eps from the scores, runs predictions on `test.csv`, and prints cluster assignments.

---

## Project Structure

```
lab5/
├── dags/
│   ├── airflow.py              # The DAG definition
│   ├── data/
│   │   ├── wholesale.csv       # Training data (440 wholesale customers)
│   │   └── test.csv            # Small test set (5 samples, pre-scaled)
│   ├── model/                  # Saved models land here at runtime
│   └── src/
│       ├── __init__.py
│       └── pipeline.py         # All the ML logic (load, preprocess, train, predict)
├── docker-compose.yaml         # Airflow + Postgres + Redis + Celery
├── setup.sh                    # Optional cleanup/init script
├── .env                        # AIRFLOW_UID setting
└── README.md
```

---

## Prerequisites

- **Docker Desktop** installed and running ([get it here](https://docs.docker.com/get-docker/))
- At least 4GB of memory allocated to Docker (8GB is better)

That's it. No Python install needed — everything runs inside the containers.

---

## How to Run

### Step 1: Create the required directories and .env

```bash
mkdir -p ./logs ./plugins ./config
echo "AIRFLOW_UID=50000" > .env
```

On Mac/Linux, use `echo -e "AIRFLOW_UID=$(id -u)" > .env` instead.

### Step 2: Initialize the Airflow database

```bash
docker compose up airflow-init
```

Wait until you see `airflow-init-1 exited with code 0`. This creates the database tables and the admin user.

### Step 3: Start everything

```bash
docker compose up
```

This spins up Postgres, Redis, the Airflow webserver, scheduler, worker, and triggerer. It also installs `pandas`, `scikit-learn`, and `kneed` in each container on startup. Give it a minute or two.

### Step 4: Open the Airflow UI

Go to **http://localhost:8080** and log in with:
- Username: `airflow2`
- Password: `airflow2`

### Step 5: Trigger the DAG

Find **Wholesale_Clustering_Pipeline** in the DAG list. Toggle it on (the switch on the left), then click the play button to trigger a run. You can watch each task turn green in the Graph view.

### Step 6: Clean up

```bash
docker compose down -v
```

The `-v` removes the Postgres volume and any other Docker volumes created during the run.

---
