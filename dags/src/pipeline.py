import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pickle
import os
import base64


def load_data():
    """
    Loads the Wholesale Customers dataset from CSV, serializes it,
    and returns a base64-encoded string (JSON-safe for XCom).
    """
    print("Loading wholesale customer data...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/wholesale.csv"))
    serialized = pickle.dumps(df)
    return base64.b64encode(serialized).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes the data, drops nulls, selects features,
    applies StandardScaler, and returns base64-encoded scaled data.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()

    # Select 3 spending features for clustering
    features = df[["Fresh", "Milk", "Grocery"]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    serialized = pickle.dumps(scaled_data)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Runs DBSCAN across a range of eps values, records silhouette scores,
    picks the best eps, saves that model, and returns the scores dict.
    """
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    eps_values = [round(0.3 + i * 0.1, 1) for i in range(15)]  # 0.3 to 1.7
    scores = {}
    best_score = -1
    best_model = None

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(data)

        # silhouette score needs at least 2 clusters and not all noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            score = silhouette_score(data, labels)
            scores[str(eps)] = round(score, 4)

            if score > best_score:
                best_score = score
                best_model = dbscan
        else:
            scores[str(eps)] = None

    # Save the best model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best eps: {best_score:.4f}")
    return scores  # dict is JSON-safe


def load_model_predict(filename: str, scores: dict):
    """
    Loads the saved DBSCAN model, reports the best eps from scores,
    and predicts cluster labels on test data.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # Find best eps from scores
    valid = {k: v for k, v in scores.items() if v is not None}
    best_eps = max(valid, key=valid.get)
    print(f"Best eps from silhouette scores: {best_eps} (score: {valid[best_eps]})")

    # Predict on test data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.fit_predict(df)
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    n_noise = list(pred).count(-1)

    print(f"Clusters found: {n_clusters}, Noise points: {n_noise}")

    return int(pred[0])