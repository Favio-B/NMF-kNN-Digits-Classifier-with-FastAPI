import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from app.nmf_core import pg_nnls, als_pg


def main():
    # Load and normalize data
    digits = load_digits()
    X = digits.data.astype(np.float64)  # (n_samples, 64)
    y = digits.target

    # Normalize to [0,1]; original digits are in [0,16]
    X = X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build V from training data: V = X_train^T -> (m=64, n_train)
    V_train = X_train.T

    # Factorize with ALS-PG
    r = 32
    W, H_train = als_pg(
        V_train, r=r, iters=25, max_nnls_iter=300, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6, random_state=0, verbose=True
    )

    # Train kNN on activations H of the training set
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(H_train.T, y_train)

    # Project test samples onto W to get H_test
    V_test = X_test.T  # (64, n_test)
    n_test = V_test.shape[1]
    H_test = np.zeros((W.shape[1], n_test), dtype=np.float64)
    for j in range(n_test):
        b = V_test[:, j]
        H_test[:, j] = pg_nnls(W, b, max_iter=300, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6)

    # Evaluate
    y_pred = knn.predict(H_test.T)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (kNN on H): {acc:.4f}")

    # Save artifacts at project root (two levels up from this file)
    base_dir = Path(__file__).resolve().parent.parent
    np.savez(base_dir / "nmf_artifacts.npz", W=W, img_shape=np.array([8, 8], dtype=np.int32))
    joblib.dump(knn, base_dir / "knn.joblib")
    print("Artifacts saved: nmf_artifacts.npz, knn.joblib")


if __name__ == "__main__":
    main()
