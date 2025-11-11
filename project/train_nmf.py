import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


def pg_nnls(A, b, max_iter=500, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6):
    """
    Projected Gradient method for NNLS: min_x>=0 0.5*||A x - b||^2

    Parameters
    - A: (m, r) matrix
    - b: (m,) vector
    - max_iter: maximum number of PG iterations
    - alpha0: initial step size
    - beta: backtracking factor (0 < beta < 1)
    - c: Armijo constant (small, e.g., 1e-4)
    - tol: tolerance on step norm

    Returns
    - x: (r,) nonnegative solution
    """
    m, r = A.shape
    x = np.zeros(r, dtype=np.float64)
    alpha = float(alpha0)

    def f(xv):
        rvec = A @ xv - b
        return 0.5 * np.dot(rvec, rvec)

    for _ in range(max_iter):
        g = A.T @ (A @ x - b)
        x_proj = np.maximum(0.0, x - alpha * g)
        d = x_proj - x

        nd = np.linalg.norm(d)
        if nd <= tol * max(1.0, np.linalg.norm(x)):
            x = x_proj
            break

        fx = f(x)
        gd = np.dot(g, d)
        t = 1.0

        # Backtracking line search with Armijo condition
        while True:
            z = x + t * d  # remains feasible as convex combination of feasible points
            fz = f(z)
            if fz <= fx + c * t * gd:
                x = z
                # Optionally increase step a bit for next iteration
                alpha = min(alpha / beta, 1.0)
                break
            t *= beta
            if t < 1e-12:
                # fall back to projected point if line search stalls
                x = x_proj
                break

    return x


def als_pg(V, r=32, iters=25, max_nnls_iter=500, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6, random_state=0, verbose=False):
    """
    NMF via Alternating Least Squares, subproblems solved by PG-NNLS.

    V ~ W H where V in R^{m x n}, W in R^{m x r}, H in R^{r x n}

    Updates:
    - With H fixed, update each row w_i by solving min ||H^T w_i^T - v_i^T||^2 s.t. w_i >= 0
    - With W fixed, update each column h_j by solving min ||W h_j - v_j||^2 s.t. h_j >= 0
    """
    m, n = V.shape
    rng = np.random.default_rng(random_state)
    # Small positive init to avoid zero-locking
    W = np.clip(rng.random((m, r)) * 0.1, 1e-8, None)
    H = np.clip(rng.random((r, n)) * 0.1, 1e-8, None)

    for it in range(iters):
        # Update W row-wise
        At = H.T  # (n, r)
        for i in range(m):
            b = V[i, :]
            W[i, :] = pg_nnls(At, b, max_iter=max_nnls_iter, alpha0=alpha0, beta=beta, c=c, tol=tol)

        # Update H column-wise
        A = W  # (m, r)
        for j in range(n):
            b = V[:, j]
            H[:, j] = pg_nnls(A, b, max_iter=max_nnls_iter, alpha0=alpha0, beta=beta, c=c, tol=tol)

        if verbose:
            WH = W @ H
            err = np.linalg.norm(WH - V, ord='fro')
            print(f"[ALS-PG] iter {it+1}/{iters} | frob err = {err:.6f}")

    return W, H


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

    # Save artifacts
    np.savez("nmf_artifacts.npz", W=W, img_shape=np.array([8, 8], dtype=np.int32))
    joblib.dump(knn, "knn.joblib")
    print("Artifacts saved: nmf_artifacts.npz, knn.joblib")


if __name__ == "__main__":
    main()

