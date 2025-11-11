import numpy as np


def pg_nnls(A, b, max_iter=500, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6):
    """
    Projected Gradient method for NNLS: min_x>=0 0.5*||A x - b||^2
    Returns nonnegative solution x.
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

        while True:
            z = x + t * d
            fz = f(z)
            if fz <= fx + c * t * gd:
                x = z
                alpha = min(alpha / beta, 1.0)
                break
            t *= beta
            if t < 1e-12:
                x = x_proj
                break

    return x


def als_pg(V, r=32, iters=25, max_nnls_iter=500, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6, random_state=0, verbose=False):
    """
    NMF via Alternating Least Squares, subproblems solved by PG-NNLS.
    V ~ W H where V in R^{m x n}
    """
    m, n = V.shape
    rng = np.random.default_rng(random_state)
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

