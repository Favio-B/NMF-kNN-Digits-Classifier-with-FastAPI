from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import joblib
from typing import List


def pg_nnls(A, b, max_iter=300, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6):
    """
    Projected Gradient NNLS used to project new samples onto W.
    Solves: min_x>=0 0.5*||A x - b||^2
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
        gd = float(np.dot(g, d))
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


app = FastAPI()

# Load artifacts on startup
try:
    npz = np.load("nmf_artifacts.npz")
    W = npz["W"].astype(np.float64)
    img_shape = tuple(npz["img_shape"].tolist())
    knn = joblib.load("knn.joblib")
except Exception as e:
    # Delay raising until endpoints are hit to return meaningful error
    W = None
    knn = None
    img_shape = (8, 8)
    _load_err = e

# Optional: Load a pretrained ImageNet classifier for general objects
_imagenet_error = None
try:
    import torch
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    _weights = MobileNet_V2_Weights.DEFAULT
    _imagenet_model = mobilenet_v2(weights=_weights)
    _imagenet_model.eval()
    _preprocess = _weights.transforms()
    _categories: List[str] = _weights.meta.get("categories", [])
except Exception as _e:
    _imagenet_model = None
    _preprocess = None
    _categories = []
    _imagenet_error = _e


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"ok": True, "message": "Use /health or /docs", "img_shape": list(img_shape)}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if W is None or knn is None:
        raise HTTPException(status_code=500, detail="Model artifacts not found. Train the model first.")

    try:
        content = await image.read()
        img = Image.open(BytesIO(content)).convert("L")  # grayscale
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Resize to 8x8 and normalize to [0,1]
    img = img.resize(img_shape, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Image processing error.")
    v = (arr / 255.0).reshape(-1)

    # Project onto W to get H (activations)
    h = pg_nnls(W, v, max_iter=300, alpha0=1.0, beta=0.5, c=1e-4, tol=1e-6)
    pred = int(knn.predict([h])[0])
    seen_8x8 = (v.reshape(img_shape)).tolist()

    return JSONResponse({"pred": pred, "seen_8x8": seen_8x8})


@app.post("/predict_imagenet")
async def predict_imagenet(image: UploadFile = File(...)):
    """
    Classify a general image using a pretrained ImageNet MobileNetV2.
    Returns top-1 and top-5 labels with probabilities.
    """
    if _imagenet_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"ImageNet model unavailable. Install torch/torchvision and restart. Error: {_imagenet_error}"
        )

    try:
        content = await image.read()
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    x = _preprocess(img).unsqueeze(0)  # shape (1,3,224,224)

    import torch
    with torch.no_grad():
        logits = _imagenet_model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)

    top5 = [
        {"label": _categories[int(catid)] if _categories else int(catid), "prob": float(p)}
        for p, catid in zip(top5_prob.tolist(), top5_catid.tolist())
    ]

    return {"top1": top5[0], "top5": top5}
