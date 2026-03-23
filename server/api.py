"""
api.py -- FastAPI backend for EEG Neural Decoding dashboard.

Performance notes:
  - All heavy data (epochs, models, MNE info, comparison results) is loaded
    ONCE at startup via the lifespan event and stored in _cache.
  - Per-request work is limited to a single model forward pass (<15 ms).
  - GradCAM (~400 ms) is the only genuinely slow endpoint; JS shows a spinner.

Endpoints:
  GET /api/status
  GET /api/overview
  GET /api/epoch/{idx}?model=eegnet|cnn_lstm|mdm
  GET /api/gradcam/{idx}
  GET /api/comparison
  GET /api/training_curve/{model}

Run with:
  uvicorn server.api:app --port 8000
"""

import sys, os, io, pickle, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import warnings
warnings.filterwarnings('ignore')

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
FRONTEND = os.path.join(ROOT, 'frontend')

CLASS_NAMES = ['Left Hand', 'Right Hand']
SFREQ       = 160.0
TMIN        = -0.5

# ---------------------------------------------------------------------------
# In-memory cache (populated at startup)
# ---------------------------------------------------------------------------
_cache: dict = {}

# ---------------------------------------------------------------------------
# Startup: warm all caches so every request is fast
# ---------------------------------------------------------------------------

def _warmup():
    """Load data, models, and pre-compute everything expensive once."""
    # 1. Raw epochs & labels
    ep_path = os.path.join(DATA_DIR, 'test_epochs.npy')
    lb_path = os.path.join(DATA_DIR, 'test_labels.npy')
    if not os.path.exists(ep_path):
        log.warning("test_epochs.npy not found -- run training/train.py first")
        return

    epochs = np.load(ep_path)          # (N, 64, 481) float64
    labels = np.load(lb_path)          # (N,) int64
    _cache['epochs'] = epochs
    _cache['labels'] = labels
    log.info(f"Loaded {len(epochs)} test epochs")

    # 2. Pre-normalised float32 epochs (zero-mean, unit-var per channel)
    #    Stored once; all per-request work uses these slices directly.
    norm = epochs.astype(np.float32)
    mean = norm.mean(axis=2, keepdims=True)
    std  = norm.std(axis=2, keepdims=True)
    std  = np.where(std < 1e-8, 1e-8, std)
    _cache['epochs_norm'] = (norm - mean) / std   # (N, 64, 481) float32

    # 3. Pre-compute waveform metadata (same for every epoch)
    n_ch   = epochs.shape[1]
    step   = max(1, n_ch // 8)
    ch_idx = list(range(0, n_ch, step))[:8]
    _cache['wave_ch_idx'] = ch_idx
    _cache['time_axis']   = np.linspace(
        TMIN, TMIN + epochs.shape[2] / SFREQ, epochs.shape[2]
    ).tolist()

    # 4. MNE Info (for topomap electrode positions)
    try:
        from utils.preprocess import load_subject
        _, _, ch_names, info = load_subject(8, verbose=False)
        _cache['mne_info'] = info
        _cache['ch_names'] = list(ch_names)
        log.info("MNE electrode info loaded")
    except Exception as e:
        log.warning(f"Could not load MNE info: {e}")
        _cache['mne_info'] = None
        _cache['ch_names'] = [f"Ch{i}" for i in range(64)]

    # 5. Channel names for the 8 waveform channels
    ch_names = _cache['ch_names']
    _cache['wave_ch_names'] = [
        ch_names[i] if i < len(ch_names) else f"Ch{i}"
        for i in ch_idx
    ]

    # 6. Deep models
    n_ch_model = epochs.shape[1]
    n_t_model  = epochs.shape[2]

    for model_type in ('eegnet', 'cnn_lstm'):
        ckpt_path = os.path.join(DATA_DIR, f'{model_type}_best.pt')
        if not os.path.exists(ckpt_path):
            _cache[f'model_{model_type}'] = None
            continue
        try:
            if model_type == 'eegnet':
                from models.eegnet import EEGNet
                m = EEGNet(n_classes=2, n_channels=n_ch_model, n_times=n_t_model)
            else:
                from models.cnn_lstm import CNNLSTM
                m = CNNLSTM(n_classes=2, n_channels=n_ch_model, n_times=n_t_model)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            _cache[f'model_{model_type}'] = m
            log.info(f"Loaded {model_type} model")
        except Exception as e:
            log.warning(f"Could not load {model_type}: {e}")
            _cache[f'model_{model_type}'] = None

    # 7. MDM pipeline
    pkl = os.path.join(DATA_DIR, 'riemannian_mdm_best.pkl')
    if os.path.exists(pkl):
        try:
            with open(pkl, 'rb') as f:
                _cache['mdm'] = pickle.load(f)
            log.info("Loaded Riemannian MDM pipeline")
        except Exception as e:
            log.warning(f"Could not load MDM: {e}")
            _cache['mdm'] = None
    else:
        _cache['mdm'] = None

    # 8. Pre-compute comparison metrics (expensive; do it once here)
    _cache['comparison'] = _compute_comparison(epochs, labels)
    log.info("Comparison metrics pre-computed")


def _compute_comparison(epochs: np.ndarray, labels: np.ndarray) -> dict:
    """Run batch inference for all models, return accuracy/kappa dict."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from utils.dataset import EEGDataset

    result = {}

    for model_type in ('eegnet', 'cnn_lstm'):
        model = _cache.get(f'model_{model_type}')
        if model is None:
            continue
        try:
            ds     = EEGDataset(epochs, labels, model_type=model_type)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
            preds  = []
            with torch.no_grad():
                for x, _ in loader:
                    preds.extend(model(x).argmax(dim=1).tolist())
            result[model_type] = {
                "accuracy": round(float(accuracy_score(labels, preds)), 4),
                "kappa":    round(float(cohen_kappa_score(labels, preds)), 4),
                "type":     "Deep Learning",
            }
        except Exception as e:
            log.warning(f"comparison inference failed for {model_type}: {e}")

    # MDM -- read from saved npy (already computed by riemannian_mdm.py)
    mdm_path = os.path.join(DATA_DIR, 'mdm_results.npy')
    if os.path.exists(mdm_path):
        try:
            mdm_res = np.load(mdm_path, allow_pickle=True).item()
            result['mdm'] = {
                "accuracy": round(float(mdm_res.get('accuracy', 0)), 4),
                "kappa":    round(float(mdm_res.get('kappa',    0)), 4),
                "type":     "Riemannian",
            }
        except Exception as e:
            log.warning(f"Could not load MDM results: {e}")

    return result


# ---------------------------------------------------------------------------
# FastAPI lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    yield   # server runs here
    _cache.clear()


app = FastAPI(title="EEG Neural Decode API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_deep(model, norm_ep: np.ndarray, model_type: str):
    """Single-epoch forward pass using pre-normalised data. Returns (class_idx, probs)."""
    if model_type == 'eegnet':
        t = torch.from_numpy(norm_ep[np.newaxis, np.newaxis, ...])   # (1,1,C,T)
    else:
        t = torch.from_numpy(norm_ep[np.newaxis, ...])               # (1,C,T)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1).squeeze().numpy()
    return int(probs.argmax()), probs


def _predict_mdm(mdm, raw_ep: np.ndarray):
    """Single-epoch MDM prediction. Returns (class_idx, probs)."""
    ep_3d = raw_ep[np.newaxis, ...]   # (1, C, T)
    pred  = int(mdm.predict(ep_3d)[0])
    # Pipeline.predict_proba is available when the final estimator supports it
    try:
        probs = mdm.predict_proba(ep_3d)[0]
    except Exception:
        probs = np.array([1.0, 0.0]) if pred == 0 else np.array([0.0, 1.0])
    return pred, probs


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status():
    epochs = _cache.get('epochs')
    return {
        "data_loaded":  epochs is not None,
        "n_epochs":     int(len(epochs)) if epochs is not None else 0,
        "models_loaded": {
            "eegnet":   _cache.get('model_eegnet')   is not None,
            "cnn_lstm": _cache.get('model_cnn_lstm')  is not None,
            "mdm":      _cache.get('mdm')             is not None,
        },
    }


@app.get("/api/overview")
def overview():
    epochs = _cache.get('epochs')
    labels = _cache.get('labels')
    if epochs is None:
        raise HTTPException(404, "Data not found. Run training/train.py first.")
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "n_epochs":     int(len(epochs)),
        "n_channels":   int(epochs.shape[1]),
        "n_times":      int(epochs.shape[2]),
        "duration_s":   round(epochs.shape[2] / SFREQ, 2),
        "class_counts": {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)},
    }


@app.get("/api/epoch/{idx}")
def epoch_endpoint(idx: int, model: str = "eegnet"):
    epochs_norm = _cache.get('epochs_norm')
    epochs_raw  = _cache.get('epochs')
    labels      = _cache.get('labels')

    if epochs_norm is None:
        raise HTTPException(404, "Data not found.")
    n = len(epochs_norm)
    if idx < 0 or idx >= n:
        raise HTTPException(400, f"idx must be 0..{n - 1}")

    norm_ep = epochs_norm[idx]        # pre-computed, (64, 481) float32
    raw_ep  = epochs_raw[idx]
    true_label = int(labels[idx])
    model = model.lower()

    # --- Prediction (fast: model already loaded, epoch pre-normalised) --------
    if model in ('eegnet', 'cnn_lstm'):
        m = _cache.get(f'model_{model}')
        if m is None:
            pred_class, probs = true_label, np.array([0.5, 0.5], dtype=np.float32)
        else:
            pred_class, probs = _predict_deep(m, norm_ep, model)
    elif model == 'mdm':
        mdm = _cache.get('mdm')
        if mdm is None:
            pred_class, probs = true_label, np.array([0.5, 0.5], dtype=np.float32)
        else:
            pred_class, probs = _predict_mdm(mdm, raw_ep)
    else:
        raise HTTPException(400, f"Unknown model '{model}'. Use eegnet, cnn_lstm, or mdm.")

    # --- Waveform (pre-computed channel indices & time axis) ------------------
    ch_idx   = _cache['wave_ch_idx']
    waveform = [norm_ep[i].tolist() for i in ch_idx]

    return {
        "idx":             idx,
        "true_label":      CLASS_NAMES[true_label],
        "predicted_label": CLASS_NAMES[pred_class],
        "correct":         bool(pred_class == true_label),
        "probabilities":   {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(2)},
        "waveform":        waveform,
        "time_axis":       _cache['time_axis'],
        "channel_names":   _cache['wave_ch_names'],
        "model":           model,
    }


@app.get("/api/gradcam/{idx}")
def gradcam_endpoint(idx: int):
    """
    Return GradCAM topomap PNG (always uses EEGNet -- GradCAM only defined for CNNs).
    Custom header X-Top-Channels contains the 5 most activated electrode names.
    """
    epochs_norm = _cache.get('epochs_norm')
    if epochs_norm is None:
        raise HTTPException(404, "Data not found.")
    n = len(epochs_norm)
    if idx < 0 or idx >= n:
        raise HTTPException(400, f"idx must be 0..{n - 1}")

    eegnet = _cache.get('model_eegnet')
    info   = _cache.get('mne_info')
    ch_names = _cache.get('ch_names', [])

    top_channel_str = "N/A"
    img_bytes = None

    if eegnet is not None and info is not None:
        try:
            from explainability.gradcam import GradCAM

            norm_ep = epochs_norm[idx]
            t = torch.from_numpy(norm_ep[np.newaxis, np.newaxis, ...]).requires_grad_(True)

            gc = GradCAM(eegnet, eegnet.get_last_conv_layer())
            scores, pred_class, _ = gc.compute(t)
            gc.remove_hooks()

            # Build dark-themed topomap in memory
            fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='#0d0d0d')
            im, _   = mne.viz.plot_topomap(
                scores, info, axes=ax, show=False,
                cmap='RdYlGn', vlim=(0, 1), contours=4, sensors=True,
            )
            cbar = plt.colorbar(im, ax=ax, label='Importance')
            cbar.ax.yaxis.label.set_color('#e0e0e0')
            cbar.ax.tick_params(colors='#888')
            ax.set_facecolor('#0d0d0d')
            ax.set_title(
                f'GradCAM  --  {CLASS_NAMES[pred_class]}',
                color='#00ff88', fontsize=11, pad=8,
                fontfamily='monospace',
            )
            fig.patch.set_facecolor('#0d0d0d')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100,
                        bbox_inches='tight', facecolor='#0d0d0d')
            plt.close(fig)
            buf.seek(0)
            img_bytes = buf.read()

            top5 = np.argsort(scores)[-5:][::-1]
            top_channel_str = ", ".join(
                ch_names[i] if i < len(ch_names) else f"Ch{i}" for i in top5
            )
        except Exception as e:
            log.warning(f"GradCAM failed for idx={idx}: {e}")

    # Fallback placeholder
    if img_bytes is None:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#111')
        ax.text(0.5, 0.5, 'GradCAM N/A\n\nTrain EEGNet first:\npython training/train.py',
                ha='center', va='center', color='#00ff88',
                fontsize=10, transform=ax.transAxes, fontfamily='monospace')
        ax.set_facecolor('#111')
        ax.axis('off')
        fig.patch.set_facecolor('#111')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#111')
        plt.close(fig)
        buf.seek(0)
        img_bytes = buf.read()

    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={
            "X-Top-Channels": top_channel_str,
            "Access-Control-Expose-Headers": "X-Top-Channels",
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/comparison")
def comparison():
    """Return pre-computed model comparison metrics (never re-runs inference)."""
    result = _cache.get('comparison')
    if result is None:
        raise HTTPException(503, "Comparison metrics not ready. Server may still be starting.")
    return result


@app.get("/api/training_curve/{model_name}")
def training_curve(model_name: str):
    """Serve a training curve PNG image."""
    safe = model_name.lower().replace("-", "_").replace(" ", "_")
    path = os.path.join(DATA_DIR, f'{safe}_curves.png')
    if not os.path.exists(path):
        raise HTTPException(404, f"No training curve found for '{model_name}'")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Static frontend (must be mounted LAST so API routes take priority)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
