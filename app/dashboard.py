"""
dashboard.py — Streamlit EEG Neural Decoding Dashboard.

Panels:
  1. Header — project title + description
  2. Sidebar — subject/model selector + run controls
  3. Data Overview — epoch counts, class distribution
  4. Live Epoch Playback — scrub through epochs, see waveform + prediction
  5. GradCAM Topomap — scalp heatmap of electrode importance
  6. Model Comparison — accuracy, kappa, confusion matrices, training curves

Run with:
  cd EegNeuralDecode
  streamlit run app/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# -- Path constants -------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# -- Page config ----------------------------------------------------------------
st.set_page_config(
    page_title="EEG Neural Decoding",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ['Left Hand', 'Right Hand']
SFREQ       = 160   # Hz
TMIN        = -0.5  # epoch start (seconds)


# ==============================================================================
# Cached helpers — only run once per session
# ==============================================================================

@st.cache_resource(show_spinner="Loading EEG data...")
def load_data():
    """Load pre-processed test epochs from disk (saved by train.py)."""
    required = ['test_epochs.npy', 'test_labels.npy']
    for f in required:
        if not os.path.exists(os.path.join(DATA_DIR, f)):
            return None, None, None, None

    epochs  = np.load(os.path.join(DATA_DIR, 'test_epochs.npy'))
    labels  = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
    ch_names_path = os.path.join(DATA_DIR, 'ch_names.npy')
    ch_names = list(np.load(ch_names_path)) if os.path.exists(ch_names_path) else None
    return epochs, labels, ch_names, None  # info loaded separately below


@st.cache_resource(show_spinner="Loading MNE electrode info...")
def load_mne_info(subject_id: int):
    """Load MNE Info object for topomap electrode positions."""
    try:
        from utils.preprocess import load_subject
        _, _, ch_names, info = load_subject(subject_id, verbose=False)
        return info
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(model_type: str, n_channels: int, n_times: int):
    """Load a trained model from its checkpoint."""
    ckpt_path = os.path.join(DATA_DIR, f'{model_type}_best.pt')
    if not os.path.exists(ckpt_path):
        return None

    try:
        if model_type == 'eegnet':
            from models.eegnet import EEGNet
            model = EEGNet(n_classes=2, n_channels=n_channels, n_times=n_times)
        else:
            from models.cnn_lstm import CNNLSTM
            model = CNNLSTM(n_classes=2, n_channels=n_channels, n_times=n_times)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load {model_type} model: {e}")
        return None


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per channel."""
    mean = epoch.mean(axis=1, keepdims=True)
    std  = epoch.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-8, 1e-8, std)
    return (epoch - mean) / std


def predict_epoch(model, epoch: np.ndarray, model_type: str):
    """
    Run inference on a single epoch, return (predicted_class, probabilities).

    Parameters
    ----------
    model      : loaded PyTorch model
    epoch      : np.ndarray (n_channels, n_times)
    model_type : 'eegnet' or 'cnn_lstm'

    Returns
    -------
    pred_class : int
    probs      : np.ndarray, shape (2,)
    """
    norm = normalize_epoch(epoch)

    if model_type == 'eegnet':
        tensor = torch.tensor(norm[np.newaxis, np.newaxis, ...], dtype=torch.float32)
    else:
        tensor = torch.tensor(norm[np.newaxis, ...], dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()

    return probs.argmax(), probs


# ==============================================================================
# Panel renderers
# ==============================================================================

def render_header():
    """Panel 1 — Project title and description."""
    st.title("🧠 EEG Neural Decoding — Motor Imagery BCI")
    st.markdown("""
    This dashboard decodes **motor imagery** (imagined left-hand vs right-hand movement)
    from EEG signals using deep learning.

    **Dataset:** PhysioNet EEGBCI (64-channel, 160 Hz, motor imagery runs 6/10/14)
    **Models:** EEGNet (Lawhern et al., 2018) · CNN-LSTM hybrid
    **Explainability:** GradCAM scalp topographic maps
    """)
    st.divider()


def render_sidebar():
    """Panel 2 — Sidebar controls. Returns selected subject, model, and run flag."""
    with st.sidebar:
        st.header("Controls")

        subject = st.selectbox(
            "Subject",
            options=[f"S{i:03d}" for i in range(1, 11)],
            index=7,  # default S008 (test set)
            help="Select a subject from the dataset."
        )
        subject_id = int(subject[1:])  # "S008" -> 8

        model_name = st.selectbox(
            "Model",
            options=["EEGNet", "CNN-LSTM"],
            help="Select which trained model to use for prediction."
        )
        model_type = 'eegnet' if model_name == 'EEGNet' else 'cnn_lstm'

        st.divider()
        st.markdown("### About")
        st.markdown(
            "Built with **PyTorch**, **MNE-Python**, and **Streamlit**.  \n"
            "Training: S001–S007 | Testing: S008–S010"
        )

    return subject_id, model_type, model_name


def render_data_overview(epochs: np.ndarray, labels: np.ndarray):
    """Panel 3 — Show epoch statistics and class distribution."""
    st.subheader("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Epochs",   len(epochs))
    col2.metric("EEG Channels",   epochs.shape[1])
    col3.metric("Time Points",    epochs.shape[2])
    col4.metric("Duration (s)",   f"{epochs.shape[2] / SFREQ:.1f}")

    # Class distribution bar chart
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.set_style('whitegrid')
    bars = ax.bar([CLASS_NAMES[i] for i in unique], counts, color=['steelblue', 'coral'])
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(cnt), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Number of epochs')
    ax.set_title('Class Distribution (Test Set)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_epoch_playback(epochs: np.ndarray, labels: np.ndarray,
                          model, model_type: str):
    """Panel 4 — Epoch scrubber with waveform display and model prediction."""
    st.subheader("▶️ Live Epoch Playback")

    if model is None:
        st.info("Train the models first (`python training/train.py`), then reload the dashboard.")
        _render_raw_waveform_only(epochs, labels)
        return

    idx = st.slider("Epoch index", min_value=0, max_value=len(epochs) - 1, value=0, step=1)
    epoch  = epochs[idx]    # (n_channels, n_times)
    true_label = labels[idx]

    # -- Model prediction ------------------------------------------------------
    pred_class, probs = predict_epoch(model, epoch, model_type)

    col_wave, col_pred = st.columns([2, 1])

    with col_wave:
        # Plot a subset of channels for readability (every 4th channel)
        time_axis = np.linspace(TMIN, TMIN + epoch.shape[1] / SFREQ, epoch.shape[1])
        channel_step = max(1, epoch.shape[0] // 16)  # show ~16 channels
        selected = epoch[::channel_step]

        fig, ax = plt.subplots(figsize=(8, 4))
        offset = 0
        for i, ch_data in enumerate(selected):
            norm = (ch_data - ch_data.mean()) / (ch_data.std() + 1e-8)
            ax.plot(time_axis, norm + offset, linewidth=0.7, color='steelblue', alpha=0.8)
            offset += 3
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Event onset')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels (offset)')
        ax.set_title(f'EEG Epoch #{idx}  |  True label: {CLASS_NAMES[true_label]}')
        ax.legend(loc='upper right')
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_pred:
        st.markdown(f"**True class:** {CLASS_NAMES[true_label]}")
        st.markdown(f"**Predicted :** :{'green' if pred_class == true_label else 'red'}[{CLASS_NAMES[pred_class]}]")

        # Confidence bar
        st.markdown("**Confidence:**")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            st.progress(float(probs[cls_idx]), text=f"{cls_name}: {probs[cls_idx]:.2%}")


def _render_raw_waveform_only(epochs: np.ndarray, labels: np.ndarray):
    """Fallback waveform display when model is not yet trained."""
    idx = st.slider("Epoch index", min_value=0, max_value=len(epochs) - 1, value=0)
    epoch = epochs[idx]
    time_axis = np.linspace(TMIN, TMIN + epoch.shape[1] / SFREQ, epoch.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, ch_data in enumerate(epoch[::4]):
        ax.plot(time_axis, ch_data + i * 5e-6, linewidth=0.6, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f'Epoch #{idx} — True: {CLASS_NAMES[labels[idx]]}')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_gradcam(epochs: np.ndarray, labels: np.ndarray,
                   model, model_type: str, info):
    """Panel 5 — GradCAM topomap for selected epoch."""
    st.subheader("🌡️ GradCAM — Electrode Importance Map")

    if model is None or model_type != 'eegnet':
        if model_type != 'eegnet':
            st.info("GradCAM topomap is available only for **EEGNet** (select it in the sidebar).")
        else:
            st.info("Train EEGNet first, then the topomap will appear here.")
        return

    if info is None:
        st.warning("MNE electrode positions unavailable. Cannot render topomap.")
        return

    idx = st.session_state.get('epoch_idx', 0)

    try:
        from explainability.gradcam import GradCAM, plot_topomap
        from models.eegnet import EEGNet

        epoch = epochs[idx]
        norm  = normalize_epoch(epoch)
        tensor = torch.tensor(norm[np.newaxis, np.newaxis, ...], dtype=torch.float32)
        tensor.requires_grad_(True)

        gradcam_obj = GradCAM(model, model.get_last_conv_layer())
        scores, pred_class, probs = gradcam_obj.compute(tensor)
        gradcam_obj.remove_hooks()

        # Save topomap to temp file and display
        topomap_path = os.path.join(DATA_DIR, f'gradcam_epoch_{idx}.png')
        plot_topomap(scores, info, pred_class, save_path=topomap_path)

        col_map, col_info = st.columns([1, 1])
        with col_map:
            if os.path.exists(topomap_path):
                img = Image.open(topomap_path)
                st.image(img, caption=f"GradCAM Topomap — Epoch #{idx}", use_container_width=True)

        with col_info:
            # Highlight top 5 most important channels
            top5_idx = np.argsort(scores)[-5:][::-1]
            st.markdown("**Most activated electrodes:**")
            for rank, ch_i in enumerate(top5_idx, 1):
                st.markdown(f"  {rank}. Channel index **{ch_i}** — score `{scores[ch_i]:.3f}`")

            st.markdown(
                """
                **How to read this map:**
                🔴 Red regions = electrodes that most strongly influenced the prediction.
                🔵 Blue regions = suppressed or uninformative for this class.
                Motor imagery typically activates **C3/C4** (central motor cortex).
                """
            )

    except Exception as e:
        st.error(f"GradCAM failed: {e}")


def render_model_comparison():
    """Panel 6 — Side-by-side model accuracy, kappa, confusion matrices, training curves."""
    st.subheader("📈 Model Comparison")

    # -- Training curves --------------------------------------------------------
    eegnet_hist_path  = os.path.join(DATA_DIR, 'eegnet_history.npy')
    cnn_hist_path     = os.path.join(DATA_DIR, 'cnn_lstm_history.npy')
    eegnet_curve_path = os.path.join(DATA_DIR, 'eegnet_curves.png')
    cnn_curve_path    = os.path.join(DATA_DIR, 'cnn_lstm_curves.png')

    curves_exist = os.path.exists(eegnet_curve_path) and os.path.exists(cnn_curve_path)

    if curves_exist:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**EEGNet Training Curves**")
            st.image(Image.open(eegnet_curve_path), use_container_width=True)
        with col2:
            st.markdown("**CNN-LSTM Training Curves**")
            st.image(Image.open(cnn_curve_path), use_container_width=True)
    else:
        st.info("Training curves not found. Run `python training/train.py` first.")

    # -- Confusion matrices -----------------------------------------------------
    eegnet_cm_path = os.path.join(DATA_DIR, 'confusion_eegnet.png')
    cnn_cm_path    = os.path.join(DATA_DIR, 'confusion_cnn_lstm.png')
    cms_exist      = os.path.exists(eegnet_cm_path) and os.path.exists(cnn_cm_path)

    if cms_exist:
        st.markdown("#### Confusion Matrices")
        col3, col4 = st.columns(2)
        with col3:
            st.image(Image.open(eegnet_cm_path), caption="EEGNet", use_container_width=True)
        with col4:
            st.image(Image.open(cnn_cm_path), caption="CNN-LSTM", use_container_width=True)
    else:
        st.info("Confusion matrices not found. Run `python training/evaluate.py` first.")

    # -- Per-subject accuracy ---------------------------------------------------
    per_sub_path = os.path.join(DATA_DIR, 'per_subject_accuracy.png')
    if os.path.exists(per_sub_path):
        st.markdown("#### Per-Subject Accuracy")
        st.image(Image.open(per_sub_path), use_container_width=False, width=600)

    # -- Numeric summary table --------------------------------------------------
    # Load final val accuracy from history files if available
    if os.path.exists(eegnet_hist_path) and os.path.exists(cnn_hist_path):
        eegnet_hist = np.load(eegnet_hist_path, allow_pickle=True).item()
        cnn_hist    = np.load(cnn_hist_path,    allow_pickle=True).item()

        eegnet_best = max(eegnet_hist.get('val_acc', [0]))
        cnn_best    = max(cnn_hist.get('val_acc', [0]))

        st.markdown("#### Best Validation Accuracy")
        col5, col6 = st.columns(2)
        col5.metric("EEGNet",   f"{eegnet_best:.2%}")
        col6.metric("CNN-LSTM", f"{cnn_best:.2%}")


# ==============================================================================
# Main app entry point
# ==============================================================================

def main():
    render_header()
    subject_id, model_type, model_name = render_sidebar()

    # -- Load data --------------------------------------------------------------
    epochs, labels, ch_names, _ = load_data()

    if epochs is None:
        st.warning(
            "[!]️ No preprocessed data found in `data/`.  \n"
            "Run the preprocessing + training pipeline first:  \n"
            "```bash\n"
            "python training/train.py\n"
            "python training/evaluate.py\n"
            "```"
        )
        # Show a demo with synthetic data so the UI is still functional
        st.markdown("---")
        st.info("Showing **synthetic demo data** (64 channels, 481 time points).")
        np.random.seed(42)
        epochs = np.random.randn(90, 64, 481).astype(np.float32)
        labels = np.random.randint(0, 2, 90).astype(np.int64)
        model  = None
    else:
        n_channels = epochs.shape[1]
        n_times    = epochs.shape[2]
        model = load_model(model_type, n_channels, n_times)
        if model is None:
            st.warning(
                f"[!]️ No trained checkpoint found for **{model_name}**.  \n"
                "Run `python training/train.py` to train both models."
            )

    # -- Load MNE info for topomap ----------------------------------------------
    info = load_mne_info(min(subject_id, 10))

    # -- Render panels ----------------------------------------------------------
    render_data_overview(epochs, labels)
    st.divider()

    render_epoch_playback(epochs, labels, model, model_type)
    st.divider()

    if model_type == 'eegnet':
        render_gradcam(epochs, labels, model, model_type, info)
        st.divider()

    render_model_comparison()


if __name__ == "__main__":
    main()
