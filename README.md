# EEG Neural Decoding — Motor Imagery BCI

Decode imagined hand movements (left vs right) from 64-channel EEG signals
using deep learning, with an interactive Streamlit dashboard and GradCAM explainability.

---

## Project Overview

This project implements a full **Brain-Computer Interface (BCI) pipeline**:

1. **Data** — PhysioNet EEGBCI dataset, downloaded automatically via MNE-Python
2. **Preprocessing** — Bandpass filter (8–30 Hz), epoching, baseline correction
3. **Models** — EEGNet (compact CNN) and CNN-LSTM hybrid
4. **Evaluation** — Accuracy, Cohen's kappa, confusion matrices, per-subject breakdown
5. **Explainability** — GradCAM topographic maps showing which electrodes matter
6. **Dashboard** — Interactive Streamlit UI for exploring predictions

---

## Folder Structure

```
EegNeuralDecode/
├── data/               # Raw data cache + saved model checkpoints + plots
├── models/
│   ├── eegnet.py       # EEGNet (Lawhern et al., 2018)
│   └── cnn_lstm.py     # CNN-LSTM hybrid
├── training/
│   ├── train.py        # Cross-subject training loop
│   └── evaluate.py     # Metrics: accuracy, kappa, confusion matrix
├── explainability/
│   └── gradcam.py      # GradCAM + MNE topomap visualisation
├── app/
│   └── dashboard.py    # Streamlit dashboard
├── utils/
│   ├── preprocess.py   # MNE data pipeline
│   └── dataset.py      # PyTorch Dataset class
├── requirements.txt
└── README.md
```

---

## Running the Dashboard

```bash
cd EegNeuralDecode

# Start the FastAPI server
uvicorn server.api:app --reload --port 8000

# Then open http://localhost:8000 in your browser
```

The dashboard serves a terminal-style BCI interface with:
- Live epoch playback with EEG waveforms
- Real-time model predictions + confidence bars
- GradCAM scalp topomaps
- Model comparison benchmark table
- Animated neural brain canvas in the hero

> **Note:** Run `python training/train.py` first to generate model checkpoints and data files.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Train the models
```bash
cd EegNeuralDecode
python training/train.py
```
This will:
- Automatically download PhysioNet EEGBCI data (first run only, ~200 MB)
- Train on subjects S001–S007, test on S008–S010
- Save model checkpoints and training curves to `data/`

### Step 2 — Evaluate
```bash
python training/evaluate.py
```
Prints accuracy + Cohen's kappa, saves confusion matrices and per-subject plots.

### Step 3 — Launch the dashboard
```bash
streamlit run app/dashboard.py
```
Open http://localhost:8501 in your browser.

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset**
- 109 subjects, 64 EEG channels, 160 Hz sampling rate
- Tasks: imagined left-hand vs right-hand movement (runs 6, 10, 14)
- Automatically downloaded by MNE: `mne.datasets.eegbci.load_data()`
- This project uses subjects 1–10 (10 subjects × ~90 epochs = ~900 epochs)

---

## Model Architectures

### EEGNet (Lawhern et al., 2018)
A compact CNN designed specifically for EEG:
- **Block 1** — Temporal conv (learns frequency bands) + Depthwise spatial conv (learns which electrodes matter per band)
- **Block 2** — Separable conv (efficient feature combination)
- **Classifier** — Flatten → Dense(2)
- ~2,500 parameters; designed to generalise across subjects

### CNN-LSTM
A hybrid model combining spatial and temporal learning:
- **CNN layers** — Conv1d extracts local spatial patterns along the time axis
- **LSTM layers** — Captures long-range temporal dependencies across the epoch
- **Classifier** — Last LSTM hidden state → Dense(2)
- Good baseline; similar accuracy to EEGNet on this dataset

---

## GradCAM Explainability

GradCAM computes the gradient of the model's output with respect to the
last convolutional layer's activations. We aggregate these gradients
**per EEG channel** and visualise them as a scalp topographic map using MNE.

**What to look for:** Red regions (high importance) should cluster around
**C3** (left motor cortex → right-hand imagery) and
**C4** (right motor cortex → left-hand imagery),
consistent with the neuroscience literature.

---

## Screenshots

*Add screenshots after running the dashboard.*

| Dashboard Overview | GradCAM Topomap |
|---|---|
| `[screenshot1.png]` | `[screenshot2.png]` |

---

## Results

| Model    | Test Accuracy | Cohen's Kappa |
|----------|:-------------:|:-------------:|
| EEGNet   | TBD after training | TBD |
| CNN-LSTM | TBD after training | TBD |

*(Run `python training/evaluate.py` to fill in this table)*

---

## References

- Lawhern et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces*. Journal of Neural Engineering.
- Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals*. Circulation.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.

---

## Author

Built as a portfolio project for an ECE student learning BCI + deep learning.
Stack: PyTorch · MNE-Python · Streamlit · scikit-learn
