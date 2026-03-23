"""
preprocess.py — MNE-based EEG data pipeline for PhysioNet EEGBCI dataset.

Downloads raw EEG, applies bandpass filter (8-30 Hz), epochs the data,
applies baseline correction, and returns cleaned epochs + labels per subject.
"""

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import warnings

warnings.filterwarnings('ignore')

# -- Constants ----------------------------------------------------------------
# Motor imagery runs: 6=left/right hands, 10=left/right hands, 14=left/right hands
MOTOR_IMAGERY_RUNS = [6, 10, 14]

# Event IDs from PhysioNet annotations
# T1 = left-hand motor imagery, T2 = right-hand motor imagery
EVENT_ID = {'T1': 1, 'T2': 2}

TMIN = -0.5    # epoch start relative to event (seconds)
TMAX = 2.5     # epoch end relative to event (seconds)
L_FREQ = 8.0   # bandpass lower cutoff (mu band starts at 8 Hz)
H_FREQ = 30.0  # bandpass upper cutoff (beta band ends at 30 Hz)
SFREQ = 160    # target sampling frequency (PhysioNet EEGBCI is 160 Hz)


def load_subject(subject_id: int, verbose: bool = False):
    """
    Download and preprocess EEG data for a single subject.

    Parameters
    ----------
    subject_id : int
        Subject number (1–10 for this project, up to 109 in full dataset)
    verbose : bool
        If True, show MNE processing messages

    Returns
    -------
    epochs_array : np.ndarray, shape (n_epochs, n_channels, n_times)
    labels       : np.ndarray, shape (n_epochs,)  — 0=left hand, 1=right hand
    ch_names     : list of str  — channel names (64 EEG channels)
    info         : mne.Info     — MNE info object for topomap plotting
    """
    # Download the raw EDF files (cached after first run)
    raw_fnames = eegbci.load_data(subject_id, MOTOR_IMAGERY_RUNS, verbose=verbose)

    # Load and concatenate the three runs into one Raw object
    raws = [read_raw_edf(f, preload=True, verbose=verbose) for f in raw_fnames]
    raw = concatenate_raws(raws)

    # Standardize electrode names to match MNE's standard 10-20 montage
    eegbci.standardize(raw)

    # Attach the standard 10-20 electrode positions (needed for topomap later)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, verbose=verbose)

    # Drop non-EEG channels if any exist
    raw.pick_types(eeg=True, verbose=verbose)

    # Bandpass filter: keep mu (8-12 Hz) and beta (12-30 Hz) bands
    # These frequency bands are most informative for motor imagery
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=verbose)

    # Extract events from the EDF annotations
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=verbose)

    # Epoch the data around each event
    # baseline=(-0.5, 0) applies baseline correction using pre-stimulus period
    epochs = mne.Epochs(
        raw, events, EVENT_ID,
        tmin=TMIN, tmax=TMAX,
        baseline=(TMIN, 0),
        preload=True,
        verbose=verbose
    )

    # Get epochs as numpy array: (n_epochs, n_channels, n_times)
    epochs_array = epochs.get_data()

    # Convert MNE event codes back to 0-indexed labels
    # T1 (code 1) -> 0 (left hand), T2 (code 2) -> 1 (right hand)
    labels = epochs.events[:, 2] - 1  # subtract 1 to get 0/1

    ch_names = epochs.ch_names
    info = epochs.info

    return epochs_array, labels, ch_names, info


def load_multiple_subjects(subject_ids: list, verbose: bool = False):
    """
    Load and concatenate EEG data from multiple subjects.

    Parameters
    ----------
    subject_ids : list of int
    verbose : bool

    Returns
    -------
    all_epochs : np.ndarray, shape (total_epochs, n_channels, n_times)
    all_labels : np.ndarray, shape (total_epochs,)
    ch_names   : list of str
    info       : mne.Info  — from last subject loaded
    """
    all_epochs = []
    all_labels = []
    ch_names = None
    info = None

    for sid in subject_ids:
        try:
            print(f"  Loading subject S{sid:03d}...")
            epochs, labels, ch_names, info = load_subject(sid, verbose=verbose)
            all_epochs.append(epochs)
            all_labels.append(labels)
        except Exception as e:
            print(f"  [WARNING] Failed to load subject S{sid:03d}: {e}")

    if len(all_epochs) == 0:
        raise RuntimeError("No subjects loaded successfully.")

    all_epochs = np.concatenate(all_epochs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"  Loaded {len(all_epochs)} epochs from {len(subject_ids)} subjects.")
    return all_epochs, all_labels, ch_names, info


if __name__ == "__main__":
    # Quick test: load subject 1 and print shapes
    print("Testing preprocessing pipeline with Subject 1...")
    epochs, labels, ch_names, info = load_subject(1, verbose=False)
    print(f"  Epochs shape : {epochs.shape}")
    print(f"  Labels shape : {labels.shape}")
    print(f"  Label values : {np.unique(labels, return_counts=True)}")
    print(f"  Channels     : {len(ch_names)} channels")
    print(f"  Time points  : {epochs.shape[2]} samples @ {SFREQ} Hz")
    print("Preprocessing test PASSED.")
