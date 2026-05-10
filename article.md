---
author: "Kyle Jones"
date_published: "July 24, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/diagnosing-residual-phase-in-seismic-traces-using-python-and-the-hilbert-transform-87ed06783896"
---

# Diagnosing Residual Phase in Seismic Traces Using Python and the Hilbert Transform Seismic attributes offer a window into subsurface structure, when the
wavelet is under control. In many exploration datasets, interpreters...

### Diagnosing Residual Phase in Seismic Traces Using Python and the Hilbert Transform
Seismic attributes offer a window into subsurface structure, when the wavelet is under control. In many exploration datasets, interpreters rely on visual coherence, amplitude strength, or envelope consistency to track reflectors and make drilling decisions. But subtle errors in wavelet phase can distort structure, push peaks off target, and cloud our picture of the earth.

This project began with a simple question: Can we detect and correct residual phase in a seismic trace using only one line of data?

We wanted to validate whether a strong reflector had the expected zero-phase response. If not, we wanted a method to measure the deviation and correct for it.

### The Business Problem
Exploration teams use seismic data to estimate structure and reservoir properties. They interpret reflections from subsurface layers --- interfaces where acoustic impedance changes --- to map faults, pick horizons, and build models. If the data has been processed to zero phase, then strong reflections align with wavelet peaks or troughs, and envelope maxima line up.

But many datasets --- especially legacy surveys or public-domain volumes --- contain phase artifacts. This affects structural picks, time-depth conversion, and ultimately drilling decisions.

Residual phase isn't always obvious to the eye. It lurks in the mismatch between trace extrema and envelope peaks. If a team assumes zero phase when the wavelet still carries a rotation, the maps will be wrong. That's the risk we wanted to address.

### The Analytics Approach
We took a trace from the public Penobscot 3D dataset and used the Hilbert transform to compute the analytic signal. From that, we extracted the envelope and instantaneous phase.


The idea was to use the Hilbert transform to compute the envelope and phase. Then we could identify envelope peaks as proxies for strong reflectors and measure the phase at those peaks. Finally, we could estimate the average residual phase and apply a frequency-domain correction by shifting the phase spectrum. And as a bonus, we could verify the alignment between envelope peaks and corrected trace extrema.

We did all this in basic Python. No seismic workstation. No proprietary code. We kept it simple and reproducible (and admittedly much easier than doing this for a full segy file).

The core of the method lies in estimating phase error not from the whole spectrum, but from the behavior of the envelope. This aligns with practical guidance from Roden and Sepúlveda (1999): if a peak in the envelope does not match a peak in the signal, phase error is present.

### The Solution
We began by loading the trace file. We rewrote the original SEG tutorial from GNU Octave in Python and removed external dependencies. The `.trace` file format was tab-delimited text.

We computed the Hilbert transform using SciPy, then extracted the envelope and unwrapped phase. By finding peaks in the envelope and recording the phase at those points, we estimated the residual phase error. In our case, it came out to about 0.64 radians --- roughly 37 degrees.

To correct this, we shifted the spectrum in the frequency domain. We applied a phase shift of --0.64 radians to the positive frequencies and +0.64 to the negative ones. This left the amplitude spectrum unchanged but rotated the wavelet back toward zero phase.

After the correction, we re-computed the Hilbert attributes and repeated the peak analysis. The result: envelope peaks and trace extrema were aligned, and the phase at envelope maxima centered on zero.


Obviously, this isn't a replacement for full seismic processing. It's a fast, analytic way to spot and fix residual phase on a single trace or in small test volumes. It gives interpreters a better handle on what they're seeing and improves the accuracy of every pick.

### What You Can Do With This
This method works with any 1D seismic trace to do a diagnostic check on suspect processing, adjust synthetic seismograms to match recorded data, evaluate whether a dataset truly has zero phase, and explore wavelet behavior.

It also opens the door to automated residual phase correction in stacked sections or attribute volumes. If you can measure the phase error trace-by-trace, you can correct it trace-by-trace.

The technique builds on ideas published in [*The Leading Edge*](https://library.seg.org/doi/epub/10.1190/tle33101164.1) (October 2014) and developed by Steve Purves for the SEG tutorial series. We've adapted it here to fit modern Python workflows.

### Full Code
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
import matplotlib as mpl


def tufte_style():
    mpl.rcParams.update({
    'axes.grid': False,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    })


def load_trace(filepath):
    with open(filepath, "r") as f:
        line = f.readline().strip()
    parts = line.split('\t')
    trace = np.array([float(x) for x in parts[2:]])
    return trace


def compute_hilbert_attributes(trace, dt):
    analytic = hilbert(trace)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    time = np.arange(len(trace)) * dt
    return time, envelope, phase


def estimate_residual_phase(phase, envelope, time, threshold_pct=75):
    peaks, _ = find_peaks(envelope, distance=10, height=np.percentile(envelope, threshold_pct))
    peak_phases = phase[peaks]
    return np.mean(peak_phases), time[peaks], peak_phases


def apply_phase_shift(trace, phase_shift):
    N = len(trace)
    spectrum = np.fft.fft(trace)
    R = np.ones(N, dtype=complex)
    mid = (N + 1) // 2
    R[:mid] = np.exp(-1j * phase_shift)
    R[mid:] = np.exp(1j * phase_shift)
    shifted = spectrum * R
    return np.real(np.fft.ifft(shifted))


def plot_trace_and_envelope(time, trace, envelope, filename):
    tufte_style()
    plt.figure(figsize=(12, 4))
    plt.plot(time, trace, label="Amplitude", linewidth=1)
    plt.plot(time, envelope, label="Envelope", linewidth=1)
    plt.title("Seismic Trace and Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_phase_at_peaks(peak_times, peak_phases, label, filename):
    tufte_style()
    plt.figure(figsize=(10, 4))
    plt.plot(peak_times, peak_phases, 'o-', label=label)
    plt.axhline(0, linestyle='--', color='gray', label="Expected Zero Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.title(f"{label} at Envelope Peaks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_original_vs_corrected(time, original, corrected, phase_shift, filename):
    tufte_style()
    plt.figure(figsize=(12, 4))
    plt.plot(time, original, label="Original Trace", alpha=0.5, linewidth=1)
    plt.plot(time, corrected, label="Phase-Corrected Trace", alpha=0.9, linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Original vs Phase-Corrected Trace\nShift applied: {-phase_shift:.2f} radians")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    trace_file = "trace_il1190_xl1155.trace"
    dt = 0.004

    trace = load_trace(trace_file)

    time, envelope, phase = compute_hilbert_attributes(trace, dt)
    phase_shift, peak_times, peak_phases = estimate_residual_phase(phase, envelope, time)

    corrected_trace = apply_phase_shift(trace, phase_shift)
    time_corr, envelope_corr, phase_corr = compute_hilbert_attributes(corrected_trace, dt)
    _, peak_times_corr, peak_phases_corr = estimate_residual_phase(phase_corr, envelope_corr, time_corr)

    plot_trace_and_envelope(time, trace, envelope, "original_trace_envelope.png")
    plot_phase_at_peaks(peak_times, peak_phases, "Original Phase", "original_phase_peaks.png")
    plot_original_vs_corrected(time, trace, corrected_trace, phase_shift, "phase_corrected_trace.png")
    plot_phase_at_peaks(peak_times_corr, peak_phases_corr, "Corrected Phase", "corrected_phase_peaks.png")


if __name__ == "__main__":
    main()
```
