"""
AFiNET — Atrial Fibrillation Detection System
Full application: preprocessing pipeline + DDNN inference + Grad-CAM + GUI
Light mode default. Dark mode toggleable. Horizontal zoom with +/- and scrollbar.
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import queue
import threading
import warnings
import traceback
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.font import Font

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# DEEP LEARNING — optional, graceful fallback
# ─────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# PREPROCESSING CONSTANTS
# ─────────────────────────────────────────────────────────────────
TARGET_FS      = 125
WINDOW_SAMPLES = 30 * TARGET_FS   # 3 750
AF_THRESHOLD   = 0.50
SAFE_BUFFER    = 6_000
PAD_SEC        = 5
PAD_SAMPLES    = PAD_SEC * TARGET_FS

_LOW  = 0.5  / (0.5 * TARGET_FS)
_HIGH = 40.0 / (0.5 * TARGET_FS)
BUTTER_B, BUTTER_A = scipy_signal.butter(6, [_LOW, _HIGH], btype="band")

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTES
# ─────────────────────────────────────────────────────────────────
LIGHT = {
    "bg":             "#F4F6FB",
    "surface":        "#FFFFFF",
    "surface2":       "#EDF0F7",
    "border":         "#CDD3E0",
    "accent":         "#2563EB",
    "accent_dim":     "#DBEAFE",
    "af_red":         "#C0392B",
    "af_red_dim":     "#FDECEA",
    "af_red_border":  "#E57373",
    "ok_green":       "#1A7A4A",
    "ok_green_dim":   "#E8F5ED",
    "ok_green_border":"#66BB8A",
    "text":           "#1A1F36",
    "text_dim":       "#4A5568",
    "text_faint":     "#9AAABF",
    "warn":           "#B45309",
    "section_lbl":    "#6B7A99",
    "plot_bg":        "#FFFFFF",
    "plot_grid":      "#E2E8F0",
    "plot_spine":     "#CBD5E0",
    "ecg_line":       "#000000",
    "cam_fill":       "#E05560",
    "cam_line":       "#9B1C24",
    "cam_zero":       "#FFFFFF",
}

DARK = {
    "bg":             "#0F1117",
    "surface":        "#1A1D27",
    "surface2":       "#22263A",
    "border":         "#2E3350",
    "accent":         "#4F8EF7",
    "accent_dim":     "#2A4A8A",
    "af_red":         "#E05560",
    "af_red_dim":     "#6B2229",
    "af_red_border":  "#E05560",
    "ok_green":       "#3DBE7A",
    "ok_green_dim":   "#1A5234",
    "ok_green_border":"#3DBE7A",
    "text":           "#E8EBF5",
    "text_dim":       "#8891B0",
    "text_faint":     "#4A5070",
    "warn":           "#F0A030",
    "section_lbl":    "#5A6488",
    "plot_bg":        "#1A1D27",
    "plot_grid":      "#2E3350",
    "plot_spine":     "#2E3350",
    "ecg_line":       "#FFFFFF",
    "cam_fill":       "#E05560",
    "cam_line":       "#FF8A94",
    "cam_zero":       "#1A1D27",
}

C = dict(LIGHT)
DARK_MODE = False


def _gradcam_cmap():
    return LinearSegmentedColormap.from_list(
        "gcam", [C["cam_zero"], C["af_red"]])


# ─────────────────────────────────────────────────────────────────
# DDNN MODEL
# ─────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.squeeze    = nn.AdaptiveAvgPool1d(1)
            self.excitation = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _ = x.size()
            y = self.squeeze(x).view(b, c)
            y = self.excitation(y).view(b, c, 1)
            return x * y.expand_as(x)

    class DenseBlock(nn.Module):
        def __init__(self, in_channels, growth_rate, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                self.layers.append(nn.Sequential(
                    nn.BatchNorm1d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(in_channels + i * growth_rate, growth_rate,
                              kernel_size=3, padding=1, bias=False)
                ))

        def forward(self, x):
            features = [x]
            for layer in self.layers:
                features.append(layer(torch.cat(features, 1)))
            return torch.cat(features, 1)

    class TransitionLayer(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.bn   = nn.BatchNorm1d(in_channels)
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        def forward(self, x):
            return self.pool(self.conv(F.relu(self.bn(x))))

    class DDNN(nn.Module):
        def __init__(self, in_channels=1, growth_rate=6,
                     block_config=(2, 4, 6, 4), reduction=0.5, num_classes=1):
            super().__init__()
            self.stem_conv1 = nn.Conv1d(in_channels, 16, kernel_size=7,  padding=3,  bias=False)
            self.stem_conv2 = nn.Conv1d(in_channels, 16, kernel_size=15, padding=7,  bias=False)
            self.stem_conv3 = nn.Conv1d(in_channels, 16, kernel_size=23, padding=11, bias=False)
            self.stem_bn    = nn.BatchNorm1d(48)
            self.stem_relu  = nn.ReLU(inplace=True)
            self.stem_pool  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            num_features    = 48
            self.blocks      = nn.ModuleList()
            self.se_blocks   = nn.ModuleList()
            self.transitions = nn.ModuleList()
            for i, num_layers in enumerate(block_config):
                self.se_blocks.append(SEBlock(num_features, reduction=16))
                block = DenseBlock(num_features, growth_rate, num_layers)
                self.blocks.append(block)
                num_features += num_layers * growth_rate
                if i != len(block_config) - 1:
                    out_features = int(num_features * reduction)
                    self.transitions.append(TransitionLayer(num_features, out_features))
                    num_features = out_features
            self.final_bn    = nn.BatchNorm1d(num_features)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc          = nn.Linear(num_features, num_classes)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.cat([self.stem_conv1(x), self.stem_conv2(x), self.stem_conv3(x)], dim=1)
            x = self.stem_relu(self.stem_bn(x))
            x = self.stem_pool(x)
            for i, (se, block) in enumerate(zip(self.se_blocks, self.blocks)):
                x = se(x)
                x = block(x)
                if i < len(self.transitions):
                    x = self.transitions[i](x)
            x = F.relu(self.final_bn(x))
            x = self.global_pool(x).squeeze(-1)
            return self.fc(x)

    class GradCAM1D:
        def __init__(self, model, target_layer, device):
            self.model        = model
            self.target_layer = target_layer
            self.device       = device
            self.gradients    = None
            self.activations  = None
            self._register_hooks()

        def _register_hooks(self):
            def fwd(m, inp, out): self.activations = out.detach()
            def bwd(m, gin, gout): self.gradients   = gout[0].detach()
            self.target_layer.register_forward_hook(fwd)
            self.target_layer.register_full_backward_hook(bwd)

        def generate(self, x):
            self.model.eval()
            x     = x.to(self.device)
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()
            self.model.zero_grad()
            logit.backward()
            weights = self.gradients.mean(dim=-1, keepdim=True)
            cam     = F.relu((weights * self.activations).sum(dim=1))
            cam     = cam.squeeze(0).cpu().numpy()
            cam_up  = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=x.shape[1], mode="linear", align_corners=False
            ).squeeze().numpy()
            lo, hi  = cam_up.min(), cam_up.max()
            cam_up  = (cam_up - lo) / (hi - lo + 1e-8)
            return cam_up, logit.item(), prob


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING ENGINE
# ─────────────────────────────────────────────────────────────────
def preprocess_segment(raw_padded, n_leads):
    processed = []
    for i in range(n_leads):
        sig = raw_padded[:, i].astype(np.float64)
        sig = scipy_signal.filtfilt(BUTTER_B, BUTTER_A, sig)
        if len(sig) > 2 * PAD_SAMPLES:
            sig = sig[PAD_SAMPLES:-PAD_SAMPLES]
        mu, sd = sig.mean(), sig.std()
        if sd > 0:
            sig = (sig - mu) / sd
        processed.append(sig)
    return np.stack(processed, axis=-1).astype(np.float32)


def load_ecg_file(filepath):
    ext = Path(filepath).suffix.lower()
    if ext in (".dat", ".hea"):
        if not WFDB_AVAILABLE:
            raise RuntimeError("wfdb not installed — run: pip install wfdb")
        rec   = wfdb.rdsamp(str(Path(filepath).with_suffix("")))
        sig   = rec[0]
        fs    = rec[1]["fs"]
        names = rec[1].get("sig_name", [f"Lead {i}" for i in range(sig.shape[1])])
        return sig, fs, sig.shape[1], names
    elif ext == ".h5":
        if not H5PY_AVAILABLE:
            raise RuntimeError("h5py not installed — run: pip install h5py")
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            for k in ("ecg", "signal", "data", "ecg_data"):
                if k in f:
                    sig = f[k][()].astype(np.float32)
                    break
            else:
                sig = f[keys[0]][()].astype(np.float32)
        if sig.ndim == 1:
            sig = sig[:, np.newaxis]
        if sig.shape[0] < sig.shape[1]:
            sig = sig.T
        fs      = 200
        n_leads = sig.shape[1]
        names   = ["Lead I", "Lead II"] if n_leads == 2 else [f"Lead {i}" for i in range(n_leads)]
        return sig, fs, n_leads, names
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .dat/.hea or .h5")


def resample_if_needed(sig, fs_orig, fs_target=TARGET_FS):
    if fs_orig == fs_target:
        return sig
    n_t = int(round(len(sig) * fs_target / fs_orig))
    if sig.ndim == 1:
        return scipy_signal.resample(sig, n_t)
    out = np.zeros((n_t, sig.shape[1]), dtype=np.float32)
    for i in range(sig.shape[1]):
        out[:, i] = scipy_signal.resample(sig[:, i], n_t)
    return out

def segment_recording(sig, n_leads):
    total = len(sig)
    segs = []
    for i in range(total // WINDOW_SAMPLES):
        s = i * WINDOW_SAMPLES
        e = s + WINDOW_SAMPLES
        if s < PAD_SAMPLES or e + PAD_SAMPLES > total:
            continue
        raw = sig[s - PAD_SAMPLES : e + PAD_SAMPLES]
        segs.append((preprocess_segment(raw, n_leads), i))
    return segs

# ─────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────
def run_inference(model, segments, device, n_leads, threshold=0.5,
                  progress_callback=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed.")
    results = []
    model.eval()
    n_segs = len(segments)
    with torch.no_grad():
        for seg_no, (seg_data, seg_idx) in enumerate(segments):
            for li in range(n_leads):
                x      = seg_data[:, li:li+1]
                xt     = torch.FloatTensor(x).unsqueeze(0).to(device)
                logit  = model(xt)
                prob   = torch.sigmoid(logit).item()
                results.append({
                    "seg_idx":      seg_idx,
                    "lead_idx":     li,
                    "prob":         prob,
                    "pred":         1 if prob >= threshold else 0,
                    "segment_data": seg_data,
                })
            if progress_callback is not None and n_segs > 0:
                progress_callback(seg_no + 1, n_segs)
    return results


def compute_gradcam(model, target_layer, x_lead_np, device):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed.")
    gcam = GradCAM1D(model, target_layer, device)
    xt   = torch.FloatTensor(x_lead_np).unsqueeze(0)
    cam, logit, prob = gcam.generate(xt)
    return cam, prob


# ─────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────
def generate_pdf_report(save_path, filename, lead_names, per_lead_stats, total_segments):
    from matplotlib.backends.backend_pdf import PdfPages

    BG, FG, DIM, RED, GRN, BLU = (
        "#FFFFFF", "#1A1F36", "#4A5568",
        "#C0392B", "#1A7A4A", "#2563EB")

    # ── Column x-positions (axes-fraction, 0–1) ──────────────────
    # "lead" is left-anchored; all numeric columns are right-anchored.
    COL_X = {
        "lead":  0.05,   # left edge of lead name
        "total": 0.40,   # right edge of "Total Segs"
        "af":    0.55,   # right edge of "AF Segs"
        "nonaf": 0.73,   # right edge of "Non-AF Segs"  ← more breathing room
        "pct":   0.90,   # right edge of "AF %"
    }

    with PdfPages(save_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.axis("off")

        y = 0.97

        def txt(text, y, size=11, color=FG, weight="normal", x=0.05):
            ax.text(x, y, text, transform=ax.transAxes,
                    fontsize=size, color=color, fontweight=weight, va="top")

        def table_row(lead, total, af, nonaf, pct, y, color=FG):
            ax.text(COL_X["lead"],  y, lead,       transform=ax.transAxes,
                    fontsize=9, color=color, va="top", ha="left")
            ax.text(COL_X["total"], y, str(total),  transform=ax.transAxes,
                    fontsize=9, color=color, va="top", ha="right")
            ax.text(COL_X["af"],    y, str(af),     transform=ax.transAxes,
                    fontsize=9, color=color, va="top", ha="right")
            ax.text(COL_X["nonaf"], y, str(nonaf),  transform=ax.transAxes,
                    fontsize=9, color=color, va="top", ha="right")
            ax.text(COL_X["pct"],   y, pct,         transform=ax.transAxes,
                    fontsize=9, color=color, va="top", ha="right")

        # ── Title block ───────────────────────────────────────────
        txt("AFiNET — Atrial Fibrillation Detection Report",
            y, 16, FG, "bold");                                y -= 0.040
        txt("Model: DDNN (Deep Densely Connected Neural Network, Cai et al. 2020)",
            y, 9,  DIM);                                       y -= 0.025
        txt(f"File:  {filename}", y, 10, DIM);                 y -= 0.050

        total_af = sum(s["af_count"] for s in per_lead_stats.values())
        overall  = "AF DETECTED" if total_af > 0 else "NO AF DETECTED"
        col      = RED if total_af > 0 else GRN
        txt(f"Overall Result:  {overall}", y, 14, col, "bold"); y -= 0.060

        txt(f"Total 30-second segments analysed: {total_segments}",
            y, 11, FG);                                        y -= 0.050

        # ── Table header ──────────────────────────────────────────
        txt("Per-Lead Summary", y, 13, FG, "bold");            y -= 0.030

        ax.text(COL_X["lead"],  y, "Lead",         transform=ax.transAxes,
                fontsize=9, color=BLU, fontweight="bold", va="top", ha="left")
        ax.text(COL_X["total"], y, "Total Segs",   transform=ax.transAxes,
                fontsize=9, color=BLU, fontweight="bold", va="top", ha="right")
        ax.text(COL_X["af"],    y, "AF Segs",      transform=ax.transAxes,
                fontsize=9, color=BLU, fontweight="bold", va="top", ha="right")
        ax.text(COL_X["nonaf"], y, "Non-AF Segs",  transform=ax.transAxes,
                fontsize=9, color=BLU, fontweight="bold", va="top", ha="right")
        ax.text(COL_X["pct"],   y, "AF %",         transform=ax.transAxes,
                fontsize=9, color=BLU, fontweight="bold", va="top", ha="right")
        y -= 0.022

        # Divider line in axes coordinates
        ax.plot([COL_X["lead"], 0.92], [y + 0.008, y + 0.008],
                color="#CDD3E0", linewidth=0.8, transform=ax.transAxes)
        y -= 0.018

        # ── Table rows ────────────────────────────────────────────
        for i, name in enumerate(lead_names):
            s   = per_lead_stats.get(i, {"total": 0, "af_count": 0, "non_af_count": 0})
            pct = (s["af_count"] / s["total"] * 100) if s["total"] > 0 else 0.0
            c   = RED if s["af_count"] > 0 else GRN
            table_row(name, s["total"], s["af_count"], s["non_af_count"],
                      f"{pct:.1f}%", y, color=c)
            y -= 0.025

        # ── Notes ─────────────────────────────────────────────────
        y -= 0.030
        txt("Notes", y, 11, FG, "bold");                       y -= 0.025
        for note in [
            "• Predictions made on 30-second lead-agnostic segments at 125 Hz.",
            "• This tool is a diagnostic support aid only. Not a substitute for",
            "  clinical diagnosis by a qualified cardiologist.",
            "• Grad-CAM heatmaps highlight ECG regions most influential for each",
            "  prediction. Visualise individual segments in the GUI.",
        ]:
            txt(note, y, 9, DIM);                              y -= 0.022

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    return save_path


# ─────────────────────────────────────────────────────────────────
# SCROLLABLE FRAME HELPER
# ─────────────────────────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    """A tk.Frame whose content scrolls vertically.

    Use ``self.inner`` as the parent for child widgets.
    The outer frame fills the container; the inner canvas/scrollbar
    handle all the scrolling.
    """

    def __init__(self, parent, bg, **kw):
        super().__init__(parent, bg=bg, **kw)

        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0,
                                 bd=0, yscrollincrement=15)
        self._vsb    = ttk.Scrollbar(self, orient="vertical",
                                     command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self._vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse-wheel scrolling - Globally routed by AFiNETApp now
        # to prevent memory leaks and bind_all overwrites on theme toggling

    def _on_inner_configure(self, event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Make the inner frame at least as wide as the canvas
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _on_mousewheel(self, event):
        # Ensure scroll only happens when hovering over the ScrollableFrame or its children
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return
        if widget and (str(widget) == str(self) or str(widget).startswith(str(self) + ".")):
            import sys
            if event.num == 4:
                self._canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self._canvas.yview_scroll(3, "units")
            else:
                if sys.platform == "darwin":
                    # With yscrollincrement=15, native trackpad delta directly translates
                    # to smooth, line-by-line sub-pixel scrolling
                    delta = int(-1 * event.delta)
                else:
                    # Windows ticks evaluate to 1. Scroll 4 units (60 pixels) per tick
                    delta = int(-1 * (event.delta / 120)) * 4
                if delta != 0:
                    self._canvas.yview_scroll(delta, "units")

    def scroll_to_top(self):
        self._canvas.yview_moveto(0.0)


# ─────────────────────────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────────────────────────
class AFiNETApp(tk.Tk):

    _zoom_level = 1.0
    _zoom_min   = 1.0
    _zoom_max   = 20.0
    _zoom_step  = 1.5

    _plot_ecg    = None
    _plot_cam    = None
    _plot_result = None
    _plot_axes   = []

    def __init__(self):
        super().__init__()
        global DARK_MODE
        DARK_MODE = False

        self.title("AFiNET — Atrial Fibrillation Detection System")
        self.configure(bg=C["bg"])
        self.minsize(1300, 780)
        self.geometry("1440x880")

        self.ecg_filepath = tk.StringVar(value="")
        self.model_path   = tk.StringVar(value="")
        self.status_text  = tk.StringVar(
            value="Ready — load a model and an ECG file to begin.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self._hscroll_pos = 0.0

        self._model        = None
        self._device       = None
        self._target_layer = None

        self._segments       = []
        self._results        = []
        self._lead_names     = []
        self._n_leads        = 0
        self._per_lead_stats = {}
        self._selected_result = None
        self._selected_idx    = -1
        self._current_cam     = None

        self._sort_col   = "seg"
        self._sort_asc   = True
        self._filter_lead_var = None
        self._filter_pred_var = None

        self._gradcam_queue  = queue.Queue()
        self._gradcam_thread = threading.Thread(
            target=self._gradcam_loop, daemon=True)
        self._gradcam_thread.start()

        self._fnt_header  = Font(family="Helvetica", size=22, weight="bold")
        self._fnt_title   = Font(family="Helvetica", size=16, weight="bold")
        self._fnt_label   = Font(family="Helvetica", size=15)
        self._fnt_small   = Font(family="Helvetica", size=13)
        self._fnt_tiny    = Font(family="Helvetica", size=11)
        self._fnt_result  = Font(family="Helvetica", size=26, weight="bold")
        self._fnt_mono    = Font(family="Courier",   size=13)
        self._fnt_section = Font(family="Helvetica", size=12, weight="bold")

        self._build_ui()
        self._apply_ttk_theme()
        
        # Single global mousewheel router for all custom scrollable panes
        self.bind_all("<MouseWheel>",       self._on_global_mousewheel)
        self.bind_all("<Shift-MouseWheel>", self._on_global_mousewheel)
        self.bind_all("<Button-4>",         self._on_global_mousewheel)
        self.bind_all("<Button-5>",         self._on_global_mousewheel)
        
        # Pinch-to-zoom binds
        self.bind_all("<Command-MouseWheel>", self._on_global_zoom)
        self.bind_all("<Control-MouseWheel>", self._on_global_zoom)
        
        self.after(100, self._autoload_model)

    # ─────────────────────────────────────────────────────────────
    # THEME
    # ─────────────────────────────────────────────────────────────
    def _apply_ttk_theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TFrame",          background=C["bg"])
        s.configure("Surface.TFrame",  background=C["surface"])

        s.configure("TLabel",
                    background=C["bg"], foreground=C["text"],
                    font=self._fnt_label)
        s.configure("Surface.TLabel",
                    background=C["surface"], foreground=C["text"],
                    font=self._fnt_label)
        s.configure("Dim.TLabel",
                    background=C["surface"], foreground=C["text_dim"],
                    font=self._fnt_small)

        s.configure("TButton",
                    background=C["surface2"], foreground=C["text"],
                    font=self._fnt_small, borderwidth=1,
                    focusthickness=0, padding=(10, 5), relief="flat")
        s.map("TButton",
              background=[("active", C["accent_dim"]),
                          ("disabled", C["surface2"])],
              foreground=[("disabled", C["text_faint"])])

        s.configure("Accent.TButton",
                    background=C["accent"], foreground="#FFFFFF",
                    font=Font(family="Helvetica", size=15, weight="bold"),
                    borderwidth=0, focusthickness=0, padding=(14, 8))
        s.map("Accent.TButton",
              background=[("active", "#1D4ED8"),
                          ("disabled", C["surface2"])],
              foreground=[("disabled", C["text_faint"])])

        s.configure("Zoom.TButton",
                    background=C["surface2"], foreground=C["text"],
                    font=Font(family="Helvetica", size=18, weight="bold"),
                    borderwidth=1, focusthickness=0, padding=(8, 3))
        s.map("Zoom.TButton",
              background=[("active", C["accent_dim"]),
                          ("disabled", C["surface2"])],
              foreground=[("disabled", C["text_faint"])])

        s.configure("TScrollbar",
                    background=C["surface2"], troughcolor=C["bg"],
                    arrowcolor=C["text_dim"], borderwidth=0)
        s.configure("Horizontal.TProgressbar",
                    troughcolor=C["surface2"], background=C["accent"],
                    borderwidth=0, thickness=5)
        s.configure("Treeview",
                    background=C["surface"], foreground=C["text"],
                    fieldbackground=C["surface"], borderwidth=0,
                    rowheight=27, font=self._fnt_mono)
        s.configure("Treeview.Heading",
                    background=C["surface2"], foreground=C["text_dim"],
                    borderwidth=0, font=self._fnt_tiny)
        s.map("Treeview",
              background=[("selected", C["accent_dim"])],
              foreground=[("selected", C["text"])])

    def _toggle_theme(self):
        global C, DARK_MODE
        DARK_MODE = not DARK_MODE
        C.update(DARK if DARK_MODE else LIGHT)
        for w in self.winfo_children():
            w.destroy()
        self._zoom_level      = 1.0
        self._left_collapsed  = False
        self._right_collapsed = False
        self._build_ui()
        self._apply_ttk_theme()

        if self.model_path.get() and self._model is not None:
            self._model_path_label.configure(
                text=Path(self.model_path.get()).name, fg=C["text"])
            self._model_status_lbl.configure(
                text=f"✓ Loaded on {str(self._device).upper()}", fg=C["ok_green"])
        if self.ecg_filepath.get():
            fp = self.ecg_filepath.get()
            hea_sibling = Path(fp).with_suffix(".hea")
            if Path(fp).suffix.lower() == ".dat" and hea_sibling.exists():
                label_text = f"{Path(fp).name}  +  {hea_sibling.name}"
            else:
                label_text = Path(fp).name
            self._file_label.configure(text=label_text, fg=C["text"])
            self._file_info_lbl.configure(
                text="File selected — ready to run.")
        self._maybe_enable_run()

        if self._results:
            self._update_ui_after_analysis(len(self._segments), self._results)

        if self._plot_ecg is not None and self._plot_result is not None:
            self._draw_gradcam(
                self._plot_ecg, self._plot_cam,
                self._plot_result, self._plot_result["prob"])
            self._update_result_card(self._plot_result)
        else:
            self._draw_placeholder()

    # ─────────────────────────────────────────────────────────────
    # BUILD UI
    # ─────────────────────────────────────────────────────────────
    _left_collapsed  = False
    _right_collapsed = False
    _left_width      = 310
    _right_width     = 350

    def _build_ui(self):
        self.configure(bg=C["bg"])
        self._build_header()

        outer = tk.Frame(self, bg=C["bg"])
        outer.pack(fill="both", expand=True)

        self._paned = tk.PanedWindow(
            outer,
            orient="horizontal",
            sashwidth=5,
            sashrelief="flat",
            bg=C["border"],
            bd=0,
            showhandle=False,
        )
        self._paned.pack(fill="both", expand=True)

        self._left_outer   = tk.Frame(self._paned, bg=C["surface"])
        self._center_outer = tk.Frame(self._paned, bg=C["bg"])
        self._right_outer  = tk.Frame(self._paned, bg=C["surface"])

        self._paned.add(self._left_outer,   minsize=0, width=self._left_width,  stretch="never")
        self._paned.add(self._center_outer, minsize=200,                         stretch="always")
        self._paned.add(self._right_outer,  minsize=0, width=self._right_width,  stretch="never")

        self._build_left_panel(self._left_outer)
        self._build_center_panel(self._center_outer)
        self._build_right_panel(self._right_outer)

        self._build_sash_toggles(outer)
        self._build_status_bar()

    def _build_sash_toggles(self, container):
        self._PINCH_PX = 48

        pill_kw = dict(
            relief="flat", bd=0, cursor="hand2",
            highlightthickness=0,
            font=Font(family="Helvetica", size=9),
            fg=C["text_dim"],
            activeforeground=C["text"],
            activebackground=C["accent_dim"],
            padx=0, pady=0,
        )
        self._sash_btn_left = tk.Button(
            self._paned, bg=C["surface2"], text="›",
            command=self._expand_left, **pill_kw)
        self._sash_btn_right = tk.Button(
            self._paned, bg=C["surface2"], text="‹",
            command=self._expand_right, **pill_kw)

        self._sash_btn_left.place_forget()
        self._sash_btn_right.place_forget()

        self._paned.bind("<Configure>",       self._sync_sash_buttons)
        self._paned.bind("<B1-Motion>",       self._sync_sash_buttons)
        self._paned.bind("<ButtonRelease-1>", self._sync_sash_buttons)
        self._sash_poll()

    def _sash_poll(self):
        try:
            self._sync_sash_buttons()
        except Exception:
            pass
        if self._paned.winfo_exists():
            self._paned.after(60, self._sash_poll)

    def _sync_sash_buttons(self, event=None):
        try:
            lx = self._paned.sash_coord(0)[0]
            rx = self._paned.sash_coord(1)[0]
        except Exception:
            return

        pw_h = self._paned.winfo_height()
        pw_w = self._paned.winfo_width()
        if pw_h < 2 or pw_w < 2:
            return

        PINCH = self._PINCH_PX
        PW, PH = 14, 40
        mid = pw_h // 2 - PH // 2

        left_width  = lx
        right_width = pw_w - rx

        if left_width <= PINCH:
            self._left_collapsed = left_width <= 4
            pill_x = max(lx + 3, 2)
            self._sash_btn_left.place(x=pill_x, y=mid, width=PW, height=PH)
            self._sash_btn_left.lift()
        else:
            self._left_collapsed = False
            self._left_width = lx
            self._sash_btn_left.place_forget()

        if right_width <= PINCH:
            self._right_collapsed = right_width <= 9
            pill_x = min(rx - PW - 3, pw_w - PW - 2)
            self._sash_btn_right.place(x=pill_x, y=mid, width=PW, height=PH)
            self._sash_btn_right.lift()
        else:
            self._right_collapsed = False
            self._right_width = pw_w - rx
            self._sash_btn_right.place_forget()

    def _toggle_left(self):
        if self._left_collapsed:
            self._expand_left()
        else:
            self._collapse_left()

    def _toggle_right(self):
        if self._right_collapsed:
            self._expand_right()
        else:
            self._collapse_right()

    def _collapse_left(self):
        try:
            sx = self._paned.sash_coord(0)[0]
            if sx > 4:
                self._left_width = sx
        except Exception:
            pass
        self._left_collapsed = True
        self._paned.sash_place(0, 0, 0)
        self._sync_sash_buttons()

    def _expand_left(self):
        self._left_collapsed = False
        w = max(self._left_width, 260)
        self._paned.sash_place(0, w, 0)
        self._sync_sash_buttons()

    def _collapse_right(self):
        try:
            sx1  = self._paned.sash_coord(1)[0]
            pw_w = self._paned.winfo_width()
            gap  = pw_w - sx1
            if gap > 10:
                self._right_width = gap
        except Exception:
            pass
        self._right_collapsed = True
        pw_w = self._paned.winfo_width()
        self._paned.sash_place(1, pw_w, 0)
        self._sync_sash_buttons()

    def _expand_right(self):
        self._right_collapsed = False
        pw_w = self._paned.winfo_width()
        w    = max(self._right_width, 280)
        self._paned.sash_place(1, pw_w - w, 0)
        self._sync_sash_buttons()

    # ── HEADER ────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=C["surface"], height=64)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="AFiNET",
                 bg=C["surface"], fg=C["accent"],
                 font=self._fnt_header
                 ).pack(side="left", padx=(20, 6), pady=10)
        tk.Label(hdr, text="Atrial Fibrillation Detection System",
                 bg=C["surface"], fg=C["text"],
                 font=Font(family="Helvetica", size=16)
                 ).pack(side="left", padx=(0, 20), pady=10)

        self._theme_btn = tk.Button(
            hdr,
            text="🌙 Dark" if not DARK_MODE else "☀ Light",
            bg=C["surface2"], fg=C["text_dim"],
            font=self._fnt_tiny, relief="flat", bd=0,
            padx=12, pady=5, cursor="hand2",
            command=self._toggle_theme)
        self._theme_btn.pack(side="right", padx=(0, 18), pady=14)

        tk.Label(hdr,
                 text=("⚠  For use by qualified medical professionals only. "
                       "Not a substitute for clinical diagnosis."),
                 bg=C["surface"], fg=C["warn"],
                 font=Font(family="Helvetica", size=12)
                 ).pack(side="right", padx=(0, 8), pady=14)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

    # ── LEFT PANEL ────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=C["surface"], width=310)
        frame.pack(fill="both", expand=True)
        frame.pack_propagate(False)

        pad = dict(padx=16, pady=5)

        self._model_path_label = tk.Label(frame, text="", bg=C["surface"])
        self._model_status_lbl = tk.Label(frame, text="", bg=C["surface"])

        # ECG FILE
        self._section_label(frame, "ECG RECORDING")
        tk.Label(frame,
                 text="Select .dat + .hea together, or a single .h5 file",
                 bg=C["surface"], fg=C["text_faint"],
                 font=self._fnt_tiny, anchor="w"
                 ).pack(fill="x", padx=16, pady=(0, 4))
        file_row = tk.Frame(frame, bg=C["surface"])
        file_row.pack(fill="x", **pad)
        self._file_label = tk.Label(
            file_row, text="No file selected",
            bg=C["surface"], fg=C["text_dim"],
            font=self._fnt_small, wraplength=200,
            anchor="w", justify="left")
        self._file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Browse",
                   command=self._on_browse_file
                   ).pack(side="right", pady=2)
        self._file_info_lbl = tk.Label(
            frame, text="", bg=C["surface"],
            fg=C["text_dim"], font=self._fnt_tiny,
            anchor="w", wraplength=278)
        self._file_info_lbl.pack(fill="x", padx=16)
        self._sep(frame)

        # RUN
        self._run_btn = ttk.Button(
            frame, text="▶   Run Analysis",
            style="Accent.TButton",
            command=self._on_run_analysis,
            state="disabled")
        self._run_btn.pack(fill="x", padx=16, pady=(8, 4))
        self._progress = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100,
            mode="determinate", style="Horizontal.TProgressbar")
        self._progress.pack(fill="x", padx=16, pady=(2, 8))
        self._sep(frame)

        # SEGMENTS
        self._section_label(frame, "SEGMENTS")
        tk.Label(frame,
                 text="Click a row to view ECG + Grad-CAM",
                 bg=C["surface"], fg=C["text_faint"],
                 font=self._fnt_tiny, anchor="w"
                 ).pack(fill="x", padx=16, pady=(0, 4))

        filter_frame = tk.Frame(frame, bg=C["surface"])
        filter_frame.pack(fill="x", padx=10, pady=(0, 4))

        tk.Label(filter_frame, text="Filter:",
                 bg=C["surface"], fg=C["text_dim"],
                 font=self._fnt_tiny
                 ).grid(row=0, column=0, sticky="w", padx=(2, 4))
        tk.Label(filter_frame, text="Lead:",
                 bg=C["surface"], fg=C["text_dim"],
                 font=self._fnt_tiny
                 ).grid(row=0, column=1, sticky="w", padx=(0, 2))

        self._filter_lead_var = tk.StringVar(value="All")
        self._filter_lead_cb  = ttk.Combobox(
            filter_frame, textvariable=self._filter_lead_var,
            state="readonly", width=9, font=self._fnt_tiny)
        self._filter_lead_cb["values"] = ["All"]
        self._filter_lead_cb.grid(row=0, column=2, padx=(0, 8))
        self._filter_lead_cb.bind("<<ComboboxSelected>>", self._on_filter_changed)

        tk.Label(filter_frame, text="Pred:",
                 bg=C["surface"], fg=C["text_dim"],
                 font=self._fnt_tiny
                 ).grid(row=0, column=3, sticky="w", padx=(0, 2))

        self._filter_pred_var = tk.StringVar(value="All")
        self._filter_pred_cb  = ttk.Combobox(
            filter_frame, textvariable=self._filter_pred_var,
            state="readonly", width=8, font=self._fnt_tiny)
        self._filter_pred_cb["values"] = ["All", "AF", "Non-AF"]
        self._filter_pred_cb.grid(row=0, column=4, padx=(0, 4))
        self._filter_pred_cb.bind("<<ComboboxSelected>>", self._on_filter_changed)

        tree_frame = tk.Frame(frame, bg=C["surface"])
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        cols = ("seg", "lead", "pred", "prob")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols,
            show="headings", selectmode="browse")

        self._tree.column("seg",  width=52, anchor="center")
        self._tree.column("lead", width=68, anchor="center")
        self._tree.column("pred", width=72, anchor="center")
        self._tree.column("prob", width=64, anchor="center")

        self._tree.heading("seg",  command=lambda: self._on_sort_col("seg"))
        self._tree.heading("lead", command=lambda: self._on_sort_col("lead"))
        self._tree.heading("pred", command=lambda: self._on_sort_col("pred"))
        self._tree.heading("prob", command=lambda: self._on_sort_col("prob"))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.tag_configure("af",     foreground=C["af_red"])
        self._tree.tag_configure("normal", foreground=C["ok_green"])
        self._tree.bind("<<TreeviewSelect>>", self._on_segment_selected)

        self._refresh_heading_labels()
        self._sep(frame)

        self._report_btn = ttk.Button(
            frame, text="⬇   Export PDF Report",
            command=self._on_generate_report,
            state="disabled")
        self._report_btn.pack(fill="x", padx=16, pady=(4, 14))

    # ─────────────────────────────────────────────────────────────
    # SORT & FILTER HELPERS
    # ─────────────────────────────────────────────────────────────
    _COL_LABELS = {"seg": "Seg #", "lead": "Lead", "pred": "Pred", "prob": "P(AF)"}

    def _refresh_heading_labels(self):
        for col, base in self._COL_LABELS.items():
            if col == self._sort_col:
                arrow = " ▲" if self._sort_asc else " ▼"
                self._tree.heading(col, text=base + arrow)
            else:
                self._tree.heading(col, text=base)

    def _on_sort_col(self, col):
        if col == self._sort_col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        self._refresh_heading_labels()
        self._repopulate_tree()

    def _on_filter_changed(self, event=None):
        self._repopulate_tree()

    def _get_filtered_sorted_results(self):
        results = list(self._results)

        lead_filter = self._filter_lead_var.get() if self._filter_lead_var else "All"
        if lead_filter and lead_filter != "All":
            results = [r for r in results
                       if self._result_lead_name(r) == lead_filter]

        pred_filter = self._filter_pred_var.get() if self._filter_pred_var else "All"
        if pred_filter == "AF":
            results = [r for r in results if r["pred"] == 1]
        elif pred_filter == "Non-AF":
            results = [r for r in results if r["pred"] == 0]

        col = self._sort_col
        asc = self._sort_asc

        def sort_key(r):
            if col == "seg":
                return (r["seg_idx"],)
            elif col == "lead":
                return (self._result_lead_name(r), r["seg_idx"])
            elif col == "pred":
                return ("AF" if r["pred"] == 1 else "Non-AF", r["seg_idx"])
            elif col == "prob":
                return (r["prob"], r["seg_idx"])
            return (r["seg_idx"],)

        results.sort(key=sort_key, reverse=not asc)
        return results

    def _result_lead_name(self, r):
        li = r["lead_idx"]
        return self._lead_names[li] if li < len(self._lead_names) else f"L{li}"

    def _repopulate_tree(self):
        prev_result = self._selected_result

        self._tree.unbind("<<TreeviewSelect>>")
        self._tree.delete(*self._tree.get_children())

        filtered = self._get_filtered_sorted_results()
        self._filtered_results = filtered

        reselect_iid = None
        for r in filtered:
            ln  = self._result_lead_name(r)
            tag = "af" if r["pred"] == 1 else "normal"
            iid = self._tree.insert("", "end",
                               values=(r["seg_idx"], ln,
                                       "AF" if r["pred"] == 1 else "Non-AF",
                                       f"{r['prob']:.3f}"),
                               tags=(tag,))
            if (prev_result is not None
                    and r["seg_idx"]  == prev_result["seg_idx"]
                    and r["lead_idx"] == prev_result["lead_idx"]):
                reselect_iid = iid

        if reselect_iid:
            self._tree.selection_set(reselect_iid)
            self._tree.see(reselect_iid)

        self._tree.bind("<<TreeviewSelect>>", self._on_segment_selected)

    def _update_lead_filter_options(self):
        options = ["All"] + list(self._lead_names)
        self._filter_lead_cb["values"] = options
        self._filter_lead_var.set("All")

    # ── CENTER PANEL ──────────────────────────────────────────────
    def _build_center_panel(self, parent):
        outer = tk.Frame(parent, bg=C["bg"])
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.rowconfigure(1, weight=0)
        outer.rowconfigure(2, weight=0)
        outer.columnconfigure(0, weight=1)

        self._fig    = plt.Figure(figsize=(9, 5.5), facecolor=C["plot_bg"])
        self._canvas = FigureCanvasTkAgg(self._fig, master=outer)
        cw = self._canvas.get_tk_widget()
        cw.configure(bg=C["plot_bg"], highlightthickness=0)
        cw.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 0))

        ctrl = tk.Frame(outer, bg=C["surface2"], height=44)
        ctrl.grid(row=1, column=0, sticky="ew", padx=6, pady=(3, 0))

        tk.Label(ctrl, text="Zoom:",
                 bg=C["surface2"], fg=C["text_dim"],
                 font=self._fnt_small
                 ).grid(row=0, column=0, padx=(12, 6), pady=8)

        self._zoom_out_btn = ttk.Button(
            ctrl, text="−", style="Zoom.TButton",
            command=self._zoom_out, width=3)
        self._zoom_out_btn.grid(row=0, column=1, padx=(0, 4), pady=8)

        self._zoom_label = tk.Label(
            ctrl, text="1×",
            bg=C["surface2"], fg=C["text"],
            font=self._fnt_label, width=6, anchor="center")
        self._zoom_label.grid(row=0, column=2, padx=4)

        self._zoom_in_btn = ttk.Button(
            ctrl, text="+", style="Zoom.TButton",
            command=self._zoom_in, width=3)
        self._zoom_in_btn.grid(row=0, column=3, padx=(4, 12), pady=8)

        ttk.Button(ctrl, text="⟳ Reset",
                   command=self._zoom_reset
                   ).grid(row=0, column=4, padx=(0, 20), pady=8)

        ctrl.columnconfigure(5, weight=1)

        nav_frame = tk.Frame(ctrl, bg=C["surface2"])
        nav_frame.grid(row=0, column=5, padx=0, pady=6, sticky="ew")
        nav_frame.columnconfigure(1, weight=1)

        self._prev_seg_btn = ttk.Button(
            nav_frame, text="◀  Prev",
            command=self._prev_segment, state="disabled", width=9)
        self._prev_seg_btn.grid(row=0, column=0, padx=(0, 6))

        self._seg_nav_label = tk.Label(
            nav_frame, text="",
            bg=C["surface2"], fg=C["text_dim"],
            font=self._fnt_small, anchor="center", width=16)
        self._seg_nav_label.grid(row=0, column=1)

        self._next_seg_btn = ttk.Button(
            nav_frame, text="Next  ▶",
            command=self._next_segment, state="disabled", width=9)
        self._next_seg_btn.grid(row=0, column=2, padx=(6, 0))

        self._hscroll = ttk.Scrollbar(
            outer, orient="horizontal",
            command=self._on_hscroll)
        self._hscroll.grid(row=2, column=0, sticky="ew",
                           padx=6, pady=(0, 6))
        self._hscroll.set(0.0, 1.0)

        self._draw_placeholder()

    # ── RIGHT PANEL ───────────────────────────────────────────────
    def _build_right_panel(self, parent):
        # Outer container — holds the scrollable frame and fills the pane
        outer = tk.Frame(parent, bg=C["surface"])
        outer.pack(fill="both", expand=True)

        # ScrollableFrame fills the outer container
        self._right_scroll = ScrollableFrame(outer, bg=C["surface"])
        self._right_scroll.pack(fill="both", expand=True)

        # All content goes into self._right_scroll.inner
        frame = self._right_scroll.inner

        self._section_label(frame, "DIAGNOSIS")
        self._result_card = tk.Frame(
            frame, bg=C["surface2"],
            highlightbackground=C["border"], highlightthickness=1)
        self._result_card.pack(fill="x", padx=16, pady=8)

        self._result_lbl = tk.Label(
            self._result_card, text="—",
            bg=C["surface2"], fg=C["text_dim"],
            font=self._fnt_result)
        self._result_lbl.pack(pady=(18, 2))

        self._conf_lbl = tk.Label(
            self._result_card, text="",
            bg=C["surface2"], fg=C["text_dim"],
            font=Font(family="Helvetica", size=16))
        self._conf_lbl.pack(pady=(0, 4))

        self._seg_info_lbl = tk.Label(
            self._result_card, text="",
            bg=C["surface2"], fg=C["text_dim"],
            font=self._fnt_small, wraplength=310, justify="left")
        self._seg_info_lbl.pack(padx=14, pady=(0, 14), anchor="w")

        self._sep(frame)

        self._section_label(frame, "RECORDING SUMMARY")
        self._summary_lbl = tk.Label(
            frame, text="No analysis run yet.",
            bg=C["surface"], fg=C["text_dim"],
            font=self._fnt_small, wraplength=318,
            justify="left", anchor="w")
        self._summary_lbl.pack(fill="x", padx=16, pady=6)
        self._sep(frame)

        self._section_label(frame, "PER-LEAD BREAKDOWN")
        self._lead_table_lbl = tk.Label(
            frame, text="—",
            bg=C["surface"], fg=C["text_dim"],
            font=self._fnt_mono, wraplength=318,
            justify="left", anchor="w")
        self._lead_table_lbl.pack(fill="x", padx=16, pady=6)
        self._sep(frame)

        self._section_label(frame, "GRAD-CAM")
        self._gradcam_lbl = tk.Label(
            frame,
            text=("Select a segment from the left panel\n"
                  "to generate the Grad-CAM heatmap overlay."),
            bg=C["surface"], fg=C["text_faint"],
            font=self._fnt_small, wraplength=318, justify="left")
        self._gradcam_lbl.pack(fill="x", padx=16, pady=(6, 20))

    # ── STATUS BAR ────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C["surface"], height=30)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=C["border"], height=1).pack(fill="x", side="top")
        tk.Label(bar, textvariable=self.status_text,
                 bg=C["surface"], fg=C["text_dim"],
                 font=self._fnt_tiny, anchor="w"
                 ).pack(side="left", padx=14, pady=5)
        tk.Label(bar,
                 text=(f"PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}  "
                       f"| wfdb: {'✓' if WFDB_AVAILABLE else '✗'}  "
                       f"| h5py: {'✓' if H5PY_AVAILABLE else '✗'}"),
                 bg=C["surface"], fg=C["text_faint"],
                 font=self._fnt_tiny, anchor="e"
                 ).pack(side="right", padx=14)

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────
    def _section_label(self, parent, text):
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", padx=0, pady=(12, 2))
        tk.Label(row, text=f"  {text}",
                 bg=C["surface"], fg=C["section_lbl"],
                 font=self._fnt_section
                 ).pack(side="left")
        tk.Frame(row, bg=C["border"], height=1).pack(
            side="left", fill="x", expand=True,
            padx=(6, 0), pady=7)

    def _sep(self, parent):
        tk.Frame(parent, bg=C["border"], height=1).pack(
            fill="x", padx=12, pady=4)

    def _set_status(self, msg):
        self.status_text.set(msg)
        self.update_idletasks()

    # ─────────────────────────────────────────────────────────────
    # ZOOM / SCROLL
    # ─────────────────────────────────────────────────────────────
    def _apply_zoom(self, factor):
        if self._plot_ecg is None:
            return
        old_zoom = self._zoom_level
        self._zoom_level = max(self._zoom_min, min(self._zoom_max, self._zoom_level * factor))
        
        # Avoid unnecessary redraws if hitting limits
        if abs(self._zoom_level - old_zoom) > 0.001:
            self._zoom_label.configure(text=f"{self._zoom_level:.1f}×")
            self._redraw_axes()

    def _zoom_in(self):
        self._apply_zoom(self._zoom_step)

    def _zoom_out(self):
        self._apply_zoom(1 / self._zoom_step)

    def _zoom_reset(self):
        if self._plot_ecg is None:
            return
        self._zoom_level  = 1.0
        self._hscroll_pos = 0.0
        self._zoom_label.configure(text="1×")
        self._hscroll.set(0.0, 1.0)
        self._redraw_axes()

    def _is_hovering_center(self, event):
        if not hasattr(self, "_center_outer") or not self._center_outer.winfo_exists():
            return False
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return False
        return widget and str(widget).startswith(str(self._center_outer))

    def _on_global_zoom(self, event):
        if not self._is_hovering_center(event):
            return
            
        import sys
        if sys.platform == "darwin":
            factor = 1.05 ** event.delta
        else:
            factor = 1.1 ** (event.delta / 120)
        self._apply_zoom(factor)

    def _on_global_mousewheel(self, event):
        # Route to right pane
        if hasattr(self, "_right_scroll") and self._right_scroll.winfo_exists():
            self._right_scroll._on_mousewheel(event)
            
        # Route to center pane
        if hasattr(self, "_center_outer") and self._center_outer.winfo_exists():
            try:
                widget = self.winfo_containing(event.x_root, event.y_root)
            except Exception:
                return
            if widget and str(widget).startswith(str(self._center_outer)):
                self._on_center_mousewheel(event)

    def _on_center_mousewheel(self, event):
        if self._zoom_level <= 1.001:
            return
            
        import sys
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            if sys.platform == "darwin":
                # Limit jumps just like the right pane
                step = -1 if event.delta > 0 else 1
            else:
                step = -1 if event.delta > 0 else 1
                
        if step != 0:
            self._on_hscroll("scroll", step, "units")

    def _on_hscroll(self, *args):
        cmd = args[0]
        lo, hi = self._hscroll.get()
        span   = hi - lo
        if cmd == "moveto":
            frac = float(args[1])
        elif cmd == "scroll":
            step  = int(args[1])
            unit  = args[2]
            delta = span * (0.1 if unit == "units" else 1.0)
            frac  = lo + step * delta
        else:
            return
        frac = max(0.0, min(1.0 - span, frac))
        self._hscroll_pos = frac
        self._hscroll.set(frac, frac + span)
        self._redraw_axes()

    def _redraw_axes(self):
        if self._plot_ecg is None:
            return
        T      = len(self._plot_ecg)
        total  = T / TARGET_FS
        span   = total / self._zoom_level
        x_lo   = self._hscroll_pos * total
        x_hi   = x_lo + span
        if x_hi > total:
            x_hi = total
            x_lo = max(0.0, total - span)

        if self._zoom_level > 1.001:
            self._hscroll.set(x_lo / total, x_hi / total)
        else:
            self._hscroll.set(0.0, 1.0)

        n_ticks = 7
        ticks   = np.linspace(x_lo, x_hi, n_ticks)
        axes_to_update = getattr(self, "_plot_axes", None) or self._fig.get_axes()
        for ax in axes_to_update:
            ax.set_xlim(x_lo, x_hi)
            ax.set_xticks(ticks)
            labels = []
            for v in ticks:
                sv = f"{v:.1f}"
                if sv.endswith(".0"):
                    labels.append(sv[:-2])
                else:
                    labels.append(sv)

            ax.set_xticklabels(labels, fontsize=10, color=C["text_dim"])

        self._canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────
    # PLACEHOLDER
    # ─────────────────────────────────────────────────────────────
    def _draw_placeholder(self):
        self._fig.clear()
        self._fig.patch.set_facecolor(C["plot_bg"])
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(C["plot_bg"])
        ax.text(0.5, 0.5,
                "Load a model and an ECG file,\nthen click  ▶  Run Analysis",
                ha="center", va="center",
                transform=ax.transAxes,
                color=C["text_faint"], fontsize=17, style="italic")
        for sp in ax.spines.values():
            sp.set_color(C["plot_spine"])
        ax.tick_params(colors=C["text_faint"])
        self._canvas.draw()

    # ─────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────
    _DEFAULT_MODEL_PATH = (
        '/Users/sheianneseblante/Desktop/college/4th year/'
        '2nd sem/CMSC 198/XAI Coding/ddnn_FINAL.pth'
    )

    def _autoload_model(self):
        path = self._DEFAULT_MODEL_PATH
        if not TORCH_AVAILABLE:
            self._model_status_lbl.configure(
                text="PyTorch not installed.", fg=C["af_red"])
            return
        if not os.path.isfile(path):
            self._model_status_lbl.configure(
                text=f"Model not found: {path}", fg=C["af_red"])
            self._set_status("Model file not found — check the bundled path.")
            return
        try:
            self._set_status("Loading model…")
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            model = DDNN(in_channels=1, growth_rate=6,
                         block_config=[2, 4, 6, 4],
                         reduction=0.5, num_classes=1)
            ckpt  = torch.load(path, map_location=self._device,
                               weights_only=False)
            state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
            model.load_state_dict(state)
            model.to(self._device)
            model.eval()

            self._model        = model
            self._target_layer = model.blocks[-1].layers[-1][-1]
            short              = Path(path).name

            self._model_path_label.configure(text=short, fg=C["text"])
            self._model_status_lbl.configure(
                text=f"✓ Loaded on {str(self._device).upper()}",
                fg=C["ok_green"])
            self._set_status(
                f"Model loaded: {short}  [{str(self._device).upper()}]")
            self.model_path.set(path)
            self._maybe_enable_run()
        except Exception as e:
            self._model_status_lbl.configure(
                text=f"Load error: {str(e)[:80]}", fg=C["af_red"])
            self._set_status("Model load failed.")

    def _on_load_model(self):
        path = filedialog.askopenfilename(
            title="Select DDNN model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"),
                       ("All files", "*.*")])
        if not path:
            return
        if not TORCH_AVAILABLE:
            messagebox.showerror("Missing Dependency",
                "PyTorch not installed.\nRun: pip install torch")
            return
        try:
            self._set_status("Loading model…")
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            model = DDNN(in_channels=1, growth_rate=6,
                         block_config=[2, 4, 6, 4],
                         reduction=0.5, num_classes=1)
            ckpt  = torch.load(path, map_location=self._device,
                               weights_only=False)
            state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
            model.load_state_dict(state)
            model.to(self._device)
            model.eval()

            self._model        = model
            self._target_layer = model.blocks[-1].layers[-1][-1]
            short              = Path(path).name

            self._model_path_label.configure(text=short, fg=C["text"])
            self._model_status_lbl.configure(
                text=f"✓ Loaded on {str(self._device).upper()}",
                fg=C["ok_green"])
            self._set_status(
                f"Model loaded: {short}  [{str(self._device).upper()}]")
            self.model_path.set(path)
            self._maybe_enable_run()
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self._set_status("Model load failed.")

    def _on_browse_file(self):
        paths = filedialog.askopenfilenames(
            title="Select ECG Recording",
            filetypes=[("ECG files", "*.dat *.hea *.h5"),
                       ("WFDB .dat", "*.dat"),
                       ("WFDB .hea", "*.hea"),
                       ("HDF5 .h5",  "*.h5"),
                       ("All files", "*.*")])
        if not paths:
            return

        paths = list(paths)
        exts  = [Path(p).suffix.lower() for p in paths]

        h5_files  = [p for p, e in zip(paths, exts) if e == ".h5"]
        dat_files = [p for p, e in zip(paths, exts) if e == ".dat"]
        hea_files = [p for p, e in zip(paths, exts) if e == ".hea"]

        if h5_files:
            chosen = h5_files[0]
            if len(paths) > 1:
                ignored = [Path(p).name for p in paths if p != chosen]
                self._file_info_lbl.configure(
                    text=f"ℹ  .h5 selected — ignoring: {', '.join(ignored)}",
                    fg=C["warn"])
            else:
                self._file_info_lbl.configure(
                    text="File selected — ready to run.", fg=C["text_dim"])
            self._set_ecg_path(chosen)
            return

        if dat_files and hea_files:
            dat_stem = Path(dat_files[0]).stem
            hea_stem = Path(hea_files[0]).stem
            if dat_stem != hea_stem:
                self._file_label.configure(
                    text=f"{Path(dat_files[0]).name} + {Path(hea_files[0]).name}",
                    fg=C["text"])
                self._file_info_lbl.configure(
                    text=(f"⚠  File name mismatch: '{dat_stem}.dat' and "
                          f"'{hea_stem}.hea' must share the same base name."),
                    fg=C["af_red"])
                self._set_status("Mismatched .dat/.hea filenames.")
                self.ecg_filepath.set("")
                self._maybe_enable_run()
                return

            extra = [Path(p).name for p in paths
                     if Path(p).suffix.lower() not in (".dat", ".hea")]
            chosen = dat_files[0]
            pair_label = f"{Path(dat_files[0]).name}  +  {Path(hea_files[0]).name}"
            self._file_label.configure(text=pair_label, fg=C["text"])
            if extra:
                self._file_info_lbl.configure(
                    text=f"ℹ  Using .dat/.hea pair — ignoring: {', '.join(extra)}",
                    fg=C["warn"])
            else:
                self._file_info_lbl.configure(
                    text="File pair selected — ready to run.", fg=C["text_dim"])
            self.ecg_filepath.set(chosen)
            self._set_status(f"ECG file pair: {pair_label}")
            self._maybe_enable_run()
            return

        if dat_files and not hea_files:
            self._file_label.configure(
                text=Path(dat_files[0]).name, fg=C["text"])
            self._file_info_lbl.configure(
                text="⚠  Missing .hea file — please also select the matching header.",
                fg=C["af_red"])
            self._set_status("Missing .hea file for selected .dat.")
            self.ecg_filepath.set("")
            self._maybe_enable_run()
            return

        if hea_files and not dat_files:
            self._file_label.configure(
                text=Path(hea_files[0]).name, fg=C["text"])
            self._file_info_lbl.configure(
                text="⚠  Missing .dat file — please also select the matching data file.",
                fg=C["af_red"])
            self._set_status("Missing .dat file for selected .hea.")
            self.ecg_filepath.set("")
            self._maybe_enable_run()
            return

        self._file_info_lbl.configure(
            text="⚠  Unsupported file combination. Use .dat+.hea or a single .h5.",
            fg=C["af_red"])
        self._set_status("Unsupported file selection.")

    def _set_ecg_path(self, path):
        self.ecg_filepath.set(path)
        self._file_label.configure(text=Path(path).name, fg=C["text"])
        self._set_status(f"ECG file: {Path(path).name}")
        self._maybe_enable_run()

    def _maybe_enable_run(self):
        ok = bool(self.ecg_filepath.get()) and self._model is not None
        self._run_btn.configure(state="normal" if ok else "disabled")

    def _on_run_analysis(self):
        self._analysis_start = time.time()
        self._run_btn.configure(state="disabled", text="Analyzing…")
        self._report_btn.configure(state="disabled")
        self._tree.delete(*self._tree.get_children())
        self.progress_var.set(0)
        self._results    = []
        self._segments   = []
        self._filtered_results = []
        self._plot_ecg   = None
        self._plot_cam   = None
        self._plot_result = None
        self._zoom_level = 1.0
        self._hscroll_pos = 0.0
        self._zoom_label.configure(text="1×")
        self._hscroll.set(0.0, 1.0)
        self._sort_col = "seg"
        self._sort_asc = True
        self._refresh_heading_labels()
        if self._filter_lead_var:
            self._filter_lead_var.set("All")
        if self._filter_pred_var:
            self._filter_pred_var.set("All")
        self._draw_placeholder()
        threading.Thread(
            target=self._run_analysis_worker, daemon=True).start()

    def _run_analysis_worker(self):
        try:
            fp = self.ecg_filepath.get()
            self._set_status("Loading ECG file…")
            sig, fs, n_leads, lead_names = load_ecg_file(fp)
            self._lead_names = lead_names
            self._n_leads    = n_leads

            info = (f"{Path(fp).name}  ·  "
                    f"{n_leads} lead{'s' if n_leads > 1 else ''}  ·  "
                    f"fs={fs} Hz  ·  {len(sig)/fs/3600:.2f} h")
            self.after(0, lambda: self._file_info_lbl.configure(text=info))

            self._set_status("Resampling…")
            sig = resample_if_needed(sig, fs, TARGET_FS)
            self.after(0, lambda: self.progress_var.set(10))

            self._set_status("Segmenting & preprocessing…")
            segments = segment_recording(sig, n_leads)
            if not segments:
                self.after(0, lambda: messagebox.showwarning(
                    "No Segments",
                    "No valid 30-second segments found.\n"
                    "Recording may be too short or lack sufficient padding."))
                self.after(0, self._reset_run_button)
                return
            self._segments = segments
            n_segs = len(segments)
            self.after(0, lambda: self.progress_var.set(25))

            self._set_status(
                f"Running DDNN on {n_segs} segs × {n_leads} leads…")

            def _inference_progress(done, total):
                # Map segment progress linearly from 25 → 80
                pct = 25 + int((done / total) * 55)
                self.after(0, lambda p=pct: self.progress_var.set(p))

            results = run_inference(
                self._model, segments, self._device, n_leads,
                progress_callback=_inference_progress)
            self._results = results
            self.after(0, lambda: self.progress_var.set(80))

            per_lead = {}
            for r in results:
                li = r["lead_idx"]
                if li not in per_lead:
                    per_lead[li] = {
                        "total": 0, "af_count": 0, "non_af_count": 0}
                per_lead[li]["total"] += 1
                if r["pred"] == 1:
                    per_lead[li]["af_count"]     += 1
                else:
                    per_lead[li]["non_af_count"] += 1
            self._per_lead_stats = per_lead

            self.after(0, lambda: self._update_ui_after_analysis(
                n_segs, results))
        except Exception as e:
            err = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror(
                "Analysis Error", f"{e}\n\n{err[:600]}"))
            self.after(0, self._reset_run_button)
            self._set_status("Analysis failed.")

    def _update_ui_after_analysis(self, n_segs, results):
        self.progress_var.set(100)
        self._run_btn.configure(state="normal", text="▶   Run Analysis")

        self._update_lead_filter_options()
        self._repopulate_tree()

        total_af = sum(1 for r in results if r["pred"] == 1)
        overall  = total_af > 0
        af_pct   = total_af / len(results) * 100 if results else 0

        card_bg  = C["af_red_dim"]    if overall else C["ok_green_dim"]
        card_bdr = C["af_red_border"] if overall else C["ok_green_border"]
        res_fg   = C["af_red"]        if overall else C["ok_green"]

        self._result_card.configure(
            bg=card_bg, highlightbackground=card_bdr)
        self._result_lbl.configure(
            text="AF DETECTED" if overall else "NON-AF",
            fg=res_fg, bg=card_bg)
        self._conf_lbl.configure(
            text=f"{af_pct:.1f}% of lead-segments flagged AF",
            fg=C["text"], bg=card_bg)
        self._seg_info_lbl.configure(
            text=(f"Total segments: {n_segs}  ·  Leads: {self._n_leads}\n"
                  f"AF: {total_af}  ·  Non-AF: {len(results)-total_af}"),
            fg=C["text_dim"], bg=card_bg)

        dur_h = n_segs * 30 / 3600
        self._summary_lbl.configure(
            text=(f"Recording duration: ≈{dur_h:.2f} hours\n"
                  f"Segments analysed: {n_segs} (30 seconds each)\n"
                  f"Lead-agnostic segments: {len(results)}\n"
                  f"AF segments: {total_af}\n"
                  f"Non-AF segments: {len(results)-total_af}"),
            fg=C["text"])

        rows = [f"{'Lead':<10}{'Total':>6}{'AF':>6}{'Non-AF':>8}{'%AF':>7}"]
        rows.append("─" * 37)
        for li, name in enumerate(self._lead_names):
            s   = self._per_lead_stats.get(
                li, {"total": 0, "af_count": 0, "non_af_count": 0})
            pct = (s["af_count"] / s["total"] * 100) if s["total"] > 0 else 0.0
            rows.append(
                f"{name:<10}{s['total']:>6}{s['af_count']:>6}"
                f"{s['non_af_count']:>8}{pct:>6.1f}%")
        self._lead_table_lbl.configure(
            text="\n".join(rows), fg=C["text"])

        self._gradcam_lbl.configure(
            text="Select a segment from the left panel\n"
                 "to generate the Grad-CAM heatmap.",
            fg=C["text_faint"])
        self._report_btn.configure(state="normal")

        elapsed = time.time() - getattr(self, "_analysis_start", time.time())
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            m, s = divmod(int(elapsed), 60)
            time_str = f"{m}m {s}s"

        self._set_status(
            f"Analysis complete — {n_segs} segments, "
            f"{total_af} AF ({af_pct:.1f}%)  ·  "
            f"Select a segment to view Grad-CAM.  ·  ⏱ {time_str}")

        # Scroll the right panel back to the top after new results load
        self._right_scroll.scroll_to_top()

    def _reset_run_button(self):
        self._run_btn.configure(
            state="normal" if (
                self.ecg_filepath.get() and self._model) else "disabled",
            text="▶   Run Analysis")

    # ─────────────────────────────────────────────────────────────
    # SEGMENT SELECTION → GRAD-CAM
    # ─────────────────────────────────────────────────────────────
    def _on_segment_selected(self, event):
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        items = list(self._tree.get_children())
        try:
            tree_idx = items.index(iid)
        except ValueError:
            return

        filtered = getattr(self, "_filtered_results", self._results)
        if tree_idx >= len(filtered):
            return

        result = filtered[tree_idx]
        try:
            global_idx = self._results.index(result)
        except ValueError:
            global_idx = tree_idx

        if global_idx == self._selected_idx:
            return
        self._select_result_by_index_internal(result, global_idx, iid)

    def _select_result_by_index_internal(self, result, global_idx, iid=None):
        self._selected_result = result
        self._selected_idx    = global_idx

        self._tree.unbind("<<TreeviewSelect>>")
        if iid is None:
            filtered = getattr(self, "_filtered_results", self._results)
            try:
                fi = filtered.index(result)
                items = list(self._tree.get_children())
                iid = items[fi] if fi < len(items) else None
            except ValueError:
                iid = None
        if iid:
            self._tree.selection_set(iid)
            self._tree.see(iid)
        self._tree.bind("<<TreeviewSelect>>", self._on_segment_selected)

        self._gradcam_lbl.configure(
            text="Computing Grad-CAM…", fg=C["warn"])
        self._update_result_card(result)

        filtered = getattr(self, "_filtered_results", self._results)
        try:
            fi = filtered.index(result)
        except ValueError:
            fi = 0
        self._update_nav_controls_filtered(fi, len(filtered))

        self._zoom_level  = 1.0
        self._hscroll_pos = 0.0
        self._zoom_label.configure(text="1×")
        self._hscroll.set(0.0, 1.0)

        self._gradcam_queue.put(result)

    def _select_result_by_index(self, idx):
        if not self._results or idx < 0 or idx >= len(self._results):
            return
        result = self._results[idx]
        self._select_result_by_index_internal(result, idx)

    def _gradcam_loop(self):
        while True:
            result = self._gradcam_queue.get()
            while not self._gradcam_queue.empty():
                try:
                    result = self._gradcam_queue.get_nowait()
                except queue.Empty:
                    break
            self._gradcam_worker(result)

    def _update_nav_controls(self, idx):
        n = len(self._results)
        self._prev_seg_btn.configure(
            state="normal" if idx > 0     else "disabled")
        self._next_seg_btn.configure(
            state="normal" if idx < n - 1 else "disabled")
        self._seg_nav_label.configure(
            text=f"{idx + 1} / {n}" if n else "")

    def _update_nav_controls_filtered(self, fi, total):
        self._prev_seg_btn.configure(
            state="normal" if fi > 0         else "disabled")
        self._next_seg_btn.configure(
            state="normal" if fi < total - 1 else "disabled")
        self._seg_nav_label.configure(
            text=f"{fi + 1} / {total}" if total else "")

    def _prev_segment(self):
        filtered = getattr(self, "_filtered_results", self._results)
        if self._selected_result is None:
            return
        try:
            fi = filtered.index(self._selected_result)
        except ValueError:
            return
        if fi > 0:
            r = filtered[fi - 1]
            try:
                gi = self._results.index(r)
            except ValueError:
                gi = fi - 1
            self._select_result_by_index_internal(r, gi)

    def _next_segment(self):
        filtered = getattr(self, "_filtered_results", self._results)
        if self._selected_result is None:
            return
        try:
            fi = filtered.index(self._selected_result)
        except ValueError:
            return
        if fi < len(filtered) - 1:
            r = filtered[fi + 1]
            try:
                gi = self._results.index(r)
            except ValueError:
                gi = fi + 1
            self._select_result_by_index_internal(r, gi)

    def _gradcam_worker(self, result):
        try:
            seg_data = result["segment_data"]
            li       = result["lead_idx"]
            x_lead   = seg_data[:, li:li+1]
            cam, prob = compute_gradcam(
                self._model, self._target_layer, x_lead, self._device)
            self._current_cam = cam
            self.after(0, lambda: self._draw_gradcam(
                x_lead[:, 0], cam, result, prob))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: (
                self._gradcam_lbl.configure(
                    text=f"Grad-CAM error: {err[:120]}",
                    fg=C["af_red"]),
                self._draw_placeholder()))

    def _update_result_card(self, result):
        ln       = (self._lead_names[result["lead_idx"]]
                    if result["lead_idx"] < len(self._lead_names)
                    else f"L{result['lead_idx']}")
        card_bg  = (C["af_red_dim"]    if result["pred"] == 1
                    else C["ok_green_dim"])
        card_bdr = (C["af_red_border"] if result["pred"] == 1
                    else C["ok_green_border"])
        res_fg   = C["af_red"] if result["pred"] == 1 else C["ok_green"]
        self._result_card.configure(
            bg=card_bg, highlightbackground=card_bdr)
        self._result_lbl.configure(
            text="AF DETECTED" if result["pred"] == 1 else "NON-AF",
            fg=res_fg, bg=card_bg)
        self._conf_lbl.configure(
            text=f"P(AF) = {result['prob']:.3f}   ({ln})",
            fg=C["text"], bg=card_bg)
        self._seg_info_lbl.configure(
            text=f"Segment #{result['seg_idx']}  ·  {ln}",
            fg=C["text_dim"], bg=card_bg)

    def _draw_gradcam(self, ecg_signal, cam, result, prob):
        self._plot_ecg    = ecg_signal
        self._plot_cam    = cam
        self._plot_result = result
        self._plot_axes   = []

        lead_name = (self._lead_names[result["lead_idx"]]
                     if result["lead_idx"] < len(self._lead_names)
                     else f"Lead {result['lead_idx']}")
        pred   = "AF" if result["pred"] == 1 else "Non-AF"
        T      = len(ecg_signal)
        time_s = np.arange(T) / TARGET_FS

        self._fig.clear()
        self._fig.patch.set_facecolor(C["plot_bg"])

        gs  = GridSpec(2, 1, figure=self._fig,
                       height_ratios=[3, 1],
                       hspace=0.06,
                       left=0.07, right=0.92,
                       top=0.88, bottom=0.12)
        ax0 = self._fig.add_subplot(gs[0])
        ax1 = self._fig.add_subplot(gs[1], sharex=ax0)
        self._plot_axes = [ax0, ax1]

        for ax in (ax0, ax1):
            ax.set_facecolor(C["plot_bg"])
            for sp in ax.spines.values():
                sp.set_color(C["plot_spine"])
            ax.tick_params(colors=C["text_dim"], labelsize=10)

        col_title = C["af_red"] if result["pred"] == 1 else C["ok_green"]
        self._fig.suptitle(
            f"Segment #{result['seg_idx']}  ·  {lead_name}  ·  "
            f"Pred: {pred}  ·  P(AF) = {prob:.3f}",
            color=col_title, fontsize=12, fontweight="bold", y=0.97)

        ax0.plot(time_s, ecg_signal,
                 color=C["ecg_line"], linewidth=0.75, zorder=3)

        cmap = _gradcam_cmap()
        step = max(1, T // 600)
        for t in range(0, T - step, step):
            intensity = float(cam[t:t+step].mean())
            ax0.axvspan(
                time_s[t], time_s[min(t+step, T-1)],
                ymin=0, ymax=1,
                alpha=0.6 * intensity + 0.02,
                color=cmap(intensity), zorder=1)

        sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = self._fig.colorbar(sm, ax=ax0, orientation="vertical",
                                  fraction=0.025, pad=0.01)
        cbar.set_label("GradCAM importance",
                       color=C["text_dim"], fontsize=10)
        cbar.ax.yaxis.set_tick_params(
            color=C["text_dim"], labelsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(),
                 color=C["text_dim"])

        ax0.set_ylabel("Amplitude (normalised)",
                       color=C["text_dim"], fontsize=11)
        ax0.set_xlim(0, time_s[-1])
        ax0.tick_params(labelbottom=False)
        ax0.grid(axis="x", color=C["plot_grid"],
                 linestyle="--", linewidth=0.5)

        ax1.fill_between(time_s, cam,
                         alpha=0.65, color=C["cam_fill"])
        ax1.plot(time_s, cam,
                 color=C["cam_line"], linewidth=0.9)
        ax1.axhline(0.5, linestyle="--",
                    color=C["text_faint"], linewidth=0.9)
        ax1.set_xlabel("Time (s)", color=C["text_dim"], fontsize=11)
        ax1.set_ylabel("Activation",  color=C["text_dim"], fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, time_s[-1])
        ax1.grid(axis="x", color=C["plot_grid"],
                 linestyle="--", linewidth=0.5)

        self._canvas.draw()
        self._hscroll.set(0.0, 1.0)

        self._gradcam_lbl.configure(
            text=f"Grad-CAM  ·  Seg #{result['seg_idx']}  ·  {lead_name}\n"
                 "Red = high model attention.",
            fg=C["ok_green"])

    # ─────────────────────────────────────────────────────────────
    # PDF REPORT
    # ─────────────────────────────────────────────────────────────
    def _on_generate_report(self):
        if not self._results:
            messagebox.showinfo("No Data", "Run analysis first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            title="Save Report")
        if not save_path:
            return
        try:
            self._set_status("Generating PDF report…")
            generate_pdf_report(
                save_path,
                filename=Path(self.ecg_filepath.get()).name,
                lead_names=self._lead_names,
                per_lead_stats=self._per_lead_stats,
                total_segments=len(self._segments),
            )
            self._set_status(f"Report saved → {save_path}")
            messagebox.showinfo("Report Saved",
                                f"PDF report saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Report Error", str(e))
            self._set_status("Report generation failed.")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AFiNETApp()
    app.mainloop()