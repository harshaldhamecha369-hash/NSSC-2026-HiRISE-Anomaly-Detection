# NSSC 2026 — Unsupervised Anomaly Detection in Mars HiRISE Imagery

**Phase 1.1 — Convolutional Autoencoder: Architecture Design**

This repository implements Phase 1.1 of the NSSC 2026 project: a Convolutional
Autoencoder (CAE), built entirely from scratch (no pretrained weights, no
transfer learning, no pretrained backbones), that compresses 227×227 grayscale
Mars HiRISE crops into a fixed-length 1D latent vector. That latent vector is
the hand-off artifact for Phase 2, where an Isolation Forest will use it to
score images for novelty ("Genesis Outlier" candidates).

```
227×227 grayscale image → Conv Autoencoder → [batch, latent_dim] vector → (Phase 2: Isolation Forest) → Novelty Score
```

> **Scope of this repo:** only Phase 1.1 (architecture design, training loop,
> reconstruction sanity checks, latent extraction) is implemented. Phase 2
> (Isolation Forest) and Phase 3 (anomaly heatmaps) are referenced only as a
> commented preview inside the notebook — nothing from those phases is run.

---

## Why a CAE and not a VAE?

A VAE's stochastic latent space (sampling + KL regularization) is mainly
useful when you want a smooth, generative latent space to sample from. For
anomaly detection via reconstruction error / a downstream Isolation Forest on
the latent vectors, a deterministic CAE bottleneck is simpler, easier to
reason about, and produces a more literal "compressed fingerprint" of each
image — exactly what Isolation Forest needs to score outliers on.

## Why no labels are needed

An autoencoder's training signal is the input image itself: `loss =
distance(x, decoder(encoder(x)))`. There's no notion of a "correct class"
anywhere in that loss, so labels are irrelevant. This is what makes the
approach unsupervised — outlier ("Genesis Outlier") images don't need to be
identified or excluded beforehand. Normal terrain vastly outnumbers outliers,
so the network learns the dominant statistical pattern of normal Mars
terrain, and outliers naturally reconstruct worse and/or sit differently in
latent space. Phase 2's Isolation Forest exploits that gap.

---

## Architecture at a glance

| Choice | Value | Why |
|---|---|---|
| Kernel size | 4 | Pairs cleanly with stride 2 for predictable up/downsampling |
| Stride | 2 | 5 stages take 227×227 down to a compact 7×7 |
| Padding | 1 | Keeps the output-size formula well-behaved |
| Encoder channels | 1→32→64→128→256→512 | Doubles each stage as spatial size shrinks |
| Activation | LeakyReLU(0.2) | Avoids dead ReLU units |
| Normalization | BatchNorm2d after each conv (except output) | Stabilizes/speeds up training |
| Bottleneck | Flatten(512×7×7=25,088) → Linear(25088, latent_dim) | Fixed-length 1D latent vector |
| Decoder upsampling | ConvTranspose2d mirroring the encoder | Learned upsampling, exact spatial inverse |
| Output activation | Sigmoid | Matches input range [0, 1] |

227 is not a power of two, so naive "halve it 5 times" reasoning breaks down.
Every stage's output size is derived explicitly from PyTorch's conv output
formula (`out = floor((in + 2*padding - kernel) / stride) + 1`) before any
`nn.Module` is written, and `output_padding` corrects the two decoder stages
where 227's odd size causes rounding truncation. Full derivation, the
layer-by-layer table, and a text architecture diagram are in the notebook
(§3–§4).

`latent_dim` defaults to `256` and is a constructor argument
(`ConvAutoencoder(latent_dim=...)`), so comparing 128 / 256 / 512 only
requires re-instantiating the model — see §10 of the notebook.

---

## Repository structure

```
.
├── notebooks/
│   └── phase1_1_conv_autoencoder.ipynb   # Phase 1.1 notebook (this repo's contents)
├── outputs/                              # generated artifacts land here (git-ignored)
│   └── latent_vectors.npy                # produced by the notebook, consumed by Phase 2
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

### 1. Environment

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required — the notebook
auto-selects `cuda` if available and falls back to `cpu` otherwise.

### 2. Dataset

Point `DATA_DIR` (top of §2 in the notebook) at a folder of grayscale HiRISE
crops. Any of `.png / .jpg / .jpeg / .tif / .tiff` are picked up
automatically; images that aren't already 227×227 are resized. No label
file or folder structure is required — this is unsupervised.

```python
DATA_DIR = "/path/to/your/images"
```

### 3. Run the notebook

```bash
jupyter lab notebooks/phase1_1_conv_autoencoder.ipynb
```

Run cells top-to-bottom. The notebook:

1. Loads and sanity-checks the dataset (§2)
2. Derives and verifies the encoder/decoder spatial dimensions for 227×227 (§3)
3. Defines `ConvAutoencoder` with `encode()` / `decode()` / `forward()` (§4)
4. Defines the reconstruction loss (§5)
5. Trains with a full train/val loop and plots loss curves (§6)
6. Visualizes original vs. reconstructed images (§7)
7. Extracts a `[num_images, latent_dim]` latent matrix for the whole dataset
   and saves it to `latent_vectors.npy` — **the only artifact Phase 2 needs**
   (§8)
8. Prints a model diagnostics summary (§9)
9. Documents the `latent_dim` experimentation pattern (§10) and the planned
   v1 → v2 → v3 architecture iteration (§11)

### 4. Hand-off to Phase 2

```python
# Phase 2 preview — not implemented in this repo
from sklearn.ensemble import IsolationForest
import numpy as np

latents = np.load("outputs/latent_vectors.npy")   # [num_images, latent_dim]
iso_forest = IsolationForest(contamination="auto", random_state=42)
iso_forest.fit(latents)
novelty_scores = iso_forest.decision_function(latents)
```

---

## Choosing `latent_dim` (don't use validation loss alone)

- **Reconstruction quality** — does a smaller `latent_dim` visibly blur out
  small craters/ridgelines that a larger one preserves?
- **Validation loss** — necessary but not sufficient; lower isn't
  automatically better if it comes from memorizing per-image detail.
- **Preservation of fine structure** — does the reconstruction keep
  high-frequency terrain detail or look "smoothed"?
- **Downstream usefulness** — the real test (Phase 2) is whether Isolation
  Forest on the resulting latent space actually separates plausible outlier
  candidates from normal terrain.

## Architecture iteration plan (v1 → v2 → v3)

| Version | Status | Trigger / Fix |
|---|---|---|
| **v1** | This repo | 5-stage strided-conv encoder, mirrored decoder, BatchNorm + LeakyReLU, `latent_dim=256`, plain MSE loss. Establishes a correct working baseline end-to-end. |
| **v2** | Planned | If reconstructions look blurred / small craters lost / val loss plateaus early → add an encoder conv block and/or increase `base_channels` and/or try `latent_dim=512`. |
| **v3** | Planned | If v2 improves detail but overfits / train-val loss diverge → add dropout at the bottleneck and/or move to an MSE+SSIM loss (Phase 1.2). |

Full symptom → diagnosis → fix reasoning is in §11 of the notebook — these
are documented as a plan, not fabricated results; each step gets filled in
with the actual numbers once training runs on the real dataset.

## Requirements traceability

The notebook's final section (§12) contains a full checklist mapping every
Phase 1.1 requirement to the exact section/cell that satisfies it (from
scratch-only weights, CAE-vs-VAE justification, shape assertions at every
stage, layer table, loss curves, latent extraction, diagnostics, and the
v1/v2/v3 plan).

## Status / Roadmap

- [x] Phase 1.1 — Convolutional Autoencoder architecture, training, latent extraction
- [ ] Phase 1.2 — Loss function refinement (MSE + SSIM)
- [ ] Phase 2 — Isolation Forest on latent vectors
- [ ] Phase 3 — Anomaly heatmaps
- [ ] Phase 4 — Engineering changelog (v1 → v2 → v3, with real results)

## License

Distributed under the MIT License — see [`LICENSE`](LICENSE).
