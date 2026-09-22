
# CrimeVision-AI

> **Multimodal Acoustic-to-Craniofacial Synthesis & Automated Case Correlation Pipeline**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Interface-Streamlit%20%7C%20Plotly-00FF66.svg)](https://streamlit.io/)
[![Engine](https://img.shields.io/badge/Generative%20AI-Stable%20Diffusion%20v1.5-orange.svg)](https://huggingface.co/sd-legacy/stable-diffusion-v1-5)
[![Biometrics](https://img.shields.io/badge/Biometrics-RetinaFace%20%2B%20FaceNet%20(512--D)-00F0FF.svg)](https://github.com/serengil/deepface)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Summary

Traditional forensic composite generation suffers from three operational bottlenecks:
1. **Artist Availability Latency:** Hand-drawn composites require specialized forensic artists, causing delays of hours or days during critical early investigation windows.
2. **Descriptive & Dialect Friction:** Witnesses routinely describe suspects in regional languages (e.g., Hindi, Punjabi)[cite: 3]. Translation loss during manual intake distorts physical descriptors.
3. **Manual Record Correlation:** Physical sketches cannot be programmatically queried against legacy case databases or first information reports (FIRs) without manual visual cross-referencing[cite: 2, 3].

**CrimeVision-AI** implements an end-to-end local multimodal pipeline:
$$\text{Witness Voice} \xrightarrow{\text{Whisper ASR}} \text{Transcription} \xrightarrow{\text{Translation}} \text{Forensic Prompt} \xrightarrow{\text{Diffusion}} \text{Composite} \xrightarrow{\text{FaceNet 512-D}} \text{Cosine Ranking}$$

The system transcribes spoken regional testimony, structures it into forensic rendering prompts, generates latent diffusion composites, and correlates them against historical FIR archives using 512-dimensional metric embedding comparisons[cite: 2, 3].

---

## End-to-End System Pipeline

```text
       [ Microphone Stream / Intercept Audio (44.1 kHz WAV) ]
                                 │
                                 ▼
         [ Stage 01: Multi-lingual ASR — OpenAI Whisper ]
                 (Decodes Hindi, Punjabi, English speech)
                                 │
                                 ▼
            [ Text Normalization — Deep Translator ]
              (Standardizes freeform dialect into EN)
                                 │
                                 ▼
           [ Rule-Conditioned Forensic Prompt Compiler ]
         (Injects front-facing sketch & lighting tokens)
                                 │
                                 ▼
     [ Stage 02: Latent Diffusion Synthesis — SD v1.5 ]
       (Attention slicing enabled; CPU-quantized float32)
                                 │
                                 ▼
       [ Cryptographic Tamper Seal — SHA-256 Hash ]
                                 │
                                 ▼
       [ Stage 03: Feature Extraction — RetinaFace + FaceNet ]
             (Extracts L2-normalized 512-D vector)
                                 │
                                 ▼
         [ Vector Similarity Search — Cosine Distance ]
           (Compares against synthetic database.json)
                                 │
                                 ▼
       [ Stage 04: Command Center & Analytics Dashboard ]
           (Streamlit + Plotly Multi-Accent Telemetry)

```

---

## Architecture & Engineering Specifications

| Layer | Implementation | Engineering Rationale |
| --- | --- | --- |
| **Acoustic Ingestion** | `sounddevice` (44.1 kHz, 16-bit Mono) | Captures low-noise voice samples directly from hardware microphones into uncompressed linear PCM.

 |
| **Speech-to-Text (ASR)** | OpenAI Whisper (`base` model) | Native zero-shot multilingual decoding robust against background noise and non-standard accents.

 |
| **Linguistic Normalization** | Deep Translator (`GoogleTranslator` engine) | Eliminates semantic ambiguity prior to latent sampling by normalizing regional syntax to English.

 |
| **Prompt Synthesis** | Token-Conditioned Prefix Injection | Appends forensic-grade negative/positive styles (`police forensic sketch`, `FBI sketch style`, `front face`) to witness descriptions.

 |
| **Generative Vision** | Stable Diffusion v1.5 (`diffusers`) | Cross-attention latent reverse diffusion; eliminates the need for scarce paired sketch-photo datasets.

 |
| **Low-Memory Optimization** | Attention Slicing (`enable_attention_slicing`) | Slices attention computation across sub-matrices, preventing out-of-memory (OOM) faults on 8 GB RAM machines.

 |
| **Facial Landmark Backend** | RetinaFace Detector | Handles stylized pencil textures and synthetic faces where traditional Haar cascades or HOG detectors fail.

 |
| **Metric Embeddings** | FaceNet (512-dimensional vector space) | Maps facial topology into a compact Euclidean hypersphere where distance directly tracks identity similarity.

 |
| **Record Retrieval** | Vector Cosine Distance Scoring | Calculates vector alignment: $S_C(u,v) = \frac{u \cdot v}{\Vert{}u\Vert{}_2 \Vert{}v\Vert{}_2}$. Parametrically filters matches $\ge 0.70$.

 |
| **Telemetry & UI** | Streamlit + Plotly (WebGL dark tactical theme) | Eliminates React/Node build dependencies; provides sub-second interactive chart hover tooltips and dynamic 3D canvas rendering.

 |

---

## Metric Search Mathematics

Given query face embedding $\mathbf{q} \in \mathbb{R}^{512}$ and stored case embedding $\mathbf{k}_i \in \mathbb{R}^{512}$:

$$\text{Similarity}(\mathbf{q}, \mathbf{k}_i) = \frac{\sum_{j=1}^{512} q_j \cdot k_{i,j}}{\sqrt{\sum_{j=1}^{512} q_j^2} \cdot \sqrt{\sum_{j=1}^{512} k_{i,j}^2}} \times 100\%$$

Matches are ranked descending by confidence. Cases meeting the threshold $\tau \ge 70.0\%$ surface in the investigator triage queue.

---

## Directory Structure

```text
V2F-CrimeVision-AI/
├── dashboard.py           # Tactical Command Center UI & Plotly telemetry
├── pipeline.py            # Core end-to-end pipeline (Audio -> Translation -> SD)
├── prompt_builder.py      # Forensic sketch conditioning & prompt compiler
├── history_lookup.py      # RetinaFace landmarking & FaceNet cosine search
├── mock_database.py       # Simulated FIR case builder & pickle embedding pre-computer
├── assistant.py           # Offline rule-based Q&A architectural assistant
├── requirements.txt       # Explicit runtime dependencies
├── database.json          # Synthetic criminal case metadata registry
├── embeddings.pkl         # Pre-computed 512-D case face embeddings
└── outputs/               # Rendered composites & SHA-256 evidence logs

```

---

## Installation & Setup

### 1. Environment Preparation

```bash
# Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/CrimeVision-AI.git
cd CrimeVision-AI

# Create isolated Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

```

### 2. Dependency Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Initialize Case Embeddings

Generate the pre-computed FaceNet embeddings for the mock FIR case database:

```bash
python mock_database.py

```

### 4. Launch Command Center

```bash
streamlit run dashboard.py

```

Access interface at `http://localhost:8501`.

---

## Technical Dependencies (`requirements.txt`)

```text
streamlit>=1.32.0
plotly>=5.18.0
numpy>=1.24.3,<2.0.0
scipy>=1.10.1
sounddevice>=0.4.6

# Deep Learning & Generative Frameworks
torch>=2.1.0
diffusers>=0.26.0
transformers>=4.38.0
accelerate>=0.27.0

# Speech Recognition & NLP Normalization
openai-whisper>=20231117
deep-translator>=1.11.4

# Biometrics & Facial Feature Extraction
deepface>=0.0.90
retina-face>=0.0.14
tf-keras>=2.15.0
tensorflow>=2.15.0

```

---

## Ethical Boundaries & Forensic Limitations

* **Synthetic Data Boundary:** This project utilizes a mock case archive (`database.json`). It does not interface with live National Crime Information Center (NCIC), Interpol, or state police CCTNS records.


* **No Automated Identification:** AI-generated facial composites are probabilistic approximations derived from witness memory recall. They represent investigative leads, never legal proof of identity or admissible courtroom identification.


* **Biometric Parity Caveats:** Face verification models display known demographic variance across skin tones, lighting conditions, and facial asymmetry.


* **Admissibility Auditing:** Every composite is generated alongside an immutable SHA-256 cryptographic hash to preserve evidentiary chain-of-custody.



---

## Author & Project Credits

**Sanchita Kumari**

*B.E. Computer Science Engineering (Artificial Intelligence & Machine Learning)*

*Chandigarh University*

Architected and developed the complete end-to-end pipeline: multi-lingual speech transcription pipeline, memory-optimized latent diffusion integration, metric face-embedding correlation engine, and tactical telemetry interface.

```

```
