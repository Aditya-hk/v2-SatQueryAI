# SatQuery AI — Agentic Multi-Modal Remote Sensing Framework

**SIH Problem Statement 167** · Interactive agentic vision-language assistant for single-image,
bi-temporal and cross-modal (optical + SAR) satellite imagery analysis through natural-language
queries.

---

## ✨ What it does

| Input | Query example | What happens |
|---|---|---|
| **Single image** (optical/multispectral or SAR) | *"Describe the land cover and major objects visible in this image."* | VQA + land-cover captioning |
| **Single image** | *"Highlight the water body referred to in the query."* | Text-guided region grounding with bounding boxes |
| **Bi-temporal pair** (T1 + T2) | *"What changed between these two dates, and where?"* | Typed spatial change map + change description + change-VQA |
| **Cross-modal pair** (optical + SAR) | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | Joint spectral + structural feature extraction with per-class fusion |

Every run is orchestrated by an **agentic controller** that validates inputs, classifies intent,
routes to specialist tools from the **Specialist Model Registry**, estimates a composite
**confidence score (0–1)**, and emits an **auditable execution summary** with the full trace.

### 🛰️ Live map mode — draw an ROI, pull real satellite imagery

Switch the sidebar to **"Live map — draw ROI & fetch"** to:

* draw a region of interest directly on an interactive world map (rectangle tool),
* pick T1/T2 dates and toggle optical / SAR sensors,
* fetch **real Sentinel-2 L2A** (optical) and **Sentinel-1 GRD** (SAR) scenes for exactly that
  footprint from the Microsoft Planetary Computer STAC catalog (signed COG windowed reads —
  only the ROI is downloaded), and
* export results as **GeoTIFF** and **GeoJSON** alongside the JSON/PDF reports.

Every acquisition is recorded in the audit trail (source, platform, scene id, cloud cover);
if the catalog is unreachable the app degrades to a deterministic synthetic scene rendered on
the requested extent and says so honestly.

---

## 🚀 Quickstart

```bash
# 1. create an environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the app
streamlit run satquery/app/main_gui.py
```

The sidebar offers four **synthetic georeferenced demo scenes** (EPSG:4326 GeoTIFFs generated
on the fly: single optical, single SAR, bi-temporal pair, optical+SAR pair) so the full
end-to-end flow can be demonstrated offline. Upload your own GeoTIFF/TIFF (or benchmark
PNG/JPEG) via the uploader to analyse real imagery.

## 🧪 Tests

```bash
pytest -q
```

The suite covers demo data integrity, intent routing, CRS/pair validation, all four mandatory
workflows, controller orchestration, confidence estimation, JSON/PDF reports, GeoTIFF
round-tripping, and the LoRA training loop.

## 🎓 Remote-sensing adaptation (BigEarthNet LoRA)

The shipped checkpoint in `satquery/models/checkpoints/` was **trained on real BigEarthNet
Sentinel-2 patches** (BEN-14K cross-modal subset, 10,428 cloud/snow-filtered patches,
canonical 19-class labels):

| epoch | train loss | val loss | val acc@3 |
|-------|-----------|----------|-----------|
| 1     | 0.3016    | 0.2693   | 0.9557    |
| 5     | 0.2299    | 0.2291   | **0.9820** |

Reproduce from scratch:

```bash
# 1. Download the BEN-14K subset (paired S1+S2, ~3.1 GB) + label metadata
curl -L -o data/BigEarthNet_14K.zip "https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet/resolve/main/BEN_14K.zip"
curl -L -o data/BEN14K_metadata.parquet "https://huggingface.co/datasets/torchgeo/bigearthnet/resolve/main/metadata.parquet"

# 2. Materialize into the trainer layout (band-select B02/B03/B04/B08 + canonical labels)
python scripts/materialize_ben14k.py

# 3. LoRA fine-tune (rank 8, alpha 16, MPS/CUDA auto)
python -m satquery.training.fine_tune_bigearthnet \
    --bigearthnet-root data/BEN14K_materialized --epochs 5 --batch-size 32

# smoke run on synthetic BigEarthNet-style data (no download needed)
python -m satquery.training.fine_tune_bigearthnet --epochs 2 --max-samples 256
```

The trainer injects **LoRA adapters (rank 8)** into all 12 linear blocks of the compact
ViT-style encoder, freezes the trunk, trains only adapters + the multi-label head, and saves
`satquery/models/checkpoints/satquery_encoder_ben_lora.pt`. The inference encoder
**auto-detects and merges this checkpoint** at startup — the adaptation loop is fully closed,
and the audit trail reports the adapter status per run
(e.g. `BigEarthNet LoRA adapter merged from satquery_encoder_ben_lora.pt (53/78 tensors; 12 adapters)`).

## 🏗️ Architecture

```
satquery/
├── config.py                      # thresholds, palettes, AppConfig
├── utils/
│   ├── geospatial.py              # GeoTIFF/TIFF/PNG/JPEG IO, CRS validation, pair checks,
│   │                              #   modality inference, reprojection, spectral indices
│   ├── logger.py                  # auditable ExecutionSummary + trace events
│   ├── report.py                  # JSON payload + dependency-free PDF writer
│   └── demo_data.py               # deterministic synthetic demo scenes
├── models/
│   ├── registry.py                # Specialist Model Registry (lazy tool factories)
│   ├── single_image_vqa.py        # RS encoder (torch) + VQA / captioning / grounding
│   ├── change_detection.py        # index differencing + ChangeNetHead, change-VQA
│   └── cross_modal_fusion.py      # early/late fusion, complementarity analysis
├── agent/
│   ├── intent_classifier.py       # NL query → task + sub-capability + parameters
│   └── controller.py              # orchestration + composite confidence estimator
├── app/
│   └── main_gui.py                # Streamlit GUI (maps, confidence, trace, reports)
└── training/
    └── fine_tune_bigearthnet.py   # BigEarthNet LoRA fine-tuning + CLI
```

**Execution flow:** `validate → classify intent → fallback-guard → registry.select_tools →
lazy tool instantiation → execute → estimate confidence → auditable summary → GUI + reports`

### Confidence model

`overall = 0.30·data_quality + 0.45·model + 0.25·agreement`

* **data_quality** — CRS validity, pair compatibility, modality certainty, cloud cover
* **model** — mean specialist-tool confidence
* **agreement** — cross-view consistency (optical↔SAR Jaccard agreement, or change-label
  consistency); a clarity-modulated prior for single-view runs

## 📦 Outputs per run

* Natural-language answer with confidence badge and component breakdown
* Side-by-side original vs. change map / grounding overlay / fused classification (+ per-class maps)
* Auditable execution summary (task, tools, parameters, timings, warnings)
* Downloadable **JSON** and **PDF** reports (`outputs/` also receives run artefacts)
* In live-map mode: an **interactive result map** with ROI + T1/T2/grounding/change overlays,
  plus **GeoTIFF** (layers) and **GeoJSON** (grounded regions) downloads

## 🗺️ Supported formats

* **GeoTIFF/TIFF** — full geospatial path (CRS validation, co-registration checks, reprojection)
* **PNG/JPEG** — accepted for prescribed benchmark datasets (VRSBench / RSVQA / CDVQA)
* Bi-temporal and cross-modal pairs must share dimensions; CRS mismatches are auto-reprojected
  (toggleable) with warnings recorded in the trace.

## ⚠️ Honest engineering notes

* The RS encoder is a compact, deterministic ViT-style network trained *via the included LoRA
  pipeline*; it is not a multi-billion-parameter foundation model. Answers are grounded in
  interpretable spectral/structural evidence, and confidence reflects that honestly.
* With no BigEarthNet checkpoint present the system runs in deterministic analytic mode and
  says so in the audit trail; run the trainer (above) to activate the adapted encoder.
* Grounding produces bounding-box regions from class segmentation; masks are available from
  the same maps if the evaluation harness requires polygon output.

## 📄 License / attribution

Built for Smart India Hackathon 2026, Problem Statement 167 (ISRO/SAC).
Datasets referenced: BigEarthNet, VRSBench, RSVQA, CDVQA.
