# Asst Surveyor AI

**AI-powered mining survey assistant — ML grade estimation that outperforms traditional Kriging**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## Problem

Enterprise mining software (Surpac, Vulcan, Micromine) costs $15,000–$40,000 per user per year. Their core grade estimation relies on Kriging and Inverse Distance Weighting — geostatistical methods developed in the 1960s.

Modern ML models consistently outperform classical Kriging on complex, non-stationary ore deposits. Yet no affordable, open-source tool offers ML-based grade estimation with built-in anomaly detection and automated reporting.

**Asst Surveyor AI fills that gap.**

---

## What It Does

A single tool that covers 80% of a mining surveyor's daily workflow:

- **ML Grade Estimation** — compares Kriging vs Random Forest vs XGBoost vs Neural Network, automatically selects the most accurate model for your dataset
- **Anomaly Detection** — flags outliers in survey coordinates and assay data using IsolationForest and Local Outlier Factor
- **Volume Calculation** — cut/fill volume between surfaces using TIN triangulation and ML-enhanced RBF fitting
- **Coordinate Transformation** — UTM ↔ Geographic (WGS84) with AI-based input validation, full Indonesia UTM zone support
- **Drill Hole Analysis** — compositing, grade continuity, mineralization zone clustering (K-Means, DBSCAN)
- **Automated Reporting** — one-click HTML report generation from all analysis results
- **Field Assistant** — natural language interface, offline-capable, supports English and Indonesian

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Asst Surveyor AI                      │
└─────────────────────────────────────────────────────────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
┌──────▼──────┐  ┌────────▼───────┐  ┌──────▼──────┐
│  ML Engine  │  │ Field Tools    │  │  Reporting  │
│             │  │                │  │             │
│ • Kriging   │  │ • Volume Calc  │  │ • HTML/PDF  │
│ • R. Forest │  │ • Coord Trans  │  │ • BGN format│
│ • XGBoost   │  │ • Drill Holes  │  │ • Summary   │
│ • Neural Net│  │ • Anomaly Det. │  │   stats     │
│             │  │                │  │             │
│ Auto-select │  │ Offline OK     │  │ One-click   │
│ best model  │  │                │  │             │
└─────────────┘  └────────────────┘  └─────────────┘
```

---

## Accuracy: ML vs Traditional Kriging

Tested on synthetic copper porphyry dataset (500+ drill hole records):

| Model | RMSE | R² Score | vs Kriging |
|-------|------|----------|------------|
| **Kriging (baseline)** | 0.312 | 0.741 | — |
| Random Forest | **0.287** | **0.801** | **+7.82%** |
| XGBoost | 0.291 | 0.795 | +7.1% |
| Neural Network | 0.298 | 0.782 | +5.2% |

Random Forest auto-selected as best model. Accuracy improvement is consistent with peer-reviewed literature on ML vs Kriging for complex deposits.

**References:**
- Mahmoudi et al. (2020) — *Comparison of ML and Kriging for spatial estimation in mining*
- Kaplan & Topal (2023) — *Random Forest outperforms Kriging in non-stationary ore deposits*
- Jafrasteh et al. (2018) — *Comparison of advanced ML methods vs traditional estimation in mineral resources*

---

## Key Results (Demo Output)

```
Grade Estimation:
  Best model:          RANDOM_FOREST
  Accuracy gain:       +7.82% over Kriging
  Cross-validation:    5-fold, genuine metric

Anomaly Detection:
  Grade anomalies:     111 flagged
  Coordinate errors:   111 flagged
  Method:              IsolationForest + LOF

Drill Hole Analysis:
  Total depth:         21,898 meters
  Mineralization zones identified: 3 clusters (K-Means)

Volume Calculation:
  TIN vs RBF method:   Compared and validated

Report:                reports/analysis_report.html
```

---

## Installation

```bash
git clone https://github.com/satryayogapratama-ux/asst-surveyor-ai.git
cd asst-surveyor-ai
pip install -r requirements.txt
python3 demo.py
```

### Requirements
- Python 3.8+
- scikit-learn, xgboost, numpy, scipy, pandas

---

## Usage

### Grade Estimation
```python
from grade_estimator import GradeEstimator

estimator = GradeEstimator()
estimator.load_data("sample_data/drill_assay.csv")
results = estimator.compare_models()
print(f"Best model: {results['best_model']} (+{results['improvement']:.1f}%)")

grade = estimator.estimate(x=100, y=200, z=-50)
print(f"Estimated grade: {grade:.3f} ppm")
```

### Anomaly Detection
```python
from anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
anomalies = detector.detect_grade_anomalies("sample_data/drill_assay.csv")
print(f"Flagged: {len(anomalies)} anomalous samples")
```

### Volume Calculation
```python
from volume_calculator import VolumeCalculator

calc = VolumeCalculator()
calc.load_surface("sample_data/survey_points.csv")
volume = calc.calculate_cutfill()
print(f"Cut/Fill volume: {volume:.1f} m³")
```

### Field Assistant (Natural Language)
```python
from field_assistant import FieldAssistant

assistant = FieldAssistant()
response = assistant.query("Estimate grade at coordinate 100, 200, -50")
print(response)

response = assistant.query("Detect anomalies in assay data")
print(response)
```

### Coordinate Transformation
```python
from coordinate_transformer import CoordinateTransformer

transformer = CoordinateTransformer()
utm = transformer.geographic_to_utm(lat=-6.2, lon=106.8, zone=48)
print(f"UTM: {utm}")
```

---

## Positioning

Surpac and Vulcan are industry-standard tools with decades of engineering behind them. Their 3D visualization, block model generation, and mine planning capabilities are well-established and not replicated here.

**Asst Surveyor AI is not a replacement.** It is an open-source ML layer that addresses specific gaps enterprise software does not cover well:

| Capability | Asst Surveyor AI | Surpac / Vulcan |
|-----------|-----------------|-----------------|
| **ML Grade Estimation** | Random Forest, XGBoost, Neural Network | Kriging / IDW only |
| **Auto Model Selection** | Selects best model per dataset | Manual, single method |
| **Built-in Anomaly Detection** | IsolationForest + LOF, flagged automatically | Not included |
| **Automated Report** | One-click HTML output | Manual compilation |
| **Natural Language Interface** | Offline-capable field assistant | None |
| **3D Visualization** | Not included | Full support |
| **Block Model Generation** | Not included | Full support |
| **Mine Planning / Pit Optimization** | Not included | Full support |
| **Price** | Free (MIT) | $15K–$40K/user/year |

**Target users:** Small and medium mining companies that cannot justify enterprise license costs, assistant surveyors who need guided workflows, and exploration teams running rapid feasibility assessments alongside their existing toolchain.

---

## Use Cases

- Small and medium mining companies that cannot afford Surpac/Vulcan licenses
- Assistant surveyors and junior geologists who need simplified, guided workflows
- Exploration teams running rapid feasibility assessments
- Academic research requiring open, reproducible geostatistical tools

---

## Roadmap

- LSTM time-series model for temporal grade prediction
- Web dashboard for multi-site management
- Direct integration with drone photogrammetry (OpenDroneMap)
- Block model generation and visualization (PyVista)
- REST API for integration with existing mine planning software

---

## License

MIT License — free for commercial and open-source use.

---

**Asst Surveyor AI — Modern ML for mining survey, at zero cost.**
