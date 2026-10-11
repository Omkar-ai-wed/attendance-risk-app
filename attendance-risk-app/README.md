# Student Attendance Risk Predictor

**Pune College Attendance Risk Classification · ML Unit 3 · Educational Prototype**

A polished Streamlit application that explores attendance patterns and classifies attendance risk using three scikit-learn models. Features a transparent target definition, leakage-aware evaluation, CSV batch prediction, and a professional dashboard — ready for deployment on Streamlit Community Cloud.

> **Educational use only.** Data are synthetic and the risk label is a simple threshold rule.
> Do not use outputs for consequential decisions about real students without validation,
> institutional policy checks, and human review.

---

## Features

| Tab | What it does |
|---|---|
| 📊 Dashboard | KPI cards, attendance distribution histogram, risk-class breakdown |
| 🧮 Single Prediction | Predict risk for one student with a visual probability bar |
| 📁 Batch Prediction | Upload CSV → validate → predict → download results |
| 🧪 Model Evaluation | Metrics table, confusion matrices, feature importance |
| 📘 Dataset & Methodology | Data provenance, target definition, leakage explanation |

---

## 1. Problem Statement

Given a student's lecture counts (and optional attendance history), classify whether the student is **At Risk** of falling below the required attendance threshold.

**Target (Approach A — implemented):**
```
At Risk     if  lectures_attended / lectures_conducted × 100  <  threshold
Not at Risk if  attendance_pct  >=  threshold
```
The 75% default is an example only. Check your institution's actual rule; the threshold is configurable in the sidebar.

---

## 2. Data Dictionary

| Column | Type | Model input? | Description |
|---|---|---|---|
| `student_id` | text | No | Fictional ID (e.g. STU0001). Never used as a feature. |
| `lectures_conducted` | integer > 0 | Yes (required) | Lectures held so far |
| `lectures_attended` | integer, 0..conducted | Yes (required) | Lectures attended |
| `previous_attendance_pct` | float 0–100 | Yes (optional) | Previous-term attendance %. Median-imputed if blank. |
| `recent_absences` | integer ≥ 0 | Yes (optional) | Lectures missed in the last two weeks. Median-imputed if blank. |
| `risk_label` | At Risk / Not at Risk | Target | Derived from the threshold rule |

---

## 3. Technologies

- **Python 3.12+**
- **Streamlit ≥ 1.32** — UI framework
- **scikit-learn ≥ 1.2** — Logistic Regression, Decision Tree, Random Forest (Pipelines with median imputation)
- **pandas ≥ 2.0** — data handling
- **numpy ≥ 1.24** — numerical operations
- **matplotlib ≥ 3.7** — confusion matrix plots
- **pytest** — test suite (52 tests)

---

## 4. Local Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/attendance-risk-app.git
cd attendance-risk-app

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
# Opens at http://localhost:8501
```

### Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
# Expected: 52 passed
```

### Regenerate synthetic data (optional)

```bash
python -m src.data_utils
# Overwrites data/sample_attendance.csv
```

---

## 5. Required Datasets / Models / Secrets

| Resource | Location | Notes |
|---|---|---|
| Sample data | `data/sample_attendance.csv` | Bundled. Auto-generated in memory if missing. |
| ML models | Trained at app startup | No saved model files — `st.cache_resource` keeps them in memory. |
| API keys / secrets | **None required** | This app has no external services, paid APIs, or credentials. |

---

## 6. Streamlit Community Cloud Deployment

1. Push the repository to GitHub (see `DEPLOYMENT.md` for a step-by-step checklist).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. No secrets or environment variables are needed.

Models are trained once per session and cached with `st.cache_resource`, so reruns are fast.
The bundled CSV means the app works out of the box with no upload required.

---

## 7. Data Leakage and Limitations

- **Leakage:** the label is a deterministic function of `lectures_attended` and `lectures_conducted`. The derived percentage is deliberately **not** passed to the models; they must re-learn the ratio from raw counts. Results are framed as *reproducing the rule* — the rule baseline scores 100% by construction.
- Feature importance shows association with model output, not causation.
- **Limitations:** synthetic data, rule-generated labels, a single train/test split (no cross-validation), simulated optional features, no fairness analysis, institution-specific thresholds.

---

## 8. Ethics and Privacy

Use fictional IDs only; do not upload real student data to a public deployment. Treat outputs as a prompt for a human conversation, never as a verdict. No authentication, database, or external service is used.

---

## 9. Project Structure

```
app.py                      Streamlit UI (5 tabs)
requirements.txt            Runtime dependencies
requirements-dev.txt        Adds pytest
pytest.ini
.streamlit/
  config.toml               Theme + server settings
data/
  sample_attendance.csv     600 synthetic student records
src/
  data_utils.py             Schema, validation, attendance %, synthetic data generation
  labeling.py               Threshold rule / target definition
  modeling.py               Pipelines, split, experiment, prediction, feature importance
  evaluation.py             Metrics, confusion matrices, plots
tests/                      pytest suite (52 tests)
DEPLOYMENT.md               Streamlit Cloud deployment checklist
```

---

## 10. Example Viva Questions

1. What exactly is the target, and why is this classification rather than prediction?
2. Why does the rule baseline score 100%, and what does that imply about the model scores?
3. What is target leakage, and how does it apply here? Why is `attendance_pct` not a feature?
4. Why is `student_id` excluded from the features?
5. Why is the imputer inside the Pipeline and fitted only on training data?
6. Why stratify the split? What would go wrong with a one-class test set?
7. Why report precision and recall for `At Risk` rather than accuracy alone?
8. Which is worse here, a false negative or a false positive?
9. Why might Logistic Regression beat tree models on this data?
10. What would you need to build a genuine Approach B (future-risk) model?
11. Does feature importance show causation? Why not?
12. What ethical safeguards are needed before using such a tool on real students?
