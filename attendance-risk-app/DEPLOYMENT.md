# Deployment Checklist — Streamlit Community Cloud

Reference: https://docs.streamlit.io/deploy/streamlit-community-cloud

---

## Required Files ✅

| File | Status | Notes |
|---|---|---|
| `app.py` | ✅ Required | Entry point for Streamlit Cloud |
| `requirements.txt` | ✅ Required | Runtime dependencies with upper bounds |
| `data/sample_attendance.csv` | ✅ Required | Bundled synthetic dataset (app falls back to in-memory generation if missing) |
| `src/__init__.py` | ✅ Required | Makes `src` a Python package |
| `.streamlit/config.toml` | ✅ Recommended | Theme + server config |
| `.gitignore` | ✅ Recommended | Excludes venv, secrets, caches |

---

## Python & Dependency Requirements

- **Python:** 3.12 recommended (3.10+ compatible)
- **Streamlit:** ≥ 1.32, < 2.0
- No GPU required — all models run on CPU
- No system packages required beyond what Streamlit Cloud provides
- No paid APIs or credentials needed

---

## Secrets & Environment Variables

**None required.** This application uses no external APIs, databases, or credentials.

If you add secrets in future:
1. Go to your app in Streamlit Cloud → **⋮ menu → Edit secrets**
2. Add `KEY = "value"` in TOML format
3. Access in code via `st.secrets["KEY"]`
4. Never commit `secrets.toml` (already in `.gitignore`)

---

## Repository Preparation

```bash
# 1. Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit — Attendance Risk Predictor"

# 2. Push to GitHub
git branch -M main
git remote add origin https://github.com/<your-username>/attendance-risk-app.git
git push -u origin main
```

### Files to exclude from git (already in .gitignore)
- `.venv/` / `venv/` — virtual environment
- `__pycache__/` / `*.pyc` — Python caches
- `.streamlit/secrets.toml` — secrets
- `.env` — local environment files
- `.pytest_cache/` — test cache

---

## Streamlit Community Cloud Setup

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **Create app**
4. Fill in:
   - **Repository:** `<your-username>/attendance-risk-app`
   - **Branch:** `main`
   - **Main file path:** `app.py` (or `attendance-risk-app/app.py` if targeting the subfolder)
   - **Python version:** 3.12
   *(Both directory levels have been synchronized with complete `app.py`, `src/`, `data/`, and `requirements.txt` so either path works seamlessly!)*
5. Click **Deploy**

---

## Startup Behaviour

- Models are trained at first launch using the bundled CSV (`data/sample_attendance.csv`)
- Training takes ~5–15 seconds; Streamlit shows a spinner
- Results are cached with `st.cache_resource` — subsequent reruns are instant
- If the CSV is missing, the app generates synthetic data in memory automatically

---

## Known Limitations & Notes

| Item | Status | Notes |
|---|---|---|
| GPU | Not needed | All scikit-learn models run on CPU |
| Persistent storage | Not used | No files are written at runtime |
| External APIs | None | Fully self-contained |
| Memory | Low | ~150 MB peak (600 students × 3 models) |
| Startup time | ~15s cold start | Model training on first load |
| File uploads | Temporary only | Uploaded CSVs exist only during the session |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | Import path issue | Ensure `src/__init__.py` exists and app is run from repo root |
| `FileNotFoundError: sample_attendance.csv` | Missing data file | The app auto-generates data; check `.gitignore` didn't exclude `data/` |
| `ModelingError: Only one risk class` | Threshold too extreme | Choose a threshold between ~10% and ~95% |
| App slow on first load | Model training | Normal — `st.cache_resource` prevents re-training on rerun |
| Broken imports on Linux | Case-sensitive paths | All imports use lowercase `src` — should be fine |
