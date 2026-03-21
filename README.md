# Sign Language Web

A Django web application for sign language recognition using WLASL landmark data.

---

## Project structure

```
sign_language_web/
├── config/            # Django project settings
├── translator/        # Main Django app (views, models)
├── models/            # Pre-built model artifacts (labels.npz, filtered_labels.txt)
├── kaggle/            # Kaggle notebooks for landmark extraction
│   └── landmark_extraction.ipynb
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running the web app

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Or with Docker:

```bash
docker-compose up --build
```

---

## Kaggle landmark extraction — multi-session guide

Because extracting landmarks from all ~21 000 WLASL video clips takes longer than a single Kaggle session (~12 h), the notebook [`kaggle/landmark_extraction.ipynb`](kaggle/landmark_extraction.ipynb) is designed to run across **multiple sessions**, each one continuing exactly where the last left off.

### Overview

| Cell | Purpose |
|------|---------|
| **Cell 0** | Global configuration (paths, dataset slugs, time budget) |
| **Cell 1** | Restore progress from previous session |
| **Cell 1b** | Sanity-check restore results |
| **Cell 2** | Extract landmarks (anti-reset resume) |
| **Cell 3** | Build label vectors (cached, skips if already done) |
| **Cell 4** | Pack everything into `landmarks_master.zip` for next session |

---

### Session 1 — starting fresh

1. Open a new Kaggle notebook and upload / link `landmark_extraction.ipynb`.
2. In the **Add data** panel attach the `wlasl-complete` dataset.
3. In **Cell 0** set:
   ```python
   PREV_SESSION_DATASET = ""   # no previous session
   ```
4. Run **Cell 0 → Cell 2** (skip Cell 1 or it will silently do nothing).
5. When the session is close to timing out (or when Cell 2 prints `[STOP] Reached time budget`), run **Cell 4** to produce `landmarks_master.zip`.
6. Download `session_output.zip` from `/kaggle/working` and **create a new Kaggle dataset** from it (e.g. name it `last-hope-s1`).

---

### Session 2+ — resuming progress

1. Start a new Kaggle notebook session with `landmark_extraction.ipynb`.
2. In the **Add data** panel attach **both**:
   - `wlasl-complete` (videos)
   - The previous session's output dataset (e.g. `last-hope-s1`)
3. In **Cell 0** set:
   ```python
   PREV_SESSION_DATASET = "last-hope-s1"   # match the dataset slug exactly
   ```
4. Run **Cell 0** then **Cell 1**.
   - You should see:
     ```
     [RESTORE] Discovered N .npy file(s) under ...
     [RESTORE] npy in working : N      ← should be non-zero
     [RESTORE] manifest exists: True
     ```
5. Run **Cell 1b** to double-check.
6. Run **Cell 2**.  
   At the top you will see:
   ```
   [RESUME] disk_done=N, manifest_done=M, union=K
   [GLOBAL START] K/21083 (XX.XX%)
   ```
   The percentage should match where the previous session left off.
7. Let Cell 2 run until timeout or completion, then run **Cell 4** again and create a new dataset for the next session.

---

### What the restore cell detects

Cell 1 (`restore_progress`) handles two source formats automatically:

| Format | How it's detected |
|--------|-------------------|
| `landmarks_master.zip` | File exists at the top level of the previous dataset |
| Folder structure | `landmarks_npy/*.npy` files present directly in the previous dataset |

It will **raise a `FileNotFoundError`** with a detailed message (including a listing of available datasets) if `PREV_SESSION_DATASET` points to a path that does not exist, so failures are never silent.

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `[RESTORE] npy in working: 0` | Wrong dataset slug or dataset not attached | Check `PREV_SESSION_DATASET` and the Kaggle input panel |
| `FileNotFoundError: Previous session directory not found` | Dataset slug typo | The error message lists available datasets — use one of those |
| `[GLOBAL START] 0/21083 (0.00%)` | Cell 1 ran but found nothing to copy | See above |
| Label encoding runs every session | Normal if `labels.npz` is missing; Cell 3 caches and skips on rerun | Run Cell 4 which bundles meta files into the zip |

---

### Tips

- **Time budget**: `SESSION_SECONDS_LIMIT` (Cell 0) defaults to 11.5 h; `RESERVE_ZIP_SECONDS` reserves 30 min for packing.  
  Adjust these if your session length differs.
- **Speed**: ~3.5–4.5 s/sample → ~9 000–11 000 samples per session.  
  Typically 2–3 sessions suffice for all 21 083 samples.
- **Manifest vs disk**: Cell 2 always builds `done` as `disk_done ∪ manifest_done`.  
  Even if the manifest is stale, any `.npy` file already on disk will be skipped.
