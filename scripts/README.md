# Scripts Directory

## Active Scripts

### `perception_pipeline_v2.py`

**Purpose:** Main perception layer pipeline execution script

**Function:** Generates all perception data (terrain, wind, threat, obstacle, fusion) and exports to CSV/JSON

**Usage:** `python perception_pipeline_v2.py`

**Output:** Saves all data to `data/1_derived/perception_outputs/run_YYYYMMDD_HHMMSS/`

### `setup_env.py`

**Purpose:** Environment setup and validation

**Function:** Checks Python version, installs dependencies, validates imports

**Usage:** `python setup_env.py`

## Archive Scripts

All legacy/deprecated scripts have been moved to `archive/` folder:

- `repro_main.py` - Old main reproduction script
- `run_experiment.py` - Deprecated experiment runner
- `run_full_perception_pipeline.py` - Earlier version of perception pipeline
- `run_perception_pipeline.py` - Original perception pipeline (replaced by v2)
- `setup_mlflow.py` - MLflow setup (currently unused)

These are kept for reference but should not be used for new work.

## Recommended Workflow

```bash
# 1. Run perception pipeline (generates all perception data)
python perception_pipeline_v2.py

# 2. Data will be saved to:
# data/1_derived/perception_outputs/run_YYYYMMDD_HHMMSS/
#   ├── datasets/          (CSV maps)
#   ├── stats/             (JSON statistics)
#   └── models/            (serialized model objects)
```
