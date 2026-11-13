"""Load and validate the YAML run config."""
import argparse
import logging
from pathlib import Path

import yaml

VALID_MODES = {"test", "preview"}
VALID_MODELS = {"kev", "laya"}

# Sample values for testing. Anything set in config.yaml overrides these;
# anything left out of config.yaml falls back to them.
DEFAULTS = {
    "mode": "test",
    "data": {
        "s3_uri": "s3://data-team-shared/jev-research-beta/echo-report-sample/echo_sample_11_11_2026.parquet",  # placeholder
        "text_column": "DOCUMENT_TEXT",
        "id_column": "DH_DOCUMENT_ID",
        "aws_profile": None,  # None = SageMaker execution role / default credentials
    },
    "task": {
        "instructions": (
            "Based on this echocardiogram report, what was the primary reason "
            "the echocardiogram was ordered?"
        ),
        "labels": {
            "chest_pain": "Chest pain, angina, or suspected myocardial ischemia",
            "dyspnea": "Shortness of breath, dyspnea, or reduced exercise tolerance",
            "heart_failure": "Heart failure, cardiomyopathy, edema, or assessment of LV function / ejection fraction",
            "valvular": "Heart murmur or known or suspected valve disease",
            "arrhythmia": "Atrial fibrillation, palpitations, or another arrhythmia",
            "syncope": "Syncope, presyncope, or dizziness",
            "stroke_embolic": "Stroke, TIA, or search for a cardiac source of embolism",
            "endocarditis": "Fever, bacteremia, or suspected endocarditis",
            "pericardial": "Pericardial effusion, pericarditis, or tamponade",
            "hypertension": "Hypertension or hypertensive heart disease",
            "preoperative": "Pre-operative or pre-procedure evaluation",
            "follow_up": "Routine follow-up of a known cardiac condition",
            "other": "A reason that is stated but fits none of the other options",
            "not_stated": "The report does not say why the echocardiogram was ordered",
        },
    },
    "test": {
        "models": ["kev"],  # laya is still a stub
        "limit": 20,        # small sample for a quick test run; None = all rows
        "output": "results/test_predictions.csv",
    },
    "kev": {
        "base_url": "http://127.0.0.1:8009",
        "model": "kev-latest",
        "api_key": None,
        "timeout_s": 60,
    },
    "logging": {"verbose": True},
}


def parse_config_arg(description: str, argv=None) -> Path:
    """The only CLI option: --config path/to/config.yaml."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", "-c", required=True, type=Path, help="Path to YAML config")
    return p.parse_args(argv).config


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        user = yaml.safe_load(f) or {}

    # Merge user values over defaults (one level of nesting is enough here)
    cfg = {}
    for key, default in DEFAULTS.items():
        val = user.get(key, default)
        cfg[key] = {**default, **(val or {})} if isinstance(default, dict) else val

    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if cfg["mode"] not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {cfg['mode']!r}")
    if not cfg["data"]["s3_uri"]:
        raise ValueError("data.s3_uri is required")
    if not cfg["task"]["instructions"]:
        raise ValueError("task.instructions is required")
    labels = cfg["task"]["labels"]
    if not isinstance(labels, dict) or len(labels) < 2:
        raise ValueError("task.labels must map at least two label names to descriptions")
    bad = set(cfg["test"]["models"]) - VALID_MODELS
    if bad:
        raise ValueError(f"Unknown models {sorted(bad)}; choose from {sorted(VALID_MODELS)}")

    cfg["config_path"] = str(path.resolve())
    return cfg


def setup_logging(cfg: dict) -> None:
    logging.basicConfig(
        level=logging.DEBUG if cfg["logging"]["verbose"] else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )