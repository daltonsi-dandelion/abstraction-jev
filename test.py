"""
Test suite: run each selected model over the narrative text column and
record the predicted echo indication for every row.

Normally invoked by `python main.py --config config.yaml` (with mode: test),
but can be run directly: `python test.py --config config.yaml`.
"""
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config import load_config, parse_config_arg, setup_logging
from data import load_parquet
from models import get_model

log = logging.getLogger("test")


def classify(df: pd.DataFrame, model, text_col: str, id_col: str | None) -> pd.DataFrame:
    rows = []
    for idx, text in df[text_col].items():
        row_id = df.at[idx, id_col] if id_col else idx
        if not isinstance(text, str) or not text.strip():
            rows.append({"id": row_id, "model": model.name, "label": None, "confidence": None,
                         "probabilities": None, "error": "empty text", "latency_s": 0.0})
            continue

        start = time.perf_counter()
        try:
            out = model.predict(text)
            err = None
        except Exception as e:  # keep going; record the failure
            log.exception("%s failed on row %s", model.name, row_id)
            out, err = {}, repr(e)
        rows.append({
            "id": row_id,
            "model": model.name,
            "label": out.get("label"),
            "confidence": out.get("confidence"),
            # full distribution over labels, as JSON, so ambiguous reports can be reviewed
            "probabilities": json.dumps(out["probabilities"]) if out.get("probabilities") else None,
            "error": err,
            "latency_s": round(time.perf_counter() - start, 4),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    cfg = load_config(parse_config_arg("Run echo indication classifiers", argv))
    setup_logging(cfg)
    d, t = cfg["data"], cfg["test"]

    df = load_parquet(d["s3_uri"], aws_profile=d["aws_profile"])
    if d["text_column"] not in df.columns:
        log.error("Column '%s' not found. Available: %s", d["text_column"], list(df.columns))
        return 1
    if t["limit"]:
        df = df.head(t["limit"])

    results = []
    for name in t["models"]:
        log.info("Loading model: %s", name)
        model = get_model(name, cfg)
        log.info("Classifying %d rows with %s", len(df), name)
        res = classify(df, model, d["text_column"], d["id_column"])
        log.info("%s: %d predictions, %d errors", name, len(res), res["error"].notna().sum())
        results.append(res)

    out = pd.concat(results, ignore_index=True)

    # Wide view: one column per model for easy side-by-side comparison
    wide = out.pivot(index="id", columns="model", values="label").add_prefix("label_")

    out_path = Path(t["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    wide.to_csv(out_path.with_name(out_path.stem + "_wide.csv"))
    log.info("Wrote %s", out_path)

    print("\nLabel distribution by model:")
    print(out.groupby(["model", "label"], dropna=False).size().unstack(0, fill_value=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
