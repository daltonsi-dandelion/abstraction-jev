"""
Entry point for the echo-indication research workflow.

All settings live in a YAML config; the only CLI option is the config path:
    python main.py --config config.yaml
"""
import logging
import subprocess
import sys
from pathlib import Path

from config import load_config, parse_config_arg, setup_logging
from data import load_parquet

HERE = Path(__file__).resolve().parent
log = logging.getLogger("main")


def run_test_suite(cfg: dict) -> int:
    """Invoke test.py as a separate script with the same config file."""
    cmd = [sys.executable, str(HERE / "test.py"), "--config", cfg["config_path"]]
    log.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def run_preview(cfg: dict) -> int:
    d = cfg["data"]
    df = load_parquet(d["s3_uri"], aws_profile=d["aws_profile"])
    print(df.shape)
    print(df.head())
    if d["text_column"] in df.columns:
        print(f"\nSample '{d['text_column']}':\n{df[d['text_column']].iloc[0]}")
    return 0


def main(argv=None) -> int:
    cfg = load_config(parse_config_arg("Echo indication classification workflow", argv))
    setup_logging(cfg)

    if cfg["mode"] == "test":
        return run_test_suite(cfg)
    return run_preview(cfg)


if __name__ == "__main__":
    sys.exit(main())
