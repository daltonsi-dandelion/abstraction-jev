"""
Adapters around the kev and laya models so test.py can treat them the same way.

Every adapter takes the loaded config and returns, for each text:
    {"label": str | None, "confidence": float | None,
     "probabilities": dict[str, float] | None, "raw": any}

kev
---
Kev runs as a local server (System One API) from its own repo, which sits
next to this project. Start it in another terminal before running the suite:

    cd ../kev
    uv sync --extra serve
    uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009

The adapter sends each narrative as one `choice` question whose options are
the labels in config.yaml (task.labels) and reads back the probabilities.

laya
----
Still a stub. It expects the laya repo next to this project (or LAYA_REPO).
"""
import logging
import os
import sys
from pathlib import Path

import requests

log = logging.getLogger("models")

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent


def _add_repo_to_path(env_var: str, default_name: str) -> Path:
    repo = Path(os.environ.get(env_var, PARENT_DIR / default_name)).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"{default_name} repo not found at {repo} (set {env_var})")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


class BaseClassifier:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.instructions = cfg["task"]["instructions"]
        self.labels = cfg["task"]["labels"]  # {label_name: description}

    def load(self):
        """Load weights / connect to a server. Called once before predict."""
        raise NotImplementedError

    def predict(self, text: str) -> dict:
        raise NotImplementedError


class KevClassifier(BaseClassifier):
    """Calls a running kev server: POST {base_url}/v1/systemone."""
    name = "kev"
    QUESTION_ID = "indication"

    def load(self):
        k = self.cfg["kev"]
        self.base_url = k["base_url"].rstrip("/")
        self.model = k["model"]
        self.timeout = k["timeout_s"]
        self.session = requests.Session()
        api_key = k["api_key"] or os.environ.get("KEV_API_KEY")
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        # Fail early with a clear message if the server is not running
        try:
            r = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Can't reach the kev server at {self.base_url} ({e}). Start it with:\n"
                "  cd ../kev && uv run --extra serve python -m kev.serve "
                "--run jaredpalmer/kev-4b --port 8009"
            ) from e
        log.info("Connected to kev at %s", self.base_url)

    def predict(self, text: str) -> dict:
        payload = {
            "state": text,
            "model": self.model,
            "questions": {
                self.QUESTION_ID: {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": self.labels,
                }
            },
        }
        r = self.session.post(f"{self.base_url}/v1/systemone", json=payload, timeout=self.timeout)
        if r.status_code == 422:  # invalid request or text too long; kev explains why
            raise ValueError(f"kev rejected the request: {r.text[:500]}")
        r.raise_for_status()
        body = r.json()
        ans = body["answers"][self.QUESTION_ID]
        return {
            "label": ans.get("choice"),
            "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities"),
            "raw": body,
        }


class LayaClassifier(BaseClassifier):
    name = "laya"

    def load(self):
        self.repo = _add_repo_to_path("LAYA_REPO", "laya")
        # TODO: replace with laya's real entry point
        self.model = None

    def predict(self, text: str) -> dict:
        # TODO: call laya and map its output onto self.labels
        return {"label": None, "confidence": None, "probabilities": None, "raw": None}


REGISTRY = {
    "kev": KevClassifier,
    "laya": LayaClassifier,
}


def get_model(name: str, cfg: dict) -> BaseClassifier:
    model = REGISTRY[name](cfg)
    model.load()
    return model
