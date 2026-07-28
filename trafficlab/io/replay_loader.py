"""replay(.json / .json.gz) 로더 (TrafficLab-3D 시각화 로더 이식)."""
import gzip
import json


def load_replay(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
