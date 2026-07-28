"""gzip 압축 JSON replay writer (TrafficLab-3D 이식, 원본 그대로)."""
import gzip
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64, np.intp)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ReplayWriter:
    @staticmethod
    def write(path, data):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
