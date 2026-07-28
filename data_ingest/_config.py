"""공용 설정 로더. project_config.json 을 읽어 절대경로로 변환한다."""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env():
    """리포 루트의 .env.local / .env 를 os.environ 에 주입(기존 값은 유지)."""
    for name in (".env.local", ".env"):
        p = os.path.join(REPO_ROOT, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")  # 양끝 따옴표 제거
                os.environ.setdefault(k, v)


def load_config():
    load_env()
    with open(os.path.join(REPO_ROOT, "project_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def data_out_dir(cfg, *sub):
    d = os.path.join(REPO_ROOT, cfg["paths"].get("web_data_out", cfg["paths"].get("viewer_data_out")), *sub)
    os.makedirs(d, exist_ok=True)
    return d
