"""
Phase 6/7: 자연어 → SQL → 답변 (Ollama Qwen). SQLite / PostGIS 백엔드 자동 선택.

- 환경변수 GEOTRAFFIC_DB_URL 있으면 PostgreSQL(PostGIS), 없으면 SQLite(db/geotraffic.db).
- 읽기전용(SELECT만) 가드.

사용:
  python llm/nl2sql.py "가장 혼잡한 카메라는 어디야?"
  set GEOTRAFFIC_DB_URL=postgresql://postgres:geotraffic@localhost:5434/geotraffic  (PostGIS 사용)
"""
import argparse
import json
import os
import re
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_DB = os.path.join(REPO, "db", "geotraffic.db")
PG_URL = os.environ.get("GEOTRAFFIC_DB_URL")
BACKEND = "postgresql" if PG_URL else "sqlite"
OLLAMA = "http://localhost:11434/api/chat"

SCHEMA = """다중 CCTV 교통 객체탐지 결과 DB.

TABLE detection (
  id, ts timestamp, frame int, cctv_id text,
  track_id int,   -- 객체 고유 추적ID (같은 차량은 여러 프레임에 동일 track_id → 대수는 COUNT(DISTINCT track_id))
  class text,     -- car, truck, van, bus, pedestrian, tricycle, awning-tricycle, motor, bicycle, people
  confidence real, speed_kmh real, heading_deg real,  -- NULL 가능
  {geomcols} )
TABLE cctv ( cctv_id text, name text{geomc2} )
VIEW cctv_summary ( cctv_id, name, unique_objects, avg_speed_kmh, detections )

규칙:
- '몇 대/대수/고유 차량' → COUNT(DISTINCT track_id).
- 속도 통계는 WHERE speed_kmh > 1 (정지/노이즈 제외).
- '혼잡/가장 많은 카메라' → cctv_summary 또는 GROUP BY cctv_id.
- 카메라는 여러 대(cctv 테이블). 카메라 이름은 cctv.name.
"""


def schema_text():
    if BACKEND == "postgresql":
        return SCHEMA.format(geomcols="geom geometry(Point,4326), geom_utm geometry(Point,32652)",
                             geomc2=", geom geometry(Point,4326)") + \
            "\n방언: PostgreSQL + PostGIS. 거리/반경은 ST_DWithin(geom::geography, ...), ST_Distance 사용 가능."
    return SCHEMA.format(geomcols="lon real, lat real, easting real, northing real",
                         geomc2=", lon real, lat real") + "\n방언: SQLite (lon/lat 컬럼)."


FEWSHOT = [
    ("버스 몇 대 지나갔어?", "SELECT COUNT(DISTINCT track_id) AS bus_count FROM detection WHERE class='bus'"),
    ("가장 혼잡한 카메라는?", "SELECT name, unique_objects FROM cctv_summary ORDER BY unique_objects DESC LIMIT 1"),
    ("카메라별 평균 속도", "SELECT name, avg_speed_kmh FROM cctv_summary ORDER BY avg_speed_kmh DESC"),
]


def ollama_chat(model, messages, think=False):
    body = json.dumps({"model": model, "messages": messages, "stream": False, "think": think}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]["content"]


def extract_sql(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    m2 = re.search(r"(SELECT\b.*)", sql, flags=re.S | re.I)
    return (m2.group(1) if m2 else sql).strip().rstrip(";").strip()


def is_safe(sql):
    low = sql.lower()
    return low.startswith("select") and ";" not in sql and not re.search(
        r"\b(insert|update|delete|drop|alter|attach|pragma|create|grant)\b", low)


def run_sql(sql):
    if BACKEND == "postgresql":
        import psycopg2
        con = psycopg2.connect(PG_URL)
        con.set_session(readonly=True)
        cur = con.cursor(); cur.execute(sql)
        cols = [d[0] for d in cur.description]; rows = cur.fetchall(); con.close()
        return cols, rows
    import sqlite3
    con = sqlite3.connect(f"file:{SQLITE_DB}?mode=ro", uri=True)
    cur = con.cursor(); cur.execute(sql)
    cols = [d[0] for d in cur.description]; rows = cur.fetchall(); con.close()
    return cols, rows


def _jsonable(rows):
    out = []
    for r in rows:
        out.append([v if isinstance(v, (int, float)) or v is None else str(v) for v in r])
    return out


def answer(question, model="qwen3:8b"):
    """질문 → SQL → 실행 → 요약. dict 반환(웹/CLI 공용)."""
    fs = "\n".join(f"Q: {q}\nSQL: {s}" for q, s in FEWSHOT)
    sys_prompt = (schema_text() + "\n예시:\n" + fs +
                  f"\n\n사용자 질문을 {BACKEND} SELECT 한 문장으로만 변환해 ```sql``` 블록으로 출력. 설명 금지.")
    try:
        sql = extract_sql(ollama_chat(model, [
            {"role": "system", "content": sys_prompt}, {"role": "user", "content": question}]))
        if not is_safe(sql):
            return {"backend": BACKEND, "sql": sql, "error": "안전하지 않은 SQL(읽기전용 SELECT만 허용)"}
        cols, rows = run_sql(sql)
        summary = ollama_chat(model, [
            {"role": "system", "content": "너는 교통 데이터 분석가다. 질문과 SQL 결과를 한국어 한두 문장으로 간결히 답하라."},
            {"role": "user", "content": f"질문: {question}\nSQL: {sql}\n컬럼: {cols}\n행: {rows[:20]}"}])
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.S).strip()
        return {"backend": BACKEND, "sql": sql, "columns": cols,
                "rows": _jsonable(rows[:50]), "answer": summary}
    except Exception as e:
        return {"backend": BACKEND, "error": str(e)}


def main():
    try:  # Windows 콘솔 기본 cp949 → UTF-8 강제(웹 백엔드가 UTF-8로 읽음)
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--json", action="store_true", help="JSON 한 줄 출력(웹 백엔드용)")
    args = ap.parse_args()
    res = answer(args.question, args.model)
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
        return
    print(f"[backend] {res.get('backend')}")
    if res.get("error"):
        sys.exit(res["error"])
    print(f"[SQL] {res['sql']}")
    print(f"[결과] {res['columns']}")
    for r in res["rows"][:20]:
        print("  ", r)
    print(f"\n[답변] {res['answer']}")


if __name__ == "__main__":
    main()
