import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import config

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "bus.db"
LOG_PATH = BASE_DIR / "logs" / "collect.log"

ARRIVAL_LIST_URL = "https://apis.data.go.kr/6410000/busarrivalservice/v2/getBusArrivalListv2"

KST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS arrivals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT NOT NULL,   -- KST ISO8601
            station_id INTEGER NOT NULL,
            route_id INTEGER,
            route_name TEXT,
            sta_order INTEGER,
            predict_time1 TEXT,
            predict_time2 TEXT,
            location_no1 TEXT,
            location_no2 TEXT,
            remain_seat_cnt1 INTEGER,       -- predictTime 비거나 좌석정보 없음(-1)이면 NULL
            remain_seat_cnt2 INTEGER,       -- predictTime 비거나 좌석정보 없음(-1)이면 NULL
            seat_info_available1 INTEGER,   -- NULL=버스없음, 0=좌석정보 미제공(-1), 1=실좌석수
            seat_info_available2 INTEGER,
            crowded1 INTEGER,               -- 혼잡도. remainSeatCnt와 독립적 신호. 버스없으면 NULL
            crowded2 INTEGER,
            plate_no1 TEXT,
            plate_no2 TEXT,
            raw TEXT NOT NULL             -- API 원본 item JSON 그대로
        )
        """
    )
    _ensure_columns(
        conn,
        {
            "seat_info_available1": "INTEGER",
            "seat_info_available2": "INTEGER",
            "crowded1": "INTEGER",
            "crowded2": "INTEGER",
        },
    )
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection, columns: dict[str, str]) -> None:
    """기존 DB(구 스키마)에 새 컬럼을 추가한다. 이미 있으면 건너뜀."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(arrivals)")}
    for name, coltype in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE arrivals ADD COLUMN {name} {coltype}")


def unique_station_ids() -> list[str]:
    """TARGETS에서 정류소ID만 중복 없이 순서 보존하여 추출."""
    seen: dict[str, None] = {}
    for t in config.TARGETS:
        seen.setdefault(str(t["station_id"]), None)
    return list(seen)


def fetch_station_arrivals(station_id: str) -> list[dict]:
    """정류소 하나의 모든 노선 도착정보 목록을 반환 (노선 필터링 없음)."""
    params = {"serviceKey": config.GBIS_API_KEY, "stationId": station_id}
    resp = requests.get(ARRIVAL_LIST_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    header = data["response"]["msgHeader"]
    if header["resultCode"] != 0:
        raise RuntimeError(
            f"API error for station {station_id}: {header['resultMessage']}"
        )

    body = data["response"].get("msgBody") or {}
    items = body.get("busArrivalList", [])
    if isinstance(items, dict):  # 단건이면 dict로 옴
        items = [items]
    return items


def _parse_seat_info(predict_time, seat):
    """remainSeatCnt 3가지 상태를 (remain_seat_cnt, seat_info_available)로 분리.

    - predictTime 비어있음 (해당 순번 버스 없음): (None, None)
    - remainSeatCnt == -1 (정수, 좌석정보 미제공 — 만석과 다름): (None, 0)
    - 그 외 정수 (실좌석수, 0 포함): (그 값, 1)
    원본 문자열/정수는 raw 컬럼에 그대로 남는다.
    """
    if predict_time is None or str(predict_time).strip() == "":
        return None, None
    if seat is None or str(seat).strip() == "":
        return None, None
    try:
        seat_int = int(seat)
    except (TypeError, ValueError):
        return None, None
    if seat_int == -1:
        return None, 0
    return seat_int, 1


def _parse_crowded(crowded):
    """crowded 값을 INTEGER로. 빈 문자열(버스 없음)이면 NULL."""
    if crowded is None or str(crowded).strip() == "":
        return None
    try:
        return int(crowded)
    except (TypeError, ValueError):
        return None


def save_arrivals(conn: sqlite3.Connection, station_id: str, items: list[dict]) -> None:
    collected_at = datetime.now(KST).isoformat()
    rows = []
    for item in items:
        seat1, avail1 = _parse_seat_info(item.get("predictTime1"), item.get("remainSeatCnt1"))
        seat2, avail2 = _parse_seat_info(item.get("predictTime2"), item.get("remainSeatCnt2"))
        rows.append(
            (
                collected_at,
                station_id,
                item.get("routeId"),
                item.get("routeName"),
                item.get("staOrder"),
                item.get("predictTime1"),
                item.get("predictTime2"),
                item.get("locationNo1"),
                item.get("locationNo2"),
                seat1,
                seat2,
                avail1,
                avail2,
                _parse_crowded(item.get("crowded1")),
                _parse_crowded(item.get("crowded2")),
                item.get("plateNo1"),
                item.get("plateNo2"),
                json.dumps(item, ensure_ascii=False),
            )
        )
    conn.executemany(
        """
        INSERT INTO arrivals (
            collected_at, station_id, route_id, route_name, sta_order,
            predict_time1, predict_time2, location_no1, location_no2,
            remain_seat_cnt1, remain_seat_cnt2,
            seat_info_available1, seat_info_available2,
            crowded1, crowded2,
            plate_no1, plate_no2, raw
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def main() -> None:
    station_ids = unique_station_ids()
    if not station_ids:
        logger.warning("config.TARGETS가 비어 있습니다. 수집 대상을 설정하세요.")
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    for station_id in station_ids:
        try:
            items = fetch_station_arrivals(station_id)
            save_arrivals(conn, station_id, items)
            logger.info("station %s: %d건 수집", station_id, len(items))
        except Exception:
            logger.exception("station %s 수집 실패", station_id)

    conn.close()


if __name__ == "__main__":
    main()
