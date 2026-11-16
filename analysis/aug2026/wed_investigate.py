# -*- coding: utf-8 -*-
"""수요일 완전궤적 저조 원인 파악: [D-1] 날짜별 breakdown + 누락 정류소,
[D-2] 그 날짜의 수집 배치 정상 여부 vs M6117 데이터 자체 결측 여부 진단."""
import sqlite3
from datetime import datetime
from collections import defaultdict, Counter

import os as _os, sys as _sys
# DB 경로: --db <경로> 인자 > BUS_DB_PATH 환경변수 > data/bus.db
DB_PATH = (_sys.argv[_sys.argv.index("--db") + 1] if "--db" in _sys.argv
           else _os.environ.get("BUS_DB_PATH", "data/bus.db"))
ROUTE_ID = 232000071  # M6117

STATIONS = {
    "113000088": ("공덕", 31),
    "113000416": ("신촌", 32),
    "113000414": ("홍대", 33),
    "113000412": ("합정", 36),
}
DOW_KR = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(f"""
    SELECT collected_at, station_id, plate_no1, remain_seat_cnt1
    FROM arrivals
    WHERE route_id = {ROUTE_ID}
      AND station_id IN ({",".join(STATIONS.keys())})
      AND remain_seat_cnt1 IS NOT NULL
      AND plate_no1 != ''
      AND substr(collected_at,1,10) != '2026-08-06'
""")
rows = cur.fetchall()

sightings = defaultdict(list)
for collected_at, station_id, plate_no, seat in rows:
    dt = datetime.fromisoformat(collected_at)
    sightings[plate_no].append((dt, str(station_id), seat))


def split_trips(obs, gap_minutes=30):
    obs_sorted = sorted(obs, key=lambda x: x[0])
    trips, current, prev = [], [], None
    for o in obs_sorted:
        if prev is not None and (o[0] - prev).total_seconds() > gap_minutes * 60:
            trips.append(current)
            current = []
        current.append(o)
        prev = o[0]
    if current:
        trips.append(current)
    return trips


all_trips = []
for plate, obs in sightings.items():
    for trip in split_trips(obs):
        per_station = {}
        for dt, sid, seat in trip:
            if sid not in per_station or dt > per_station[sid][0]:
                per_station[sid] = (dt, seat)
        start_dt = trip[0][0]
        ordered_stations = sorted(per_station.items(), key=lambda x: x[1][0])
        sta_seq = [STATIONS[s][1] for s, _ in ordered_stations]
        is_ordered = all(sta_seq[i] < sta_seq[i + 1] for i in range(len(sta_seq) - 1))
        n = len(per_station)
        all_trips.append({
            "plate": plate, "start_dt": start_dt, "date": start_dt.date().isoformat(),
            "dow": DOW_KR[start_dt.weekday()], "per_station": per_station, "n": n,
            "is_ordered": is_ordered,
        })

wed_trips = [t for t in all_trips if t["dow"] == "수"]
print(f"수요일 전체 트립 수: {len(wed_trips)}\n")

print("=" * 70)
print("[D-1] 수요일 트립 날짜별 breakdown")
print("=" * 70)

by_date = defaultdict(list)
for t in wed_trips:
    by_date[t["date"]].append(t)

for d in sorted(by_date.keys()):
    trips = by_date[d]
    total = len(trips)
    four = sum(1 for t in trips if t["n"] == 4 and t["is_ordered"])
    three = sum(1 for t in trips if t["n"] == 3 and t["is_ordered"])
    missing_counter = Counter()
    for t in trips:
        if t["n"] == 3 and t["is_ordered"]:
            missing = set(STATIONS.keys()) - set(t["per_station"].keys())
            for m in missing:
                missing_counter[STATIONS[m][0]] += 1
    print(f"{d}: 총 {total}건, 완전궤적 {four}건({four/total*100:.0f}%), 3개통과 {three}건, "
          f"3개 트립의 누락 정류소: {dict(missing_counter) if missing_counter else '-'}")

# 가장 저조한 날짜 자동 식별 (완전궤적 비율 최저, 표본 5건 이상)
worst_date = None
worst_ratio = 1.1
for d, trips in by_date.items():
    if len(trips) < 5:
        continue
    ratio = sum(1 for t in trips if t["n"] == 4 and t["is_ordered"]) / len(trips)
    if ratio < worst_ratio:
        worst_ratio = ratio
        worst_date = d

print(f"\n→ 가장 저조한 날짜: {worst_date} (완전궤적 비율 {worst_ratio*100:.0f}%)")

# 그 날짜의 3개통과 트립들이 공통으로 어느 정류소를 빼먹었는지
if worst_date:
    trips_that_day = by_date[worst_date]
    missing_counter = Counter()
    for t in trips_that_day:
        if t["n"] == 3 and t["is_ordered"]:
            missing = set(STATIONS.keys()) - set(t["per_station"].keys())
            for m in missing:
                missing_counter[m] += 1
    suspect_station = missing_counter.most_common(1)[0][0] if missing_counter else None
    print(f"→ 의심 정류소: {STATIONS.get(suspect_station, ('없음',))[0] if suspect_station else '없음'} "
          f"({missing_counter})")

    # ---- [D-2] 그 날짜의 배치 정상 여부 vs M6117 데이터 결측 여부 ----
    print("\n" + "=" * 70)
    print(f"[D-2] {worst_date} 정류소별 수집 배치 진단")
    print("=" * 70)

    for sid, (name, order) in STATIONS.items():
        cur.execute("""
            SELECT COUNT(DISTINCT collected_at) FROM arrivals
            WHERE station_id=? AND substr(collected_at,1,10)=?
        """, (sid, worst_date))
        total_cycles = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM arrivals
            WHERE station_id=? AND route_id=? AND substr(collected_at,1,10)=?
        """, (sid, ROUTE_ID, worst_date))
        m6117_rows = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM arrivals
            WHERE station_id=? AND route_id=? AND substr(collected_at,1,10)=?
              AND (predict_time1 IS NULL OR predict_time1='')
        """, (sid, ROUTE_ID, worst_date))
        empty_predict = cur.fetchone()[0]

        mark = "  <== 의심 정류소" if sid == suspect_station else ""
        print(f"  {name:4} 전체배치(cycle)={total_cycles:4}  M6117행={m6117_rows:4}  "
              f"(그중 predictTime공백={empty_predict:4}, M6117 자체 결측={total_cycles - m6117_rows:4}){mark}")

conn.close()
