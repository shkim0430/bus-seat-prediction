# -*- coding: utf-8 -*-
"""M6117 완전 궤적(4정류소, 순서대로) 트립에서 구간별 좌석 감소 패턴 정량화.
trip_analysis.py의 트립 구성 로직(30분 트립 분리, last-seen 정렬)을 그대로 재사용."""
import sqlite3
import math
from datetime import datetime
from collections import defaultdict

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
STATION_SEQ = ["113000088", "113000416", "113000414", "113000412"]

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


complete_trips = []
for plate, obs in sightings.items():
    for trip in split_trips(obs):
        per_station = {}
        for dt, sid, seat in trip:
            if sid not in per_station or dt > per_station[sid][0]:
                per_station[sid] = (dt, seat)
        if len(per_station) != 4:
            continue
        ordered = sorted(per_station.items(), key=lambda x: x[1][0])
        sta_seq = [STATIONS[s][1] for s, _ in ordered]
        is_ordered = all(sta_seq[i] < sta_seq[i + 1] for i in range(len(sta_seq) - 1))
        if not is_ordered:
            continue
        complete_trips.append(per_station)

print(f"완전 궤적(4정류소, 순서대로) 트립 수: {len(complete_trips)}\n")


def percentile(sorted_list, p):
    if not sorted_list:
        return None
    k = (len(sorted_list) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)


# ---- [쿼리 E-1] 구간별 좌석 감소 통계 ----
print("=" * 70)
print("[쿼리 E-1] 구간별 좌석 감소 통계")
print("=" * 70)

SEGMENTS = [("113000088", "113000416", "공덕→신촌"),
            ("113000416", "113000414", "신촌→홍대"),
            ("113000414", "113000412", "홍대→합정")]

seg_drops = {}
for a, b, label in SEGMENTS:
    drops = [t[a][1] - t[b][1] for t in complete_trips]
    seg_drops[label] = drops
    sd = sorted(drops)
    n = len(sd)
    mean = sum(sd) / n
    median = percentile(sd, 0.5)
    p25 = percentile(sd, 0.25)
    p75 = percentile(sd, 0.75)
    mx = max(sd)
    increase_ratio = sum(1 for d in sd if d < 0) / n * 100
    print(f"\n[{label}] n={n}")
    print(f"  평균={mean:.2f}  중앙값={median:.1f}  p25={p25:.1f}  p75={p75:.1f}  최대={mx}")
    print(f"  좌석 증가(하차 우세) 비율: {increase_ratio:.1f}% ({sum(1 for d in sd if d<0)}건)")


# ---- [쿼리 E-2] 신촌→홍대 구간, 시간대별 (홍대 도착 시각 기준) ----
print("\n" + "=" * 70)
print("[쿼리 E-2] 신촌→홍대 구간 좌석 감소, 시간대별 (홍대 도착 시각 기준)")
print("=" * 70)

by_hour = defaultdict(list)
for t in complete_trips:
    drop = t["113000416"][1] - t["113000414"][1]
    hr = t["113000414"][0].strftime("%H")
    by_hour[hr].append(drop)

print(f"{'시간대':6}{'n':>5}{'평균':>8}{'중앙값':>8}{'p90':>8}")
for hr in sorted(by_hour.keys()):
    vals = sorted(by_hour[hr])
    n = len(vals)
    mean = sum(vals) / n
    median = percentile(vals, 0.5)
    p90 = percentile(vals, 0.9)
    print(f"{hr}시{'':3}{n:5}{mean:8.2f}{median:8.1f}{p90:8.1f}")


# ---- [쿼리 E-3] 홍대 vs 신촌 좌석수 상관 ----
print("\n" + "=" * 70)
print("[쿼리 E-3] 신촌 도착 좌석수 vs 홍대 도착 좌석수 상관")
print("=" * 70)

sinchon_seats = [t["113000416"][1] for t in complete_trips]
hongdae_seats = [t["113000414"][1] for t in complete_trips]
r = pearson(sinchon_seats, hongdae_seats)
print(f"피어슨 상관계수: {r:.3f}  (n={len(complete_trips)})")

print("\n신촌 좌석 구간별 → 홍대 평균 좌석 (5석 단위)")
bins = defaultdict(list)
for s, h in zip(sinchon_seats, hongdae_seats):
    lo = (s // 5) * 5
    hi = lo + 4
    key = (lo, hi)
    bins[key].append(h)

print(f"{'신촌구간':10}{'n':>5}{'홍대평균':>10}{'홍대중앙값':>10}")
for (lo, hi) in sorted(bins.keys()):
    vals = sorted(bins[(lo, hi)])
    n = len(vals)
    mean = sum(vals) / n
    median = percentile(vals, 0.5)
    print(f"{lo:2}-{hi:<7}{n:5}{mean:10.1f}{median:10.1f}")

conn.close()
