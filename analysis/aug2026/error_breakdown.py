# -*- coding: utf-8 -*-
"""모델1(공덕만) 오차 분포 분석: 공덕 좌석 구간별 / 홍대 실제 좌석 구간별 / 시간대별 MAE.
predict_hongdae.py와 동일한 데이터셋·트립 구성 로직 재사용, 모델1만 다시 적합."""
import sqlite3
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

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
conn.close()

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


records = []
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
        if not all(sta_seq[i] < sta_seq[i + 1] for i in range(len(sta_seq) - 1)):
            continue
        g_dt, g_seat = per_station["113000088"]
        h_dt, h_seat = per_station["113000414"]
        records.append({"gongdeok_seat": g_seat, "hongdae_seat": h_seat, "hour": g_dt.hour})

print(f"완전 궤적 트립 수: {len(records)}\n")

y = np.array([r["hongdae_seat"] for r in records], dtype=float)
X1 = np.array([[r["gongdeok_seat"]] for r in records], dtype=float)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()
y_pred = cross_val_predict(model, X1, y, cv=kf)
abs_err = np.abs(y - y_pred)

gongdeok = np.array([r["gongdeok_seat"] for r in records])
hongdae_actual = y
hours = np.array([r["hour"] for r in records])


def bin59(v):
    if v <= 5:
        return (0, 5)
    lo = ((v - 6) // 5) * 5 + 6
    return (lo, lo + 4)


def print_bin_breakdown(title, values):
    print("=" * 60)
    print(title)
    print("=" * 60)
    groups = defaultdict(list)
    for v, e in zip(values, abs_err):
        groups[bin59(int(v))].append(e)
    print(f"{'구간':10}{'n':>6}{'MAE':>8}")
    for key in sorted(groups.keys()):
        errs = groups[key]
        n = len(errs)
        mae = np.mean(errs)
        lo, hi = key
        print(f"{lo:2}-{hi:<7}{n:6}{mae:8.2f}")
    print()


print_bin_breakdown("[분석 1] 공덕 좌석수 구간별 예측 오차 (5석 단위)", gongdeok)
print_bin_breakdown("[분석 2] 홍대 실제 좌석수 구간별 예측 오차 (5석 단위)", hongdae_actual)

print("=" * 60)
print("[분석 3] 시간대별 예측 오차 (16~22시)")
print("=" * 60)
hour_groups = defaultdict(list)
for h, e in zip(hours, abs_err):
    hour_groups[h].append(e)
print(f"{'시간대':6}{'n':>6}{'MAE':>8}")
for h in sorted(hour_groups.keys()):
    errs = hour_groups[h]
    n = len(errs)
    mae = np.mean(errs)
    print(f"{h}시{'':3}{n:6}{mae:8.2f}")

print(f"\n전체 MAE (검증용): {np.mean(abs_err):.2f}")
