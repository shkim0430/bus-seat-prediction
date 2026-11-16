# -*- coding: utf-8 -*-
"""홍대입구역 도착 좌석수 예측 가능성 검증.
완전 궤적(4정류소, 순서대로) 260트립 대상, 신촌 도착 '전' 정보만 피처로 사용
(모델3은 비교용으로 신촌 정보 사용 허용).
trip_analysis.py의 트립 구성 로직(30분 트립 분리, last-seen 정렬)을 그대로 재사용."""
import sqlite3
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        s_dt, s_seat = per_station["113000416"]
        h_dt, h_seat = per_station["113000414"]
        records.append({
            "gongdeok_seat": g_seat,
            "sinchon_seat": s_seat,
            "hongdae_seat": h_seat,
            "hour": g_dt.hour,          # 공덕 시점(예측 시점) 기준 — 신촌 이전 정보만 사용
            "dow": g_dt.weekday(),      # 0=월 ... 6=일
        })

print(f"완전 궤적 트립 수: {len(records)}\n")

y = np.array([r["hongdae_seat"] for r in records], dtype=float)


def onehot_dow(dow_list, drop_first=True):
    n_cat = 7
    mat = np.zeros((len(dow_list), n_cat))
    for i, d in enumerate(dow_list):
        mat[i, d] = 1
    if drop_first:
        mat = mat[:, 1:]  # 월요일을 기준(baseline)으로 drop
    return mat


X1 = np.array([[r["gongdeok_seat"]] for r in records], dtype=float)

hour_arr = np.array([[r["hour"]] for r in records], dtype=float)
dow_oh = onehot_dow([r["dow"] for r in records])
X2 = np.hstack([X1, hour_arr, dow_oh])

X3 = np.array([[r["sinchon_seat"]] for r in records], dtype=float)

MODELS = {
    "모델1 (공덕만)": X1,
    "모델2 (공덕+시간대+요일)": X2,
    "모델3 (신촌만, 참고용)": X3,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'모델':28}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'오차≤5석':>10}{'오차≤10석':>10}")
results = {}
for name, X in MODELS.items():
    model = LinearRegression()
    y_pred = cross_val_predict(model, X, y, cv=kf)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    err = np.abs(y - y_pred)
    within5 = np.mean(err <= 5) * 100
    within10 = np.mean(err <= 10) * 100
    results[name] = (mae, rmse, r2, within5, within10)
    print(f"{name:28}{mae:8.2f}{rmse:8.2f}{r2:8.3f}{within5:9.1f}%{within10:9.1f}%")

print("\n판정 기준: MAE<=3 서비스가능 / MAE<=5 유용 / MAE>=8 애매")
for name, (mae, *_ ) in results.items():
    if mae <= 3:
        verdict = "서비스 가능 수준"
    elif mae <= 5:
        verdict = "유용"
    elif mae >= 8:
        verdict = "애매"
    else:
        verdict = "중간(5~8)"
    print(f"  {name}: MAE={mae:.2f} -> {verdict}")

# 참고: 전체 계수(마지막 fold가 아니라 전체 데이터로 재적합) — 해석용
print("\n(참고) 전체 데이터로 재적합한 회귀계수")
for name, X in MODELS.items():
    model = LinearRegression().fit(X, y)
    print(f"  {name}: intercept={model.intercept_:.2f}, coef[0](주좌석피처)={model.coef_[0]:.3f}")
