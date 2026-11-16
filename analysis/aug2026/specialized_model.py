# -*- coding: utf-8 -*-
"""'빡빡한 상황' 특화 모델 실험.
실험 A: 공덕<=30 트립만으로 학습+CV (특화 모델)
실험 B: 공덕>=31 트립만으로 학습+CV (참고용, 여유 전용)
실험 C: 실험 A와 '동일한 테스트 포인트'에 대해, 학습은 전체 데이터(해당 폴드 테스트포인트만 제외)로
        한 모델의 예측 — 기존 전체모델을 그 구간에서 재확인하는 대조군.
        (A와 C는 같은 5-fold 분할·같은 held-out 포인트를 써서 apples-to-apples 비교)
"""
import sqlite3
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import os as _os, sys as _sys
# DB 경로: --db <경로> 인자 > BUS_DB_PATH 환경변수 > data/bus.db
DB_PATH = (_sys.argv[_sys.argv.index("--db") + 1] if "--db" in _sys.argv
           else _os.environ.get("BUS_DB_PATH", "data/bus.db"))
ROUTE_ID = 232000071

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
        g_seat = per_station["113000088"][1]
        h_seat = per_station["113000414"][1]
        records.append((g_seat, h_seat))

print(f"완전 궤적 트립 수 (전체): {len(records)}")

X_all = np.array([[r[0]] for r in records], dtype=float)
y_all = np.array([r[1] for r in records], dtype=float)

le30_idx = np.where(X_all[:, 0] <= 30)[0]
ge31_idx = np.where(X_all[:, 0] >= 31)[0]
print(f"공덕<=30 (빡빡) 트립: {len(le30_idx)}건")
print(f"공덕>=31 (여유) 트립: {len(ge31_idx)}건\n")


def bin_user(v):
    v = int(v)
    if v <= 5:
        return (0, 5)
    lo = ((v - 6) // 5) * 5 + 6
    return (lo, lo + 4)


def hongdae_bin_breakdown_user(y_true, y_pred, bins_wanted):
    err = np.abs(y_true - y_pred)
    groups = defaultdict(list)
    for yt, e in zip(y_true, err):
        groups[bin_user(yt)].append(e)
    print(f"  {'홍대구간':10}{'n':>6}{'MAE':>8}")
    for key in bins_wanted:
        errs = groups.get(key, [])
        if errs:
            print(f"  {key[0]:2}-{key[1]:<7}{len(errs):6}{np.mean(errs):8.2f}")
        else:
            print(f"  {key[0]:2}-{key[1]:<7}{0:6}{'(표본없음)':>8}")


WANTED_BINS = [(0, 5), (6, 10), (11, 15), (16, 20)]

kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

# ---- 실험 A: 빡빡 특화 모델 (공덕<=30 subset 내에서 5-fold CV) ----
print("=" * 60)
print("[실험 A] 빡빡 특화 모델 (학습: 공덕<=30 트립만)")
print("=" * 60)
Xs, ys = X_all[le30_idx], y_all[le30_idx]
predA = np.zeros(len(ys))
for tr, te in kf5.split(Xs):
    m = LinearRegression().fit(Xs[tr], ys[tr])
    predA[te] = m.predict(Xs[te])
maeA = mean_absolute_error(ys, predA)
print(f"n={len(ys)}  전체 MAE={maeA:.2f}\n")
hongdae_bin_breakdown_user(ys, predA, WANTED_BINS)

# ---- 실험 B: 여유 전용 모델 (참고용) ----
print("\n" + "=" * 60)
print("[실험 B] 여유 전용 모델 (학습: 공덕>=31 트립만, 참고용)")
print("=" * 60)
Xs2, ys2 = X_all[ge31_idx], y_all[ge31_idx]
predB = np.zeros(len(ys2))
for tr, te in kf5.split(Xs2):
    m = LinearRegression().fit(Xs2[tr], ys2[tr])
    predB[te] = m.predict(Xs2[te])
maeB = mean_absolute_error(ys2, predB)
print(f"n={len(ys2)}  전체 MAE={maeB:.2f}\n")
hongdae_bin_breakdown_user(ys2, predB, WANTED_BINS)

# ---- 실험 C: 대조군 — 실험 A와 동일 테스트포인트, 학습은 전체데이터(해당 폴드 test만 제외) ----
print("\n" + "=" * 60)
print("[실험 C] 대조군 — 실험 A와 동일 테스트셋, 학습은 전체 데이터(폴드별 test만 제외)")
print("=" * 60)
predC = np.zeros(len(ys))  # 실험 A와 동일 순서(le30_idx 순서)로 채움
for tr_rel, te_rel in kf5.split(Xs):
    te_abs = le30_idx[te_rel]           # 이 폴드의 테스트 포인트 = 원본 데이터에서의 절대 인덱스
    train_mask = np.ones(len(X_all), dtype=bool)
    train_mask[te_abs] = False          # 테스트 포인트만 학습에서 제외, 나머지 전부(여유 포함) 사용
    m = LinearRegression().fit(X_all[train_mask], y_all[train_mask])
    predC[te_rel] = m.predict(X_all[te_abs])
maeC = mean_absolute_error(ys, predC)
print(f"n={len(ys)} (실험 A와 동일)  전체 MAE={maeC:.2f}\n")
hongdae_bin_breakdown_user(ys, predC, WANTED_BINS)

# ---- 핵심 비교: A vs C, 저좌석 구간(홍대<=15) ----
print("\n" + "=" * 60)
print("[핵심 비교] 홍대<=15석 구간에서 A(특화) vs C(전체모델)")
print("=" * 60)
low_mask = ys <= 15
nlow = low_mask.sum()
maeA_low = mean_absolute_error(ys[low_mask], predA[low_mask]) if nlow else float("nan")
maeC_low = mean_absolute_error(ys[low_mask], predC[low_mask]) if nlow else float("nan")
print(f"n={nlow}")
print(f"실험 A(빡빡특화) MAE = {maeA_low:.2f}")
print(f"실험 C(전체모델) MAE = {maeC_low:.2f}")
if nlow:
    diff = maeC_low - maeA_low
    print(f"차이(C-A) = {diff:.2f}  ({'A가 더 정확' if diff>0 else 'C가 더 정확 또는 동률'})")
