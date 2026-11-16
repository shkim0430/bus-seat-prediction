# -*- coding: utf-8 -*-
# 주의: scratchpad 유실로 보고 기준 코드를 재작성한 버전이다. 11-02 결과 재현은
# --until 2026-11-02 (collected_at < '2026-11-02' 필터)로 하고,
# 수치는 원본 실행과 대조되지 않음.
# 사용: BUS_DB_PATH=<bus.db 경로> python reanalysis.py --until YYYY-MM-DD   (DB는 읽기 전용으로만 연다)
"""정식 재분석 (3단계). 기존 5개 스크립트와 동일 방법론:
substr/ISO 파싱(strftime 금지), 08-06 제외, IS NOT NULL, 30분 트립 분리, last-seen 정렬,
완전 궤적(4정류소 순서대로) 트립만. 트립당 정류소별 last-seen 1건 -> 중복 관측은 이미 하나로 묶임."""
import sqlite3
from datetime import datetime, date
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

import argparse, os, sys
from pathlib import Path

_ap = argparse.ArgumentParser(description="DB는 BUS_DB_PATH 환경변수 또는 --db 로 지정(읽기 전용).")
_ap.add_argument("--db", default=os.environ.get("BUS_DB_PATH"), help="bus.db 경로 (기본: 환경변수 BUS_DB_PATH)")
_ap.add_argument("--until", required=True, help="이 날짜 미만의 collected_at만 사용 (YYYY-MM-DD, 기본값 없음)")
_args = _ap.parse_args()
if not _args.db:
    sys.exit("BUS_DB_PATH 환경변수 또는 --db 로 DB 경로를 지정하세요.")
UNTIL = _args.until
DB = Path(_args.db).resolve().as_uri() + "?mode=ro"
ROUTE_ID = 232000071
STATIONS = {"113000088": 31, "113000416": 32, "113000414": 33, "113000412": 36}
G, S_, H, J = "113000088", "113000416", "113000414", "113000412"

conn = sqlite3.connect(DB, uri=True)
rows = conn.execute(f"""
    SELECT collected_at, station_id, plate_no1, remain_seat_cnt1 FROM arrivals
    WHERE route_id = {ROUTE_ID} AND station_id IN ({",".join(STATIONS)})
      AND remain_seat_cnt1 IS NOT NULL AND plate_no1 != ''
      AND substr(collected_at,1,10) != '2026-08-06'
      AND substr(collected_at,1,10) < '{UNTIL}'""").fetchall()
last_day = conn.execute("select max(substr(collected_at,1,10)) from arrivals where substr(collected_at,1,10) < ?", (UNTIL,)).fetchone()[0]
conn.close()

sight = defaultdict(list)
for ca, sid, plate, seat in rows:
    sight[plate].append((datetime.fromisoformat(ca), str(sid), seat))


def split_trips(obs, gap=30):
    obs = sorted(obs, key=lambda x: x[0])
    out, cur, prev = [], [], None
    for o in obs:
        if prev is not None and (o[0] - prev).total_seconds() > gap * 60:
            out.append(cur); cur = []
        cur.append(o); prev = o[0]
    if cur: out.append(cur)
    return out


recs, n_trips_all, n_4_misordered = [], 0, 0
for plate, obs in sight.items():
    for trip in split_trips(obs):
        n_trips_all += 1
        ps = {}
        for dt, sid, seat in trip:
            if sid not in ps or dt > ps[sid][0]:
                ps[sid] = (dt, seat)
        if len(ps) != 4:
            continue
        order = sorted(ps.items(), key=lambda x: x[1][0])
        seq = [STATIONS[s] for s, _ in order]
        if not all(seq[i] < seq[i + 1] for i in range(3)):
            n_4_misordered += 1
            continue
        recs.append(dict(plate=plate, g=ps[G][1], s=ps[S_][1], h=ps[H][1], j=ps[J][1],
                         g_dt=ps[G][0], h_dt=ps[H][0]))

N = len(recs)
g = np.array([r["g"] for r in recs], float)
s = np.array([r["s"] for r in recs], float)
h = np.array([r["h"] for r in recs], float)
j = np.array([r["j"] for r in recs], float)
h_hour = np.array([r["h_dt"].hour for r in recs])
g_hour = np.array([r["g_dt"].hour for r in recs])
g_date = np.array([r["g_dt"].date() for r in recs])

print("=" * 72)
print(f"[1단계] 데이터 현황  (DB 마지막 수집일 {last_day})")
print("=" * 72)
print(f"전체 트립(30분 분리) {n_trips_all} / 4정류소 순서꼬임 {n_4_misordered} / 완전 궤적 {N}")
print(f"  참고: 시작일 <= 08-12 완전 궤적 {int(np.sum(g_date <= date(2026,8,12)))}건 (8월 분석 281건과 대조)")
mon = defaultdict(int)
for d in g_date: mon[d.strftime("%Y-%m")] += 1
print("  월별 완전 궤적:", dict(sorted(mon.items())))
tight = h <= 15
peak = np.isin(h_hour, (17, 18))
peak_g = np.isin(g_hour, (17, 18))
print(f"\n홍대 15석 이하 트립: {int(tight.sum())}건 (8월 34건)")
print(f"17-18시 트립 (홍대 도착 시각 기준): {int(peak.sum())}건")
print(f"  [대안정의] 공덕 통과 시각 기준 17-18시: {int(peak_g.sum())}건")
print(f"17-18시 AND 홍대 15석 이하: {int((peak & tight).sum())}건  "
      f"(17-18시 중 {100*(peak & tight).sum()/max(peak.sum(),1):.1f}%, 그 외 시간대 {100*((~peak)&tight).sum()/max((~peak).sum(),1):.1f}%)")

# ---------------- 2단계 ----------------
print("\n" + "=" * 72)
print("[2단계] 전체 재분석 (8월 대비)")
print("=" * 72)
print("\n[E-1] 구간별 좌석 감소 (앞 정류소 - 뒤 정류소)   | 8월 평균: 1 / 6.6 / 12.7")
for lab, a, b in (("공덕→신촌", g, s), ("신촌→홍대", s, h), ("홍대→합정", h, j)):
    d = a - b
    print(f"  {lab}: 평균 {d.mean():.2f} 중앙 {np.median(d):.1f} p25 {np.percentile(d,25):.1f} "
          f"p75 {np.percentile(d,75):.1f} 최대 {int(d.max())} 증가비율 {100*np.mean(d<0):.1f}%")

r = np.corrcoef(s, h)[0, 1]
print(f"\n[E-3] 신촌-홍대 피어슨 r = {r:.3f} (n={N})   | 8월 0.831")

X1 = g.reshape(-1, 1)
def cv_pred(X, y, seed):
    return cross_val_predict(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=seed))

p = cv_pred(X1, h, 42)
err = np.abs(h - p)
print(f"\n[모델1 공덕만] 5-fold CV(seed42): MAE {err.mean():.2f}  RMSE {np.sqrt(mean_squared_error(h,p)):.2f}  "
      f"R2 {r2_score(h,p):.3f}   | 8월 MAE 4.00 R2 0.567")
low_mae = err[tight].mean()
print(f"  홍대 15석 이하 MAE {low_mae:.2f} (n={int(tight.sum())})   | 8월 11.33")
seeds = np.arange(20)
lows = [np.abs(h - cv_pred(X1, h, sd))[tight].mean() for sd in seeds]
alls = [np.abs(h - cv_pred(X1, h, sd)).mean() for sd in seeds]
print(f"  20개 시드 평균: 전체 MAE {np.mean(alls):.2f}±{np.std(alls):.2f}, 15석 이하 MAE {np.mean(lows):.2f}±{np.std(lows):.2f}")
m = LinearRegression().fit(X1, h)
print(f"  전체 적합: 홍대 = {m.intercept_:.2f} + {m.coef_[0]:.3f} x 공덕")

BINS = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 99)]
def bin_table(y, e, title):
    print(f"  {title}")
    print(f"    {'홍대실제':8}{'n':>5}{'MAE':>8}{'평균오차(예측-실제)':>20}")
    return
print("\n  홍대 실제 좌석 구간별 MAE (모델1)  [예측-실제 > 0 이면 과대예측]")
print(f"    {'구간':8}{'n':>5}{'MAE':>8}{'bias':>8}")
for lo, hi in BINS:
    mk = (h >= lo) & (h <= hi)
    if mk.sum():
        print(f"    {lo:2}-{hi:<5}{int(mk.sum()):5}{err[mk].mean():8.2f}{(p[mk]-h[mk]).mean():8.2f}")
print("\n  공덕 좌석 구간별 MAE (모델1)")
for lo, hi in [(0, 20), (21, 30), (31, 40), (41, 99)]:
    mk = (g >= lo) & (g <= hi)
    if mk.sum():
        print(f"    공덕 {lo:2}-{hi:<3}{int(mk.sum()):5}{err[mk].mean():8.2f}")

# ---------------- 3단계 ----------------
print("\n" + "=" * 72)
print("[3단계] 17-18시 특화 모델 (홍대 도착 17-18시 트립)")
print("=" * 72)
idx = np.where(peak)[0]
n = len(idx)
print(f"표본 n = {n}")
if n <= 30:
    print("표본 부족 (<=30) -> 중단"); raise SystemExit
if n < 50:
    print("주의: 31~49 경계 구간 -> 참고용 결과")

Xs, ys = g[idx].reshape(-1, 1), h[idx]
def run(seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    pA, pC, pM = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in kf.split(Xs):
        pA[te] = LinearRegression().fit(Xs[tr], ys[tr]).predict(Xs[te])
        mask = np.ones(N, bool); mask[idx[te]] = False
        pC[te] = LinearRegression().fit(X1[mask], h[mask]).predict(Xs[te])
        pM[te] = DummyRegressor().fit(Xs[tr], ys[tr]).predict(Xs[te])
    return pA, pC, pM

pA, pC, pM = run(42)
eA, eC, eM = np.abs(ys - pA), np.abs(ys - pC), np.abs(ys - pM)
print(f"\n전체 (5-fold, seed42; A=17-18시 특화, C=전체모델 대조군[같은 held-out], M=상수(평균) 기준선)")
for nm, pp, ee in (("A 특화", pA, eA), ("C 전체모델", pC, eC), ("M 평균상수", pM, eM)):
    print(f"  {nm:10} MAE {ee.mean():.2f}  RMSE {np.sqrt(np.mean((ys-pp)**2)):.2f}  R2 {r2_score(ys,pp):.3f}")
mA = LinearRegression().fit(Xs, ys)
print(f"  특화 적합: 홍대 = {mA.intercept_:.2f} + {mA.coef_[0]:.3f} x 공덕  | 전체모델: {m.intercept_:.2f} + {m.coef_[0]:.3f} x 공덕")

print("\n홍대 실제 좌석 구간별 MAE (A 특화 vs C 전체)")
print(f"    {'구간':8}{'n':>5}{'A':>8}{'C':>8}{'bias A':>8}{'bias C':>8}")
for lo, hi in BINS:
    mk = (ys >= lo) & (ys <= hi)
    if mk.sum():
        print(f"    {lo:2}-{hi:<5}{int(mk.sum()):5}{eA[mk].mean():8.2f}{eC[mk].mean():8.2f}"
              f"{(pA[mk]-ys[mk]).mean():8.2f}{(pC[mk]-ys[mk]).mean():8.2f}")

def paired(mask, label):
    k = int(mask.sum())
    if k < 5:
        print(f"  {label}: n={k} (표본 너무 적음)"); return
    d = eC[mask] - eA[mask]  # >0 이면 특화가 더 정확
    rng = np.random.default_rng(0)
    bs = [rng.choice(d, k, replace=True).mean() for _ in range(5000)]
    try:
        pw = wilcoxon(d).pvalue
    except ValueError:
        pw = float("nan")
    print(f"  {label}: n={k}  MAE A {eA[mask].mean():.2f} / C {eC[mask].mean():.2f} / 상수 {eM[mask].mean():.2f}  "
          f"C-A = {d.mean():+.2f} (95%CI {np.percentile(bs,2.5):+.2f}~{np.percentile(bs,97.5):+.2f}, Wilcoxon p={pw:.3f})")

print("\n빡빡한 상황 비교 (C-A > 0 이면 특화가 더 정확)")
paired(ys <= 15, "홍대 실제 <=15석 (결과 조건부)")
paired(ys <= 10, "홍대 실제 <=10석 (결과 조건부)")
paired(Xs[:, 0] <= 30, "공덕 <=30석 (입력 조건부, 실제 운영 가능한 정의)")
paired(Xs[:, 0] <= 20, "공덕 <=20석 (입력 조건부)")

lab = ys <= 15
if 0 < lab.sum() < n:
    print(f"\n빡빡(홍대<=15) 탐지력 AUC (예측좌석이 낮을수록 빡빡 판정, 양성 {int(lab.sum())}/{n}):"
          f" A {roc_auc_score(lab, -pA):.3f} / C {roc_auc_score(lab, -pC):.3f}")

dl, dh = [], []
for sd in seeds:
    a, c, _ = run(int(sd))
    ea, ec = np.abs(ys - a), np.abs(ys - c)
    dl.append((ec[ys <= 15].mean() - ea[ys <= 15].mean(), ec[Xs[:, 0] <= 30].mean() - ea[Xs[:, 0] <= 30].mean(),
               ec.mean() - ea.mean()))
dl = np.array(dl)
print(f"\n분할 시드 20개 반복 (C-A 평균±표준편차): 홍대<=15 {dl[:,0].mean():+.2f}±{dl[:,0].std():.2f} | "
      f"공덕<=30 {dl[:,1].mean():+.2f}±{dl[:,1].std():.2f} | 전체 {dl[:,2].mean():+.2f}±{dl[:,2].std():.2f}")
