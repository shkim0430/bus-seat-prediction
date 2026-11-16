# -*- coding: utf-8 -*-
# 주의: scratchpad 유실로 보고 기준 코드를 재작성한 버전이다. 11-02 결과 재현은
# --until 2026-11-02 (collected_at < '2026-11-02' 필터)로 하고,
# 수치는 원본 실행과 대조되지 않음.
# 사용: BUS_DB_PATH=<bus.db 경로> python prob.py --until YYYY-MM-DD   (DB는 읽기 전용으로만 연다)
"""확률 예측: P(홍대 도착 좌석 <= 15 | 공덕 좌석, 17-18시 여부, 요일).
트립 구성은 기존 방법론 그대로 (08-06 제외, IS NOT NULL, 30분 트립 분리, last-seen 정렬, 완전 궤적)."""
import sqlite3, math
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
STATIONS = {"113000088": 31, "113000416": 32, "113000414": 33, "113000412": 36}
G, H = "113000088", "113000414"

conn = sqlite3.connect(DB, uri=True)
rows = conn.execute(f"""SELECT collected_at, station_id, plate_no1, remain_seat_cnt1 FROM arrivals
    WHERE route_id = 232000071 AND station_id IN ({",".join(STATIONS)})
      AND remain_seat_cnt1 IS NOT NULL AND plate_no1 != ''
      AND substr(collected_at,1,10) != '2026-08-06'
      AND substr(collected_at,1,10) < '{UNTIL}'""").fetchall()
conn.close()
sight = defaultdict(list)
for ca, sid, plate, seat in rows:
    sight[plate].append((datetime.fromisoformat(ca), str(sid), seat))


def split_trips(obs, gap=30):
    obs = sorted(obs, key=lambda x: x[0]); out, cur, prev = [], [], None
    for o in obs:
        if prev is not None and (o[0] - prev).total_seconds() > gap * 60:
            out.append(cur); cur = []
        cur.append(o); prev = o[0]
    if cur: out.append(cur)
    return out


recs = []
for plate, obs in sight.items():
    for trip in split_trips(obs):
        ps = {}
        for dt, sid, seat in trip:
            if sid not in ps or dt > ps[sid][0]: ps[sid] = (dt, seat)
        if len(ps) != 4: continue
        order = sorted(ps.items(), key=lambda x: x[1][0])
        seq = [STATIONS[s] for s, _ in order]
        if not all(seq[i] < seq[i + 1] for i in range(3)): continue
        recs.append((ps[G][0], ps[G][1], ps[H][1]))
recs.sort(key=lambda r: r[0])  # 공덕 통과 시각 순
N = len(recs)
g_dt = [r[0] for r in recs]
g = np.array([r[1] for r in recs], float)
h = np.array([r[2] for r in recs], float)
peak = np.array([d.hour in (17, 18) for d in g_dt], float)   # 예측 시점(공덕 통과)에 알 수 있는 값
dow = np.array([d.weekday() for d in g_dt])
dow_oh = np.zeros((N, 6))
for i, d in enumerate(dow):
    if d > 0: dow_oh[i, d - 1] = 1
g10 = (g / 10).reshape(-1, 1)
pk = peak.reshape(-1, 1)
FEATS = {
    "F1 공덕": g10,
    "F2 공덕+17-18시": np.hstack([g10, pk]),
    "F2x 공덕+17-18시+교호": np.hstack([g10, pk, g10 * pk]),
    "F3 공덕+17-18시+요일": np.hstack([g10, pk, dow_oh]),
}
cut = int(N * 0.8)


def lr():
    return LogisticRegression(C=1e6, max_iter=2000)


def wilson(k, n):
    if n == 0: return (float("nan"),) * 2
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def cv_proba(X, y, seed):
    return cross_val_predict(lr(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]


def boot_delta(y, p_a, p_b, B=1000):
    rng = np.random.default_rng(0); n = len(y); da, db = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        da.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
        db.append(brier_score_loss(y[i], p_b[i]) - brier_score_loss(y[i], p_a[i]))
    f = lambda a: f"{np.mean(a):+.4f} ({np.percentile(a,2.5):+.4f}~{np.percentile(a,97.5):+.4f})"
    return f(da), f(db)


def calib(y, p, title):
    print(f"\n  보정표: {title}")
    print(f"    {'예측확률':10}{'n':>6}{'평균예측':>9}{'실제15석↓':>11}{'(k/n)':>9}{'  실제 95%CI':>16}")
    xs, ys, ece = [], [], 0.0
    edges = np.arange(0, 1.0001, 0.1)
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (p >= a) & ((p < b) if b < 1 else (p <= b))
        n = int(mk.sum())
        if n == 0: continue
        k = int(y[mk].sum()); lo, hi = wilson(k, n)
        print(f"    {int(a*100):3}-{int(b*100):<3}%{n:8}{p[mk].mean()*100:8.1f}%{100*k/n:10.1f}%{f'({k}/{n})':>10}{f'  {100*lo:.0f}~{100*hi:.0f}%':>16}")
        ece += n * abs(p[mk].mean() - k / n)
        if n >= 10: xs.append(p[mk].mean()); ys.append(k / n)
    rho = spearmanr(xs, ys)[0] if len(xs) >= 3 else float("nan")
    print(f"    ECE(가중 평균 |예측-실제|) = {100*ece/len(y):.2f}%p, n>=10 구간 Spearman = {rho:.2f}")
    return 100 * ece / len(y)


def evaluate(label, y, idx_sets, thr_pos=30):
    """idx_sets: 이 데이터셋의 전역 인덱스 배열(시간순). 랜덤 CV + 시간순 분할 평가."""
    pos = int(y[idx_sets].sum()); n = len(idx_sets)
    print(f"\n--- {label}: n={n}, 양성 {pos} ({100*pos/n:.1f}%) ---")
    if pos < thr_pos:
        print(f"  양성 {pos} < {thr_pos} -> 표본 얇음, n만 보고하고 모델링 중단"); return None
    yy = y[idx_sets]; res = {}
    names = [k for k in FEATS if not (label.startswith("17-18시만") and k in ("F2 공덕+17-18시", "F2x 공덕+17-18시+교호", "F3 공덕+17-18시+요일"))]
    if label.startswith("17-18시만"):
        names = ["F1 공덕", "F1+요일"]
    base_b = pos / n * (1 - pos / n)
    print(f"  [랜덤 5-fold CV] 상수예측 Brier {base_b:.4f}")
    print(f"    {'피처':24}{'AUC':>8}{'(10시드평균)':>12}{'Brier':>9}{'BSS':>8}")
    for nm in names:
        X = FEATS[nm][idx_sets] if nm in FEATS else np.hstack([g10, dow_oh])[idx_sets]
        p = cv_proba(X, yy, 42); res[nm] = (X, p)
        aucs = [roc_auc_score(yy, cv_proba(X, yy, s)) for s in range(10)]
        b = brier_score_loss(yy, p)
        print(f"    {nm:24}{roc_auc_score(yy,p):8.3f}{np.mean(aucs):12.3f}{b:9.4f}{1-b/base_b:8.3f}")
    ns = list(res)
    for a, b in zip(ns[:-1], ns[1:]):
        dA, dB = boot_delta(yy, res[a][1], res[b][1])
        print(f"    {a} -> {b}: dAUC {dA} / dBrier {dB}")
    # 시간순 분할: 해당 데이터셋 내부 시간순 앞 80% / 뒤 20%
    c = int(n * 0.8); tr, te = idx_sets[:c], idx_sets[c:]
    pte = int(y[te].sum())
    print(f"  [시간순 분할] 학습 {len(tr)}건(양성률 {100*y[tr].mean():.1f}%, ~{g_dt[tr[-1]].date()}) / "
          f"테스트 {len(te)}건(양성률 {100*y[te].mean():.1f}%, {g_dt[te[0]].date()}~, 양성 {pte})")
    if pte < 15:
        print(f"    테스트 양성 {pte} < 15 -> n만 보고"); return res
    base_t = y[tr].mean() * (1 - y[tr].mean())
    bt_const = brier_score_loss(y[te], np.full(len(te), y[tr].mean()))
    print(f"    {'피처':24}{'AUC':>8}{'Brier':>9}{'BSS(vs 학습양성률)':>20}")
    tsres = {}
    for nm in names:
        Xall = FEATS[nm] if nm in FEATS else np.hstack([g10, dow_oh])
        m = lr().fit(Xall[tr], y[tr]); p = m.predict_proba(Xall[te])[:, 1]; tsres[nm] = p
        b = brier_score_loss(y[te], p)
        print(f"    {nm:24}{roc_auc_score(y[te],p):8.3f}{b:9.4f}{1-b/bt_const:20.3f}")
    res["_ts"] = (y[te], tsres)
    return res


def table(y, thr_label):
    bins = [(0, 25), (26, 30), (31, 35), (36, 40), (41, 99)]
    print(f"\n  룩업 테이블: 공덕 좌석 x 17-18시 -> 홍대 {thr_label} 실제 비율 (k/n, Wilson 95%CI)")
    print(f"    {'공덕좌석':10}{'17-18시':>26}{'그 외 시간대':>26}{'전체':>18}")
    def cell(mk):
        n = int(mk.sum()); k = int(y[mk].sum())
        if n == 0: return "-"
        lo, hi = wilson(k, n); flag = "*" if k < 5 else ""
        return f"{100*k/n:4.1f}% ({k}/{n}){flag} [{100*lo:.0f}~{100*hi:.0f}]"
    for lo, hi in bins:
        mg = (g >= lo) & (g <= hi)
        print(f"    {lo:2}-{hi:<7}{cell(mg&(peak==1)):>26}{cell(mg&(peak==0)):>26}{cell(mg):>18}")
    print("    (* = 양성 5건 미만, 신뢰 불가)")


allidx = np.arange(N)
pk_idx = np.where(peak == 1)[0]
print("=" * 78)
print(f"데이터: 완전 궤적 {N}트립 ({g_dt[0].date()} ~ {g_dt[-1].date()}), 17-18시(공덕 시각 기준) {len(pk_idx)}건")
print(f"  *제안한 536건은 홍대 도착 시각 기준; 여기선 예측 시점에 알 수 있는 공덕 시각 기준 사용")
print("=" * 78)

for thr in (15, 10):
    y = (h <= thr).astype(int)
    print("\n" + "#" * 78)
    print(f"#  타깃: 홍대 좌석 <= {thr}   (전체 양성 {int(y.sum())}/{N} = {100*y.mean():.1f}%, "
          f"17-18시 양성 {int(y[pk_idx].sum())}/{len(pk_idx)})")
    print("#" * 78)
    if thr == 10 and y.sum() < 30:
        print("양성 30 미만 -> 중단"); continue
    print("\n[1] 로지스틱 회귀")
    r_all = evaluate("전체 트립", y, allidx)
    r_pk = evaluate("17-18시만", y, pk_idx)
    print("\n[2] 보정 확인")
    if r_all:
        calib(y, r_all["F2 공덕+17-18시"][1], "전체 트립 F2, 랜덤 5-fold CV")
        best = min((k for k in r_all if k != "_ts"), key=lambda k: brier_score_loss(y, r_all[k][1]))
        if best != "F2 공덕+17-18시":
            calib(y, r_all[best][1], f"전체 트립 {best} (CV Brier 최소), 랜덤 5-fold CV")
        if "_ts" in r_all:
            yt, tp = r_all["_ts"]
            calib(yt, tp["F2 공덕+17-18시"], "전체 트립 F2, 시간순 분할 테스트(미래)")
            calib(yt, tp["F3 공덕+17-18시+요일"], "전체 트립 F3, 시간순 분할 테스트(미래)")
            import io, contextlib
            def _q(*a):
                with contextlib.redirect_stdout(io.StringIO()):
                    return calib(*a)
            print(f"\n  [F2 vs F3 비교, 전체 트립, 타깃 <={thr}] (시간순 테스트 n={len(yt)}, 양성 {int(yt.sum())})")
            print(f"    {'모델':22}{'랜덤 AUC':>10}{'랜덤 Brier':>11}{'랜덤 ECE%p':>11}{'시간순 AUC':>11}{'시간순 Brier':>12}{'시간순 ECE%p':>12}")
            for nm in ("F2 공덕+17-18시", "F3 공덕+17-18시+요일"):
                p_r = r_all[nm][1]; p_t = tp[nm]
                print(f"    {nm:24}{roc_auc_score(y,p_r):10.3f}{brier_score_loss(y,p_r):11.4f}{_q(y,p_r,nm):11.2f}"
                      f"{roc_auc_score(yt,p_t):11.3f}{brier_score_loss(yt,p_t):12.4f}{_q(yt,p_t,nm):12.2f}")
    if r_pk:
        calib(y[pk_idx], r_pk["F1 공덕"][1], "17-18시만 F1, 랜덤 5-fold CV")
    print("\n[3] 서비스 시나리오 표" if thr == 15 else "\n[4] 서비스 시나리오 표 (<=10)")
    table(y, f"{thr}석 이하")
