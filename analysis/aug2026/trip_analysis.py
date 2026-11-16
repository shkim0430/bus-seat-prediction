# -*- coding: utf-8 -*-
"""M6117 궤적 추적 1단계: 트립 분리 + last-seen 기준 정류소 통과 순서 분석.
VM(bus-data)에서 직접 실행. 30분 이상 공백은 별도 트립으로 분리하는 원칙,
정류소 통과 순서는 first-seen이 아닌 last-seen 기준으로 판정하는 원칙을 그대로 적용.
"""
import sqlite3
from datetime import datetime, date
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
STATION_SEQ = ["113000088", "113000416", "113000414", "113000412"]  # sta_order asc
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
print(f"기초 관측 행 수: {len(rows)}")

sightings = defaultdict(list)  # plate_no -> [(dt, station_id, seat)]
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


# ---- 트립 단위로 정리 ----
all_trips = []  # dict: plate, start_dt, dow, per_station{sid:(last_dt, seat)}, n_stations, ordered, order_bucket
for plate, obs in sightings.items():
    for trip in split_trips(obs):
        per_station = {}
        for dt, sid, seat in trip:
            if sid not in per_station or dt > per_station[sid][0]:
                per_station[sid] = (dt, seat)
        start_dt = trip[0][0]
        ordered_stations = sorted(per_station.items(), key=lambda x: x[1][0])  # by last-seen time
        sta_seq = [STATIONS[s][1] for s, _ in ordered_stations]
        is_ordered = all(sta_seq[i] < sta_seq[i + 1] for i in range(len(sta_seq) - 1))
        n = len(per_station)
        all_trips.append({
            "plate": plate,
            "start_dt": start_dt,
            "dow": DOW_KR[start_dt.weekday()],
            "per_station": per_station,
            "ordered_stations": ordered_stations,
            "n": n,
            "is_ordered": is_ordered,
        })

print(f"총 트립 수: {len(all_trips)}\n")

# ---- [쿼리 A] 트립 통계 ----
print("=" * 70)
print("[쿼리 A] 트립 통계")
print("=" * 70)


def bucket(t):
    if t["n"] == 4 and t["is_ordered"]:
        return "4개_순서대로(완전궤적)"
    if t["n"] == 3 and t["is_ordered"]:
        return "3개_순서대로"
    if t["n"] == 2 and t["is_ordered"]:
        return "2개_순서대로(사각지대)"
    if t["n"] == 1:
        return "1개만"
    return "순서꼬임(이상관측)"


for t in all_trips:
    t["bucket"] = bucket(t)

bucket_counts = defaultdict(int)
for t in all_trips:
    bucket_counts[t["bucket"]] += 1

print(f"{'구분':25} {'건수':>6}")
for b in ["4개_순서대로(완전궤적)", "3개_순서대로", "2개_순서대로(사각지대)", "1개만", "순서꼬임(이상관측)"]:
    print(f"{b:25} {bucket_counts.get(b,0):6}")
print(f"{'합계':25} {sum(bucket_counts.values()):6} (총 트립 수와 일치해야 정상)")

print("\n-- 요일별 breakdown --")
dow_bucket = defaultdict(lambda: defaultdict(int))
for t in all_trips:
    dow_bucket[t["dow"]][t["bucket"]] += 1
    dow_bucket[t["dow"]]["총트립"] += 1

cols = ["총트립", "4개_순서대로(완전궤적)", "3개_순서대로", "2개_순서대로(사각지대)", "1개만", "순서꼬임(이상관측)"]
print(f"{'요일':4}" + "".join(f"{c:>14}" for c in ["총","4개","3개","2개","1개","꼬임"]))
for d in ["월", "화", "수", "목", "금", "토", "일"]:
    row = dow_bucket.get(d, {})
    print(f"{d:4}" + "".join(f"{row.get(c,0):14}" for c in cols))

# ---- [쿼리 B] 완전 궤적 트립의 요일별·시간대별 분포 ----
print("\n" + "=" * 70)
print("[쿼리 B] 완전 궤적(4개 순서대로 통과) 트립의 요일별·시간대별 분포")
print("(시간대 = 공덕역 통과(last-seen) 시각 기준)")
print("=" * 70)

complete_trips = [t for t in all_trips if t["bucket"] == "4개_순서대로(완전궤적)"]
print(f"완전 궤적 총 {len(complete_trips)}건\n")

dow_hour = defaultdict(int)
hour_total = defaultdict(int)
dow_total = defaultdict(int)
for t in complete_trips:
    gongdeok_dt = t["per_station"]["113000088"][0]
    hr = gongdeok_dt.strftime("%H")
    dow_hour[(t["dow"], hr)] += 1
    hour_total[hr] += 1
    dow_total[t["dow"]] += 1

hours = sorted(hour_total.keys())
print(f"{'요일':4}" + "".join(f"{h:>5}" for h in hours) + f"{'합계':>7}")
for d in ["월", "화", "수", "목", "금", "토", "일"]:
    print(f"{d:4}" + "".join(f"{dow_hour.get((d,h),0):5}" for h in hours) + f"{dow_total.get(d,0):7}")
print(f"{'합계':4}" + "".join(f"{hour_total.get(h,0):5}" for h in hours) + f"{len(complete_trips):7}")

print("\n-- 시간대별 합계 정렬(많은 순) --")
for h, c in sorted(hour_total.items(), key=lambda x: -x[1]):
    print(f"  {h}시: {c}건")

# ---- [쿼리 C] 대표 완전 궤적 사례 20개 ----
print("\n" + "=" * 70)
print("[쿼리 C] 대표 완전 궤적 사례 20개 (공덕→합정 좌석 감소량 큰 순)")
print("=" * 70)

records = []
for t in complete_trips:
    ps = t["per_station"]
    g_dt, g_seat = ps["113000088"]
    s_dt, s_seat = ps["113000416"]
    h_dt, h_seat = ps["113000414"]
    j_dt, j_seat = ps["113000412"]
    duration_min = (j_dt - g_dt).total_seconds() / 60
    drop = g_seat - j_seat
    trip_id = f"{t['plate']}_{t['start_dt'].strftime('%Y%m%dT%H%M')}"
    records.append({
        "trip_id": trip_id, "dow": t["dow"],
        "g_dt": g_dt, "g_seat": g_seat,
        "s_dt": s_dt, "s_seat": s_seat,
        "h_dt": h_dt, "h_seat": h_seat,
        "j_dt": j_dt, "j_seat": j_seat,
        "duration_min": duration_min, "drop": drop,
    })

records.sort(key=lambda r: -r["drop"])

hdr = (f"{'trip_id':28}{'요일':4}{'공덕(시각/석)':16}{'신촌(시각/석)':16}"
       f"{'홍대(시각/석)':16}{'합정(시각/석)':16}{'소요(분)':9}{'감소량':6}")
print(hdr)
for r in records[:20]:
    print(
        f"{r['trip_id']:28}{r['dow']:4}"
        f"{r['g_dt'].strftime('%H:%M'):>6}/{r['g_seat']:<7}"
        f"{r['s_dt'].strftime('%H:%M'):>6}/{r['s_seat']:<7}"
        f"{r['h_dt'].strftime('%H:%M'):>6}/{r['h_seat']:<7}"
        f"{r['j_dt'].strftime('%H:%M'):>6}/{r['j_seat']:<7}"
        f"{r['duration_min']:9.1f}{r['drop']:6}"
    )

conn.close()
