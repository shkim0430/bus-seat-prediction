import os

from dotenv import load_dotenv

load_dotenv()

GBIS_API_KEY = os.environ["GBIS_API_KEY"]

# 수집 대상: M6117 김포(구래환승센터) 방면 경유 정류소
# 각 항목은 특정 노선 도착정보(getBusArrivalItemv2) 호출에 필요한
# (정류소ID, 노선ID, staOrder) 조합. 호출 절약을 위해 노선 단위로 콕 찍어 조회한다.
#
# station_name 은 사람이 알아보기 위한 메모용(수집에는 미사용).
TARGETS: list[dict] = [
    {"station_id": "113000088", "route_id": "232000071", "sta_order": 31,
     "station_name": "공덕역2번출구"},
    {"station_id": "113000416", "route_id": "232000071", "sta_order": 32,
     "station_name": "신촌오거리.현대백화점(중)"},
    {"station_id": "113000414", "route_id": "232000071", "sta_order": 33,
     "station_name": "홍대입구역(중)"},
    {"station_id": "113000412", "route_id": "232000071", "sta_order": 36,
     "station_name": "합정역(중)"},
]
