### 주제: "산불 확산 패턴을 통한 피해 면적 예측 프로젝트"
# 핵심 키워드 정리
# 산불 확산 패턴 - 산불의 이동 및 확산 경로 분석
# 피해 면적 예측 - 산불로 인한 최종 피해 면적(km²)을 수치로 예측
# 위성 데이터 - NASA FIRMS(산불)와 POWER API(기상) 활용
# 머신러닝 모델	회귀 분석 기반 피해 예측 모델 개발

### 대략적 일정
## 1일차 (월요일)
# 오전 (4h): NASA FIRMS 데이터 수집 → 4명 분담: 위도/경도, brightness 등 확인 (pandas.read_csv)
#          : SUOMI VIIRS C2(VNP14IMG) + J1 VIIRS C2(VJ114IMG) 데이터 수집 → 위도/경도, FRP 확인.
# 오후 (4h): NASA POWER API 다운로드 → 기온, 풍속, 습도 체크 후 병합 시작 (pd.merge)
#          : NASA POWER API 기상 데이터 병합 → 기온, 풍속 등 체크.
# 화요일 대비 - GeoJSON 데이터 찾아보고 없으면 shapely(시각화에서 활용되는 라이브러리(사실 나도 잘 모름...))으로 다각형 생성 준비
# MODIS C6.1도 과거 참고용으로 다운로드 시작

## 2일차 (화요일)
# 오전 (4h): 데이터 전처리 → 결측치(평균값), 이상치 제거 (IQR)
# 오후 (4h): 탐색적 데이터 분석(EDA) + Folium → 상관관계 분석 (seaborn) 후 VIIRS C2로 산불 위치 맵 생성 (folium.Marker).
# Folium 사용 이유:
# 패턴 파악: 산불 분포를 지도로 시각화해 지역별 확산 경향 조기 확인
# 발표 준비: PPT용 기초 지도를 만들어 5일차 부담 줄임
# GPT 팁! - heatmap으로 brightness 강도 추가하면 패턴 더 선명해짐!

## 3일차 (수요일)
# 오전 (4h): 초기 모델 구축 → 선형 회귀, Ridge, Lasso 학습 (scikit-learn), RMSE/MAE 계산
# 오후 (4h): 피처 엔지니어링 → 풍속습도, 온도강수량 등 추가, 그래프 분석 (Folium 생략)
# 피처 엔지니어링 요약
#  - 데이터를 더 똑똑하게 가공해서 모델 성능 높이는 작업
#  - 풍속*습도, 고온 여부 등 산불 피해와 관련성 높은 변수 생성
#  - VIIRS의 FRP 활용해 피해 강도 변수 만들어보기

# 4일차 (목요일)
# 오전 (4h): 모델 개선 → Random Forest, XGBoost 등 추가 학습 및 튜닝 (GridSearchCV)
# 오후 (4h): 결과 분석 + Folium → VIIRS C2 최적 모델의 예측 피해 면적(km²)을 색상 원으로 맵에 추가 (folium.Circle)
# Folium 사용 이유: 최종 예측 결과를 시각화해 발표용 핵심 자료 완성, 모델별 성능 비교 가능
# 최고 모델의 예측만 빨간색으로 표시하면 시각적 임팩트 추가하기
# MODIS C6.1 예측과 비교 슬라이드 넣으면 깊이 추가

# 5일차 (금요일)
# 오전 (4h): PPT 완성 → Folium(VIIRS C2) 지도 2개(위치, 예측)와 모델 성능 표 삽입.
# 오후 (4h): 완성 되지 못한 부분 보완 및 PPT 발표 대본 및 연습


# PPT 구성
# 표지: "산불 피해 예측 프로젝트" – 팀명
# 소개: 산불 피해의 심각성 (통계 그래프).
# 데이터: NASA FIRMS + POWER API 개요, Folium 산불 위치 맵
# 방법: 전처리 → 다중 회귀 모델 → 검증 다이어그램
# 모델 비교: 선형 회귀, Random Forest 등 RMSE/MAE 표
# 결과: 예측 면적(km²) + Folium 피해 범위 맵 (4일차), 최적 모델 성능
# 결론: 기대 효과와 확장 가능성 (예: 실시간 예측)

###
# SUOMI VIIRS C2: 최신, 고해상도 산불 데이터의 주력
# J1 VIIRS C2   : SUOMI와 짝꿍으로 더 촘촘한 관측
# MODIS C6.1    : 과거를 돌아보는 데 유용한 백업

### SUOMI VIIRS C2 (Suomi NPP VIIRS Collection 2)
# SUOMI VIIRS는 Suomi National Polar-orbiting Partnership (Suomi NPP)라는 위성에 탑재된 VIIRS(Visible Infrared Imaging Radiometer Suite) 센서에서 나온 데이터
# 특징:
#  - 해상도     : 375m (산불 탐지에 더 세밀함).
#  - 관측 시간  : 하루에 오전 1:30쯤과 오후 1:30쯤, 지구를 두 번 커버
#  - 업데이트   : 최신 알고리즘으로 데이터를 재처리해서 더 정확함.
#  - 산불 데이터: NASA FIRMS에서 제공하는 "VNP14IMG"라는 이름으로 산불 위치와 강도(FRP: Fire Radiative Power)를 탐지
#  - 쉽게 말하면: Suomi NPP라는 위성이 찍은 최신 산불 사진첩

# J1 VIIRS C2 (NOAA-20 VIIRS Collection 2)
# J1 VIIRS는 NOAA-20 위성(원래 이름은 JPSS-1)에 탑재된 VIIRS 센서 데이터
# 특징:
#  - 해상도: SUOMI VIIRS와 동일한 375m.
#  - 관측 시간: 오전 2:20쯤과 오후 2:20쯤으로, SUOMI보다 50분 뒤에 지나감.
#  - 목적: SUOMI와 비슷하지만, 더 자주 관측해서 데이터 빈틈을 줄임.
#  - 산불 데이터: "VJ114IMG"라는 이름으로 FIRMS에서 제공. SUOMI와 거의 똑같은 정보를 주지만, 시간 차이 때문에 보완적인 역할.
#  - 쉽게 말하면: NOAA-20 위성이 SUOMI의 동생처럼 뒤따라가며 찍은 산불 사진첩

# MODIS C6.1 (MODIS Collection 6.1)
# MODIS는 Terra와 Aqua라는 두 위성에 탑재된 Moderate Resolution Imaging Spectroradiometer 센서의 데이터야. "C6.1"은 Collection 6.1, 즉 6번째 버전의 업데이트판.
# 특징:
# 해상도: 1km (VIIRS보다 덜 세밀함).
# 관측 시간: Terra는 오전 10:30쯤, Aqua는 오후 1:30쯤 지나감.
# 역사: 2000년대 초반부터 데이터를 쌓아왔으니 더 긴 시간 기록을 볼 수 있음.
# 산불 데이터: "MCD14"라는 이름으로 FIRMS에서 제공. 산불 위치와 FRP를 탐지하지만, 해상도가 낮아서 작은 불은 놓칠 수 있어.
# 쉽게 말하면: 오래된 형님 위성이 찍은 산불 사진첩의 6.1번째 버전이야. 좀 더 흐릿하지만 오래된 이야기를 들려줘!

# 차이점        비교
# 항목          SUOMI VIIRS C2      J1 VIIRS C2         MODIS C6.1
# 위성          Suomi NPP           NOAA-20 (JPSS-1)    Terra & Aqua
# 센서          VIIRS               VIIRS               MODIS
# 해상도        375m                375m                1km
# 관측 시간     1:30 AM/PM          2:20 AM/PM          10:30 AM, 1:30 PM
# 버전          Collection 2        Collection 2        Collection 6.1
# 장점          세밀하고 최신       SUOMI와 보완        긴 시간 기록
# 단점          짧은 역사           짧은 역사           덜 세밀함
# FIRMS 이름    VNP14IMG            VJ114IMG            MCD14

# SUOMI VIIRS C2와 J1 VIIRS C2: 최신이고 해상도가 높아서 산불 위치와 피해 면적 예측에 더 정확. 특히 375m 해상도는 작은 불도 잡아내니까 EDA와 Folium 시각화에 딱!
# MODIS C6.1: 과거 데이터(20년 이상)를 보고 싶을 때 유용해. 예를 들어, "과거 산불 패턴이 지금이랑 얼마나 달라졌나?" 같은 분석에 쓰면 좋음.
# 추천: 프로젝트에선 주로 VIIRS C2(SUOMI + J1)를 쓰고, 필요하면 MODIS C6.1로 과거 비교




### 1. 산불 데이터 (fire_data)
# 필수 데이터 (NASA FIRMS 활용)
# fire_id	    str	        산불 고유 ID
# latitude	    float	    위도
# longitude	    float	    경도
# date	        str	        관측 날짜 (YYYY-MM-DD)
# brightness	float	    화재 밝기 지수 (K)
# confidence	int	        신뢰도 (0~100)
# frp	        float	    화재 방출력 (MW)

### 2. 기상 데이터 (weather_data)
# 필수 데이터 (NASA POWER API 활용)
# latitude	    float	    위도
# longitude	    float	    경도
# date	        str	        기상 데이터 날짜 (YYYY-MM-DD)
# temperature	float	    기온 (°C)
# humidity	    float	    상대 습도 (%)
# wind_speed	float	    풍속 (m/s)
# precipitation	float	    강수량 (mm)

### 3. 산불 확산 경계 데이터 (geojson_data)
# region_name	str	        지역명 (ex: California)
# geometry	    dict	    GeoJSON 형식의 다각형 좌표
# fire_count	int	        해당 지역에서 발생한 산불 수

# 산불 경계 데이터 (GeoJSON)
# GeoJSON을 찾아봐야 할 곳
# Global Fire Atlas: https://www.globalfiredata.org
# NASA FIRMS (GIS Data): https://firms.modaps.eosdis.nasa.gov/gisdata/
# OpenStreetMap Wildfire Data: https://overpass-turbo.eu
# 없으면 산불데이터로 형성 (open ai 질문 결과 shapely 라이브러리 활용, 산불 데이터 (latitude, longitude)를 Convex Hull 알고리즘을 이용해 다각형(Polygon)으로 변환

### 4. 예측 피해 면적 데이터 (pred_data)
# latitude	    float	    예측된 피해 중심 위도
# longitude	    float	    예측된 피해 중심 경도
# pred_area	    float	    예측된 피해 면적 (km²)
# confidence	float	    예측 신뢰도 (0~1)

# 공통 사용 데이터 변수명
# fire_data: NASA FIRMS 데이터
# weather_data: NASA POWER API 기상 데이터
# geojson_data: 산불 확산 경계 데이터 (GeoJSON)
# pred_data: 머신러닝 모델의 예측 결과

### 필요할거로 예상되는 파일 구조
# wildfire_project/
# - data/
#   - fire_data.csv  # NASA FIRMS 데이터
#   - weather_data.csv  # NASA POWER API 데이터
#   - fire_boundary.geojson  # 산불 확산 경계 (GeoJSON) (없으면 위도 경도 직접 활용해서 사용)

# - models/
#   - 우리가 배운 회귀 모델들 사용 후 선정 할 파일
#     (클래스로 만들어서 데이터 프레임으로 하거나 모델별로 파일 생성.) - 상의 후 결정

# - visualizations/
#   - fire_map.html  # Folium 지도 (산불 위치)
#   - fire_heatmap.html  # Folium 히트맵 (산불 강도)

# - main.ipynb  # 분석 코드
# - presentation.pptx  # 최종 발표 자료

### 가나다라