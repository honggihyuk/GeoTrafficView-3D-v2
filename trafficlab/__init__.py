"""TrafficLab-3D 엔진 이식본 (GeoTrafficView-3D).

TrafficLab-3D(yuk068)의 비-GUI 코어를 이식:
  - projection.g_projection : CCTV↔지면 양방향 투영(왜곡보정→호모그래피→시차) + 3D 리프팅
  - motion.kinematics       : 속도/방향 스무딩(TrackSmoother)
  - inference.pipeline      : 탐지→추적→투영→기구학→3D → .json.gz
  - io                      : G-Projection 스키마 / gzip replay writer·loader

원본과의 차이(GIS 적응):
  - SAT 좌표계 = 위성 이미지 픽셀 대신 **로컬 미터 평면**(UTM52N 원점 기준, px_per_meter=1.0).
  - 시각화는 PyQt5 대신 MapLibre 웹맵(webmap/)에서 .json.gz 를 재생.
  - G-Projection JSON 에 world 블록(epsg/원점/지면고) 추가.
"""
