# PPO Drone Showcase

학습 및 평가 코드를 제외하고 전시 실행에 필요한 파일만 포함한 버전입니다.

## 실행

처음 한 번 [install_dependencies.py](./install_dependencies.py)를 실행해 필요한 패키지를 설치합니다.
VS Code에서 파일을 연 뒤 오른쪽 위의 실행 버튼을 누르면 됩니다.

설치가 끝나면 프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
python .\run_showcase.py
```

## 동작

- NW / NE / SW / SE / KWU 프리셋 중 하나를 선택합니다.
- 시작점은 하단 150px, 타겟은 상단 150px의 안전한 빈 공간에 무작위로 생성됩니다.
- 모든 드론이 도착하면 1초 뒤 Localization이 자동으로 실행됩니다.

## 구조

```text
Showcase_PPO/
├─ run_showcase.py
├─ install_dependencies.py
├─ showcase/          # 실행 코드
└─ assets/
   ├─ models/         # PPO, ANN, AMC 모델
   ├─ maps/           # 5개 프리셋 맵
   └─ data/           # RSSI 및 AMC 추론 데이터
```
