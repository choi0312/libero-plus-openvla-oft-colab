# Runtime scripts

사용법과 재현 설정은 저장소 루트의 [README](../README.md)를 참고하세요.

- `setup_local_simulator.ps1`: WSL 시뮬레이터, pinned source, assets 설치
- `setup_colab_a100.ps1`: Colab 세션 생성 및 검증된 OFT 정책 로드
- `run_official_failure_sample.ps1`: 공식 difficulty-5 초기자세 태스크 1회 실행
- `run_colab_a100.ps1`: 단일 episode runner
- `run_one_episode.py`: MuJoCo rollout, 관측 전처리, live window 및 기록
- `prompt_watcher.ps1`: `prompt.txt` 저장 감지 runner
