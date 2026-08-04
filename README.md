# LIBERO-Plus + OpenVLA-OFT + Colab

Windows/VS Code에서 LIBERO-Plus MuJoCo 시뮬레이션을 실행하고, 공식 OpenVLA-OFT
Spatial 체크포인트의 추론만 Google Colab GPU에 맡기는 재현 가능한 실험 환경입니다.

이 저장소에는 모델 가중치, 약 6.4 GB의 LIBERO-Plus assets, 실행 영상이 포함되지
않습니다. 설치 스크립트가 공식 upstream과 assets를 내려받습니다.

## 재현 범위

- checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial`
- checkpoint revision: `6d0231af0e48c5985f1ff86908f4674b84bc049b`
- OpenVLA-OFT source: `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO-Plus source: `4976dc30028e805ff8094b55501d532c48fec182`
- Transformers-OFT fork: `bc339d9ad707454c0c115970db43c260067c61ab`
- external RGB → wrist RGB → proprioception 8D
- 90% center crop, 224×224 policy input
- `8 × 7` action chunk, `libero_spatial_no_noops`
- 10 stabilization steps, Spatial horizon 220
- 태스크당 1 episode, 공식 `task.language` 및 init-state 사용

기본 hard sample은 LIBERO-Plus 분류 ID 270(zero-based task ID 269),
`Robot Initial States`, difficulty 5입니다.

## 요구사항

- Windows 11
- WSL2와 `Ubuntu-22.04`
- VS Code 권장
- Google Colab 계정과 GPU runtime 사용 권한
- assets와 캐시를 위한 최소 15 GB 여유 공간

WSL이 없다면 관리자 PowerShell에서 먼저 실행하고 재부팅합니다.

```powershell
wsl --install -d Ubuntu-22.04
```

## 설치

```powershell
git clone https://github.com/choi0312/libero-plus-openvla-oft-colab.git
cd libero-plus-openvla-oft-colab
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

설치 과정은 pinned upstream 두 개를 `third_party/`에 clone하고, WSL Python 환경,
Google Colab CLI, LIBERO-Plus assets를 설치합니다. assets 다운로드는 오래 걸릴 수
있으며 중단 후 다시 실행하면 이어받습니다.

## Colab 정책 준비

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\libero_plus_oft\setup_colab_a100.ps1
```

Google 로그인이 처음 필요하면 터미널에 표시되는 URL/브라우저 인증을 완료합니다.
스크립트는 Colab 세션을 만들고 공식 체크포인트와 OFT head를 BF16으로 로드한 뒤
checkpoint/source/input contract를 검증합니다.

## 공식 hard sample 실행

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\libero_plus_oft\run_official_failure_sample.ps1
```

WSLg 창에서 외부 카메라, wrist 카메라, 현재 step/action/success를 확인할 수 있습니다.
결과는 `libero_plus_oft/output/<run-id>/` 아래에 저장됩니다.

## 30분 idle 시 GPU 자동 종료

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\libero_plus_oft\start_idle_gpu_guard.ps1
```

guard는 1분마다 Colab 및 로컬 runner 상태를 확인합니다. episode, 추론, 모델 로딩이
시작되면 타이머를 초기화하고, 연속 30분 동안 아무 작업도 없으면 `openvla` Colab
세션을 종료합니다. 상태는 `libero_plus_oft/idle_gpu_guard_status.json`에 기록됩니다.

## VS Code prompt watcher

1. 저장소 폴더를 VS Code로 엽니다.
2. `Ctrl+Shift+P` → `Tasks: Run Task`를 선택합니다.
3. `LIBERO-Plus A100: Start prompt watcher`를 실행합니다.
4. 열린 `libero_plus_oft/prompt.txt`를 편집하고 저장합니다.

prompt watcher는 비교 실험용 고정 task ID 988을 사용합니다. 공식 benchmark language를
사용하는 hard sample 재현은 별도 task를 실행하세요.

## 출력

- `status.json`: 최신 step, action, success, checkpoint/runtime 검증 정보
- `events.jsonl`: episode 전체 action 및 event
- `live_external_wrist.mp4`: 외부/wrist 카메라와 상태 overlay 영상
- `result.json`: 선택적으로 보존한 요약 결과

## 완전 재현에 관한 주의

핵심 policy rollout 설정은 공식 코드와 맞췄지만, 이 저장소의 기본 로컬 호환 환경은
Robosuite 1.4.1을 사용합니다. 논문 전체 수치를 재현하려면 한 샘플이 아니라
LIBERO-Plus 전체 태스크를 각각 1회 평가해야 합니다. GPU 종류와 렌더링 드라이버 차이로
bitwise 동일성은 보장하지 않습니다.

## 라이선스

이 저장소의 wrapper 코드는 MIT License입니다. 내려받는 OpenVLA-OFT,
LIBERO-Plus, 모델 및 assets에는 각각의 upstream 라이선스가 적용됩니다.
