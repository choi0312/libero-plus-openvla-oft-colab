# LIBERO-Plus 공식 취약 태스크 재현

이 실행은 사용자가 임의로 만든 실패 프롬프트가 아니라 LIBERO-Plus의 공식 태스크 정의와 분류표를 사용한다.

- suite: `libero_spatial`
- classification ID: `270` (1-based)
- benchmark index / runner `TaskId`: `269` (0-based)
- category: `Robot Initial States`
- difficulty: `5`
- task: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_231`
- prompt override: 사용하지 않음. 태스크 정의의 `task.language`를 그대로 사용
- episode: 공식 initial state의 episode `0`, 정확히 1회
- horizon: Spatial 공식 설정인 stabilization 10 step + control 220 step
- policy: `moojink/openvla-7b-oft-finetuned-libero-spatial`
- policy input: external RGB, wrist RGB, proprioception 8D 순서
- action: 8 x 7 open-loop chunk
- inference: 연결된 Colab A100, BF16, 비양자화

VS Code에서 `Ctrl+Shift+P` → `Tasks: Run Task` →
`LIBERO-Plus A100: Reproduce official hard init-state task`를 선택한다.

최근 재현 결과는 `status.json`, 개별 step/action은 해당 output 폴더의
`events.jsonl`, 영상은 `live_external_wrist.mp4`에 기록된다.
