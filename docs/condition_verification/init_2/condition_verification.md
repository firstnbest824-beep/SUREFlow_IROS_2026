# 조건 오염 검증 — libero_object

seed 0, init state 2. 정책은 실행하지 않고 vanilla / perturbed 환경에 같은 시작 상태를 적용해 좌표만 비교했습니다.

| task | condition | source displacement | other moved objects | left scene | change class | clean |
|---|---|---:|---|---|---|:--:|
| 0 | `y0.1` | 0.070 m | 없음 | — | `clean_source_only` | ✅ |
| 0 | `y0.2` | 0.140 m | 없음 | — | `clean_source_only` | ✅ |
| 0 | `y0.3` | 0.210 m | 없음 | — | `clean_source_only` | ✅ |
| 1 | `y0.1` | 0.069 m | 없음 | — | `clean_source_only` | ✅ |
| 1 | `y0.2` | 0.139 m | 없음 | — | `clean_source_only` | ✅ |
| 1 | `y0.3` | 0.208 m | 없음 | — | `clean_source_only` | ✅ |
| 2 | `y0.1` | 0.070 m | 없음 | — | `clean_source_only` | ✅ |
| 2 | `y0.2` | 0.140 m | 없음 | — | `clean_source_only` | ✅ |
| 2 | `y0.3` | 0.210 m | 없음 | — | `clean_source_only` | ✅ |
| 3 | `y0.1` | 0.072 m | 없음 | — | `clean_source_only` | ✅ |
| 3 | `y0.2` | 0.144 m | 없음 | — | `clean_source_only` | ✅ |
| 3 | `y0.3` | 0.216 m | 없음 | — | `clean_source_only` | ✅ |
| 4 | `y0.1` | 0.070 m | 없음 | — | `clean_source_only` | ✅ |
| 4 | `y0.2` | 0.140 m | 없음 | — | `clean_source_only` | ✅ |
| 4 | `y0.3` | 0.210 m | 없음 | — | `clean_source_only` | ✅ |
| 5 | `y0.1` | 0.070 m | 없음 | — | `clean_source_only` | ✅ |
| 5 | `y0.2` | 0.140 m | `chocolate_pudding_1` (10.0 m) | `chocolate_pudding_1` | `source_and_distractor` | ❌ |
| 5 | `y0.3` | 0.210 m | `chocolate_pudding_1` (10.0 m) | `chocolate_pudding_1` | `source_and_distractor` | ❌ |
| 6 | `y0.1` | 0.063 m | 없음 | — | `clean_source_only` | ✅ |
| 6 | `y0.2` | 0.133 m | 없음 | — | `clean_source_only` | ✅ |
| 6 | `y0.3` | 0.207 m | 없음 | — | `clean_source_only` | ✅ |
| 7 | `y0.1` | 0.072 m | 없음 | — | `clean_source_only` | ✅ |
| 7 | `y0.2` | 0.138 m | 없음 | — | `clean_source_only` | ✅ |
| 7 | `y0.3` | 0.216 m | 없음 | — | `clean_source_only` | ✅ |
| 8 | `y0.1` | 0.070 m | 없음 | — | `clean_source_only` | ✅ |
| 8 | `y0.2` | 0.140 m | 없음 | — | `clean_source_only` | ✅ |
| 8 | `y0.3` | 0.210 m | 없음 | — | `clean_source_only` | ✅ |
| 9 | `y0.1` | 0.070 m | 없음 | — | `clean_source_only` | ✅ |
| 9 | `y0.2` | 0.140 m | 없음 | — | `clean_source_only` | ✅ |
| 9 | `y0.3` | 0.210 m | 없음 | — | `clean_source_only` | ✅ |

## 조건별 요약

- `y0.1` — clean 10/10 task, 이동량 0.063–0.072 m
- `y0.2` — clean 9/10 task, 오염된 task: [5], 이동량 0.133–0.144 m
- `y0.3` — clean 9/10 task, 오염된 task: [5], 이동량 0.207–0.216 m
