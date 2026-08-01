# `libero_object` task 5 제외 — 근거와 방침

> **이 결정은 확장 수집을 시작하기 전에 확정되었습니다.** 결과를 보고 조건을 고른
> 것이 아닙니다. 아래 검증은 정책을 한 번도 실행하지 않고, 환경 두 개에 같은 시작
> 상태를 적용해 좌표만 비교한 것입니다. 성공률이나 궤적은 판단 근거에 들어가지
> 않았습니다.

## 결정

`libero_object` task 5 — *"pick up the tomato sauce and place it in the basket"* —
를 **y축 교란 크기 곡선의 주 분석과 확장 수집 대상에서 전면 제외**한다.

주 분석은 모든 교란 크기에서 **동일한 9개 task**(`0,1,2,3,4,6,7,8,9`)를 사용한다.

## 무엇이 관찰되었나

`y0.2`와 `y0.3`에서 source인 `tomato_sauce_1`만 이동하는 것이 아니라, 배경 물체인
`chocolate_pudding_1`도 약 10 m 밖으로 이동한다.

| 조건 | `tomato_sauce_1` | `chocolate_pudding_1` | change class |
|---|---:|---:|---|
| vanilla | — | — | `no_detected_change` |
| `y0.1` | 0.070 m | 0 | `clean_source_only` |
| **`y0.2`** | 0.140 m | **10.0 m** | `source_and_distractor` |
| **`y0.3`** | 0.210 m | **10.0 m** | `source_and_distractor` |

5개 시작 상태(`init_state_id` 0–4) 전부에서 동일하게 재현된다. 나머지 9개 task는
모든 조건·모든 시작 상태에서 clean이다 (135/135 쌍).

## 왜 제외해야 하는가

task 5를 포함하면 그 조건에서의 실패 원인이 둘로 갈린다:

* 토마토 소스가 14 cm 옮겨져서인가
* 초코 푸딩이 장면에서 사라져서인가

구분할 수 없다. 다른 9개 task는 **집을 물건만** 움직이므로 이 모호함이 없다.

부분 제외(예: task 5의 `y0.1`만 포함)는 하지 않는다. 그렇게 하면 곡선의 task
구성이 교란 크기마다 달라져, 곡선의 하강이 정책 때문인지 task 구성이 바뀐 탓인지
구분할 수 없게 된다.

## 원인 검증 — 우리 코드나 로컬 파일 문제가 아니다

제외를 확정하기 전에 여섯 가지 가설을 각각 확인했다.

**로컬 수정 / 다운로드 문제 — 배제.**
`Zxy-MLlab/LIBERO-PRO`를 별도 경로에 새로 clone하여 대조한 결과, task 5의 네 BDDL이
바이트 동일하다.

| 파일 | sha256 (앞 16자) |
|---|---|
| `libero_object/…tomato_sauce….bddl` | `b1f4bb69d256a05f` |
| `libero_object_temp_y0.1/…` | `3770dd926fdc1ec8` |
| `libero_object_temp_y0.2/…` | `5c8357fd0c18156c` |
| `libero_object_temp_y0.3/…` | `c44785b8ca10d74e` |

`bddl_files`와 `init_files` 전체에 대한 `diff -r -q`가 비어 있다. 우리 저장소의 git
이력이 해당 경로를 건드린 적은 최초 일괄 import(`f314390`) 뿐이다. 해당 자산은
upstream에 `f9725f9`(2025-10-31)로 처음 들어온 이후 변경된 적이 없다.

**우리 수집 코드 / 환경 로더 문제 — 배제.**
`tools/openvla`를 전혀 import하지 않는 독립 로더로, `OffScreenRenderEnv`를 BDDL
경로에서 직접 만들고 `env.step()`을 **한 번도 호출하지 않고** 원시 MuJoCo 상태를
읽었을 때 같은 값이 재현된다. 결정적으로, **시작 상태를 적용하지 않은 맨 reset에서도**
`chocolate_pudding_1_main`이 이미 `[0.15, 10.03, 0.035]`에 있다 — 즉 원인은 BDDL의
영역 정의이지 시작 상태 파일이 아니다.

파일 혼동 가설도 배제된다. vanilla 시작 상태를 `y0.2` 환경에 적용하면 초코 푸딩이
0.03으로 돌아오지만 **토마토 소스도 함께** −0.10으로 돌아가, 우리가 관측한 0.14 m
이동 자체가 사라진다.

**자산 내용 — 런타임이 아니라 파일에 하드코딩.**
`other_object_region_3`(초코 푸딩이 놓이는 영역)의 y 범위:

```
vanilla :  0.0049999999999999975 ~ 0.055
y0.1    :  0.0049999999999999975 ~ 0.055     (동일)
y0.2    : 10.0049999999999999975 ~ 10.055    (+10.0)
y0.3    : 10.0049999999999999975 ~ 10.055    (+10.0)
```

10개 temp 디렉터리 전체 102개 이탈 좌표가 예외 없이 **vanilla 십진 문자열의 부호 뒤에
문자 `1`을 삽입한 형태**다. 부동소수 덧셈이었다면 `0.0049999999999999975 + 10.0`의
repr이 `10.005`로 뭉개지는데, vanilla 파일에 남아 있는 mantissa 잡음이 그대로
보존되어 있다.

## 판정: 공식 자산에 존재하나 의도가 확인되지 않음

**"공식 자산 오류"라고 단정하지 않는다.** 편집이 계통적이기 때문이다 — 교란 크기에
단조 증가하고(x: 0→3→9→10→10, y: 0→1→1→6→9), 항상 교란과 같은 축에 부호를 보존하며,
대상은 target의 새 위치 근처 distractor다. 무작위 손상이 이런 성질을 가질 수 없다.

**"의도된 동작"이라고도 단정하지 않는다.** README 전문, 논문 PDF, 프로젝트 웹페이지,
HuggingFace 카드, upstream 커밋 메시지, GitHub 이슈 12건 어디에도 언급이 없다. 오히려
문서가 말하는 곳마다 반대를 가리킨다 — 논문 §2.3은 "**the** manipulated object is
displaced"(단수), README 표는 "Object translation along the X-axis"(단수), 논문 §4.1의
형식화 `e(R) = (W, O, R₀(R))`는 물체 집합 `O`를 불변으로 두고, 유계 제약
`dist_k(x, x⁽ᵏ⁾) ≤ δ_k`는 10 m 이동과 양립하지 않는다.

따라서 문서에는 다음과 같이 표현한다:

> 공식 LIBERO-PRO 자산에서 문서화되지 않은 추가 물체 변경이 발견되어, 단일 source
> 위치 교란이라는 통제 조건을 만족하지 못하는 task 5를 주 분석에서 제외하였다.

## 데이터 보관 방침

**아무것도 삭제하지 않는다.**

| 대상 | 처리 |
|---|---|
| pilot v3 30 episodes (`pilot_v2_9e381f0`) | 그대로 보존 |
| 기존 `x0.1` 데이터 | 그대로 보존. x축과 y축은 기본 분석에서 섞지 않는다 |
| task 5 관련 기존·향후 데이터 | `contaminated_auxiliary` 라벨로 별도 보관 |
| 공식 LIBERO-PRO BDDL 자산 | **수정하지 않는다** |

task 5는 **y축 주 분석에서만** 제외된다. 보조 데이터로서는 유효하며, 특히 "distractor
제거가 정책에 얼마나 영향을 주는가"를 따로 묻고 싶을 때 쓸 수 있다.

## 다른 조건에도 적용되어야 할 것

같은 검사를 통과하지 않은 조건은 쓰지 않는다. 현재까지 측정된 오염 현황:

| 조건 | clean task 수 |
|---|---|
| `y0.1` | **10/10** |
| `y0.2`, `y0.3` | **9/10** (task 5 제외 시 9/9) |
| `y0.4` | 4/10 |
| `y0.5` | 1/10 |
| `x0.2`–`x0.5` | 대부분 오염 |

또한 조사 과정에서 자산 자체의 산술 오류 두 건이 발견되었다. 해당 조건을 쓸 경우
별도 확인이 필요하다:

* `y0.5` `pick_up_the_cream_cheese…` — target 영역의 `y_min`(0.225)이 `y_max`(0.205)보다
  커서 구간이 역전되어 있다
* `x0.5` `pick_up_the_butter…` — target 영역 x 폭이 0.05 m여야 하는데 0.345 m다

## 재현

```bash
# 자산이 upstream과 동일한지
git clone --depth 50 https://github.com/Zxy-MLlab/LIBERO-PRO.git /tmp/up
diff -r -q LIBERO-PRO/libero/libero/bddl_files /tmp/up/libero/libero/bddl_files

# 좌표를 직접 눈으로
grep -A 4 "other_object_region_3" \
  LIBERO-PRO/libero/libero/bddl_files/libero_object_temp_y0.2/pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl

# 10개 task x 3조건 x 5개 시작 상태 전수 검증
python tools/openvla/verify_condition_cleanliness.py --suite libero_object \
  --conditions y0.1,y0.2,y0.3 --tasks 0-9 --init_state_id 0 \
  --out_dir docs/condition_verification
```

결과: `docs/condition_verification/` (csv · md · json, `init_0`–`init_4`)
