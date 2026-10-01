# TREAD attention 진단: tread-v / 0400000

## 조건

- EMA, SiT-B/2, 400,000 step. 순수 TREAD, RouteSync 없음.
- Routed block: zero-based [3, 4, 5, 6, 7, 8]; 256개 중 128개 active.
- ImageNet **학습 데이터** latent 2개, timestep [0.15, 0.7], mask 1개/이미지. t=0 clean, t=1 noise.
- 동일 이미지·posterior sample·noise·mask로 paired 비교. Conditional CFG=1, FP32, TF32 off.
- 대각선 보정: log(127/255) = -0.69707646. 가중치 업데이트 없음.
- local: 각 dense layer의 Q/K/V를 고정하고 같은 active query/key/value만 선택. full: 실제 sparse 전체 forward. corrected는 모든 routed block에 보정 적용.
- Attention 출력 오차는 output projection 전 각 head의 AV를 동일 위치의 dense AV와 비교. head와 query를 평균하며, 통계 단위는 이미지.

## 첫 routed layer 결과: noise level을 동일 가중 평균

| Mode | Self attention | AV MSE vs dense | AV cosine |
|---|---:|---:|---:|
| dense | 0.015705 | 0.00000000 | 1.000000 |
| local_sparse | 0.030149 | 0.00706123 | 0.967234 |
| local_corrected | 0.016025 | 0.00724371 | 0.966380 |

- 보정의 AV MSE 변화: **+2.584%** (negative=개선).
- Paired MSE 차이 corrected−sparse: 0.00018247927; 이미지 bootstrap 95% CI [0.00017506962, 0.00018988892].

## timestep별 첫 layer

| t | Dense self | Sparse self | Corrected self | Sparse AV MSE | Corrected AV MSE | MSE change |
|---|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.020455 | 0.038967 | 0.020920 | 0.00691979 | 0.00705910 | +2.01% |
| 0.7 | 0.010955 | 0.021330 | 0.011131 | 0.00720266 | 0.00742831 | +3.13% |

## 실제 full forward: layer별 비교

| Block (zero-based) | Dense self | Sparse self | Corrected self | Sparse AV MSE | Corrected AV MSE | MSE change |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.015705 | 0.030149 | 0.016025 | 0.00706123 | 0.00724371 | +2.58% |
| 4 | 0.022396 | 0.044163 | 0.024223 | 0.01681664 | 0.01770196 | +5.26% |
| 5 | 0.027303 | 0.050282 | 0.028802 | 0.01346261 | 0.01472982 | +9.41% |
| 6 | 0.025340 | 0.045941 | 0.026141 | 0.01161818 | 0.01285205 | +10.62% |
| 7 | 0.016915 | 0.031903 | 0.017522 | 0.00858555 | 0.01007605 | +17.36% |
| 8 | 0.025502 | 0.047180 | 0.026450 | 0.01090677 | 0.01275561 | +16.95% |

## 최종 velocity: 보정을 학습 없이 적용한 영향

| t | Dense FM MSE | Sparse FM MSE | Corrected FM MSE | FM MSE change |
|---|---:|---:|---:|---:|
| 0.15 | 0.797042 | 0.797844 | 0.798253 | +0.05% |
| 0.7 | 0.536727 | 0.531934 | 0.531182 | -0.14% |

## 해석과 한계

- Sparse self mass 증가는 key 수 감소만으로도 생긴다. 보정의 타당성은 self mass와 함께 AV 출력 오차를 확인해야 한다.
- Local 비교는 key subsampling만의 효과를 분리한다. Full 비교의 깊은 layer에는 이전 layer부터 누적된 feature 변화도 포함된다.
- AV MSE는 mask별 출력 오차이며, 반복 mask의 기대 출력을 이용한 bias²/variance 분해가 아니다.
- Dense 출력은 reference이지 ground truth가 아니다. Dense와 가까워지는 것만으로 FID 개선을 보장하지 않는다.
- FM target은 해당 clean/noise pair의 conditional target이다. FID 또는 실제 marginal velocity 오차가 아니다.
- 이미 학습된 모델에 inference 때만 보정을 적용했다. 보정을 넣어 재학습한 결과를 예측하거나 검증한 실험은 아니다.
- 생성 trajectory/FID 및 held-out validation은 실행하지 않았다. 학습 데이터 입력의 attention 진단이다.
- 95% CI는 2,000회 이미지 단위 paired bootstrap. 각 이미지의 mask/head 평균 후 계산하며, timestep/head를 독립 표본으로 세지 않는다. 다중비교 보정 없는 탐색적 CI.

## 검증

```json
{
  "dense_instrumentation_max_abs": 1.430511474609375e-06,
  "dense_instrumentation_rmse": 3.0507968062920554e-07,
  "first_layer_local_full_max_metric_difference": 0.0,
  "sparse_instrumentation_max_abs": 1.430511474609375e-06
}
```

## 파일

- [시각화 dashboard](dashboard.html)
- summary.csv / paired_contrasts.csv: timestep·layer별 평균과 paired 95% CI.
- head_summary.csv: 각 head별 평균.
- velocity_summary.csv / velocity_contrasts.csv: 최종 velocity/FM 결과.
- measurements.npz: 이미지·timestep·mask·layer·mode·head별 원자료.
- sample_manifest.csv / metadata.json: 입력 목록과 실행 설정.
