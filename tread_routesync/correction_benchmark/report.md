# Attention correction: implementation and benchmark

## 사용 방법

`tread_token_routing`, `tread_routesync`, `tread_routesync2`의 기존 학습 명령에 추가:

```bash
--tread-attn-correction \
  --tread-attn-correction-strength 1.0 \
  --tread-attn-correction-backend flex
```

- 기본값: 보정 OFF. `--no-tread-attn-correction` 또는 strength 0은 원래 attention 경로.
- `flex`: PyTorch FlexAttention + torch.compile(fullgraph=True, dynamic=False). 대각선 보정을 fused score_mod에서 계산.
- `sdpa`: 기존 scaled_dot_product_attention에 additive K×K bias 전달. 컴파일 불필요.
- Sparse routed block에서만 strength × log((K−1)/(N−1)) 적용. K는 실제 active count. K=1은 softmax가 항상 1이므로 보정 생략.
- Dense 실행, prefix, suffix는 기존 attention 경로. Sparse eval에는 학습과 같은 보정 적용. 최종 dense 평가에는 `--tread-eval-mode dense` 사용.
- 새로운 parameter/buffer/state_dict key 없음. 기존 checkpoint strict load, EMA deepcopy 호환.
- Resume 시 correction 설정 변경을 허용하고 warning으로 이전/새 값을 기록한다. 이는 기존 checkpoint에서 보정 continuation 실험을 위한 동작이다. 나머지 resume 검사는 유지.
- RouteSync 유무 및 recursive 공유 block 모두 지원. standalone tread_routesync2 내부에도 독립 구현을 포함.
- Flex는 CUDA 필요. PyTorch 2.5.1 동적 shape lowering 제약 때문에 정적 shape 컴파일 사용. dtype/shape가 바뀌면 재컴파일될 수 있으며 여러 active ratio/가변 batch를 자주 사용한다면 sdpa 경로가 단순하다.
- `--fused-attn`은 원래 attention 경로의 설정이다. 보정을 켠 routed block은 위 backend 선택을 사용한다.

## 측정 조건

- GPU: NVIDIA RTX 6000 Ada 48GB, PyTorch 2.5.1+cu124.
- SiT-B/2, latent 32×32, 원래 256 tokens, routed [3,9), active 128, local batch 32.
- BF16 autocast, 마지막 routed block FP32 유지, TF32 ON(정확도 검증에서는 OFF).
- 400K TREAD model 가중치에서 시작. GPU-resident synthetic 입력으로 FM forward + backward + gradient clipping + AdamW step.
- RouteSync 비교는 weight 0.1, sample ratio 1.0. 원본 checkpoint를 덮어쓰지 않으며 업데이트된 임시 가중치는 저장하지 않는다.
- 각 방식 warm-up 5 step, 측정 15 step × 3 rounds; 각 round에서 순서를 바꿈. CUDA synchronize 전후 wall time 측정.
- **공유 GPU에 다른 학습 작업들이 진행 중**이었다. 1% 미만의 차이는 속도 우위의 증거로 해석하지 않는다.
- Dataloader, DDP/all-reduce, EMA update, checkpoint 저장, sampling은 제외. 전체 실제 분산 학습 throughput을 보장하는 결과가 아니다.

## 전체 학습 연산 step: round 평균들의 중앙값

| 설정 | 기존 | SDPA 보정 | Flex 보정 | SDPA 변화 | Flex 변화 |
|---|---:|---:|---:|---:|---:|
| TREAD | 131.731 ms | 132.178 ms | 131.425 ms | +0.339% | -0.232% |
| TREAD + RouteSync | 153.018 ms | 152.192 ms | 153.127 ms | -0.540% | +0.071% |

**관찰:** 이 설정의 전체 step에서는 유의미한 감속이 관찰되지 않았다. attention 연산 자체의 비용이 0이라는 의미는 아니다.

## Attention Q/K/V microbenchmark

| dtype | 범위 | 기존 | SDPA 보정 | Flex 보정 |
|---|---|---:|---:|---:|
| torch.bfloat16 | forward | 0.0643 ms | 0.1243 ms | 0.1549 ms |
| torch.bfloat16 | forward + backward | 0.6602 ms | 0.6919 ms | 1.3049 ms |
| torch.float32 | forward | 0.2499 ms | 0.2303 ms | 0.1414 ms |
| torch.float32 | forward + backward | 0.9586 ms | 1.0876 ms | 1.3246 ms |

- 128-token attention 단독 측정에서는 Flex wrapper/dispatch 및 kernel 비용이 보인다. 전체 step은 projection, MLP, optimizer 등도 포함하므로 비율이 다르다.
- 짧은 microbenchmark는 공유 GPU scheduling과 CPU dispatch 변동의 영향을 특히 크게 받는다. 원자료의 각 round 값을 함께 확인할 것.

## 실제 SDPA backend 확인

- BF16 기존: `aten::_scaled_dot_product_flash_attention`.
- BF16 additive bias: `aten::_scaled_dot_product_efficient_attention`.
- FP32 endpoint: 기존/보정 모두 efficient attention.
- 따라서 additive bias가 FlashAttention 경로를 바꾸는 것은 실제 확인됐다. 하지만 이 장치/shape에서는 math fallback이 아니라 memory-efficient 경로였고, 전체 step 감속은 작았다.

## 정확도 및 회귀 검증

- float32 explicit softmax 수식 대비 CUDA forward 및 Q/K/V gradient 검증 통과. BF16은 dtype 반올림에 따른 약 0.2–0.35% 상대 L2 오차, FP32는 약 1e-6 이하.
- 보정 strength 0 및 dense 출력은 원래 모델과 bitwise 일치(CPU model integration).
- 기존 checkpoint strict load, sparse 출력 변화, recursive/nonrecursive, EMA deepcopy, 세 패키지의 backward 검증.
- tread_routesync 25 tests, tread_routesync2 16 tests: CUDA와 2-process DDP 포함 모두 통과.
- Flex 최초 컴파일/검증 및 full warm-up 시간은 각 results.json에 별도 기록. Triton disk cache 상태에 따라 달라지며 steady-state 시간에서 제외.

## 재현

```bash
python -m tread_routesync.benchmark_correction --device cuda:0 \
  --output tread_routesync/correction_benchmark/tread
python -m tread_routesync.benchmark_correction --device cuda:0 --routesync --skip-micro \
  --output tread_routesync/correction_benchmark/routesync
```

- [TREAD 원자료](tread/results.json)
- [RouteSync 원자료](routesync/results.json)
- 같은 shape의 컴파일은 최초 호출 시 발생한다. 다른 shape/precision에는 추가 컴파일이 필요할 수 있다.
