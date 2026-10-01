# Mixture-of-Recursions SiT

`attention_alignment_hdiv`가 사용하는 **`models/sit_b1.py`** 기반의 독립 실험입니다.
기존 파일은 수정하지 않습니다. `models/sit.py`의 REPA projector 출력 모델과는 구분됩니다.
SiTBlock, timestep/class embedding, AdaLN, final layer, unpatchify는 원본을 재사용합니다.
학습은 기존 linear/cosine flow-matching objective를 사용합니다.
이 실험에 attention alignment/H-div auxiliary loss를 추가하지는 않습니다.

## 구조와 routing

SiT-B/2 기본 MoR 설정:

```text
3 unique prefix blocks                       c_base
2 shared recurrent blocks × 3 recursions      c_base
3 unique suffix blocks                       c_base
final layer                                  c_base
```

- 최대 effective depth: `3 + 2 * 3 + 3 = 12`. 실제 unique transformer blocks는 8개입니다.
- 원본 생성자에 unique depth만 전달해 초기화한 뒤 stage별 `ModuleList`로 등록합니다.
  `self.recurrent_blocks`의 동일한 두 object를 모든 recursion에서 호출합니다.
  recursion별 block 생성이나 복제는 없습니다. 학습의 EMA는 전체 모델을 별도로 복제합니다.
- `c_base = t_embed + y_embed`. 기본 설정은 recursion embedding 없이 모든 stage에서 `c_base`를 사용합니다.
  `--use-recursion-conditioning`을 켜면 recurrent stage에만 `[R,D]` embedding (`std=0.01`)을 더합니다.
- R1은 전체 token을 처리합니다. R1/R2 뒤의 독립 router는 `Linear(D, 1) -> sigmoid`입니다.
  LayerNorm/MLP 없이 hidden state에서 직접 score를 계산합니다.
  기본 hierarchical mode에서는 현재 active subset 안에서 sigmoid score로 `topk(dim=1)`을 수행합니다.
- capacity는 **원래 N 기준** `max(1, floor(N * ratio))`입니다.
  기본 `N=256`에서는 `[256, 128, 64]`, 모든 sample의 K가 같고 `S3 ⊆ S2 ⊆ S1`입니다.
  작은 N이나 동일 ratio에서는 subset이 같을 수 있습니다. ratio는 양수, 비증가, 첫 값 1이어야 합니다.
- 선택된 `[B,K,D]` token끼리만 원본 block의 self-attention을 실행합니다.
  batched `gather`와 **out-of-place `Tensor.scatter`**로 원래 위치를 갱신합니다.
  inactive token은 그대로 보존하며 suffix 입력은 항상 `[B,N,D]`입니다.
  hidden state에 in-place assignment, sample별 Python loop, `.item()`, `.clone()`을 사용하지 않습니다.

`--mor-global-topk`를 추가하면 각 router가 scatter로 복원된 전체 `[B,N,D]` sequence를 다시 평가합니다.
이 경우 R2는 전체 N에서 K2개, R3도 전체 N에서 K3개를 독립적으로 선택하므로 `S3 ⊆ S2` 제약이 없습니다.
R2에서 제외됐던 token도 R3에서 다시 선택될 수 있습니다. 선택된 token 수와 recurrent FLOPs는 동일하고,
두 번째 router의 scoring만 `[B,K2,D]` 대신 `[B,N,D]`에서 수행됩니다.

## Router gradient

[원 MoR 논문](https://papers.nips.cc/paper_files/paper/2025/file/8b08bbf8b420faa6eeb4020720582ec7-Paper-Conference.pdf)의
linear/sigmoid expert-choice 설정을 따라 각 router가 독립적으로 sigmoid score를 계산합니다.
R1은 gate 없이 dense하게 실행하며 router_0의 선택된 score가 R2를, router_1의 score가 R3를 gate합니다.
마지막 recursion 뒤에는 router가 없습니다. fixed K는 percentile selection의 고정 capacity 의미를 구현하며,
동점이 있어도 정확히 K개를 선택합니다.

```python
g = sigmoid(router_r(h_active))
selected_g, local_indices = g.topk(K_next, dim=1)
x_before = gather_tokens(h_active, local_indices)
x_after = recurrent_unit(x_before, c_next)
x_new = x_before + selected_g.unsqueeze(-1) * x_after
```

기존 SiTBlock 내부 residual을 그대로 유지하고, unit의 **전체 output**에 gate를 곱한 뒤
recursion-level outer residual을 더합니다. 두 수준의 residual이 의도적으로 함께 존재합니다.
예를 들어 unit이 identity라면 선택된 token은 `(1 + g) * x_before`가 됩니다.
실제 forward의 update 크기를 변경하며, 학습과 `no_grad` sampling 모두 동일하게 적용합니다.
선택된 sigmoid score를 통해 router에 실제 loss gradient가 전달됩니다. hard Top-K의 index 자체는
미분되지 않지만, straight-through estimator나 별도 auxiliary routing loss는 사용하지 않습니다.
`router_scores` 진단 값은 이제 logits가 아닌 sigmoid probability입니다.

Recursion embedding은 gating에 필수인 구성요소는 아닙니다. 다만 gate는 **업데이트 크기**, embedding은
**공유 unit의 recursion별 conditioning**을 담당하므로 서로 대체하지 않습니다. 기본값은
`use_recursion_conditioning=False`이며 embedding parameter 자체를 생성하지 않습니다.
이는 SiT에 추가한 독립적인 ablation 옵션이며 원 MoR router의 요구조건은 아닙니다.

원본 SiT의 AdaLN-Zero와 zero output initialization을 유지하므로 첫 backward에서 recurrent/embedding/router
gradient가 0인 것은 정상입니다. 테스트는 원래 초기화에서 optimizer 4회 업데이트 후 nonzero gradient를
확인하며, 별도로 학습 이후를 모사한 nonzero modulation/head에서도 gradient 경로를 검증합니다.

## 실행

저장소 root에서 실행합니다. 데이터 경로는 `CustomDataset`이 읽는 전처리된 ImageNet latent moments
경로로 바꾸세요. 아래 명령은 GPU 8개를 사용하는 단일 노드 실험입니다.

### 학습: JSON 없이 CLI로 실행

| 모델 | Prefix + recurrent × recursions + suffix | Unique blocks |
| --- | --- | --- |
| SiT-B/2 | `3 + 2 × 3 + 3 = 12` | 8 |
| SiT-L/2 | `6 + 4 × 3 + 6 = 24` | 16 |
| SiT-XL/2 | `8 + 4 × 3 + 8 = 28` | 20 |

표의 구성은 기준 모델과 effective depth를 맞춘 기본 예시입니다. `--model`은 hidden size, head 수와
patch size를 정하고, MoR stage 인자들은 실제 effective depth를 별도로 정합니다. 따라서 SiT-B/2에서도
`1 + 4 × 3 + 1 = 14`처럼 기준 depth 12와 다른 구성을 사용할 수 있습니다.

아래 명령은 JSON을 읽지 않습니다. 모델 구조, capacity, conditioning, batch, 학습 길이와 precision만
주요 실험 인자로 표시하고, learning rate 등은 코드 기본값을 사용합니다.
공통 기본값은 ImageNet 256×256, 1000 classes, linear path / velocity prediction / uniform sampling입니다.
Router는 `Linear → sigmoid`이며 선택된 다음 recursion은
`x_before + g * recurrent_unit(x_before, c_base)`로 업데이트합니다.

**SiT-B/2**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m mixture_of_recursions.train \
  --model SiT-B/2 --exp-name sit_b2_mor \
  --data-dir /path/to/imagenet_256 \
  --output-dir /path/to/experiments \
  --use-mor \
  --mor-prefix-blocks 3 --mor-recurrent-blocks 2 --mor-suffix-blocks 3 \
  --mor-max-recursions 3 --mor-capacity-ratios 1.0 0.5 0.25 \
  --no-use-recursion-conditioning \
  --batch-size 256 --max-train-steps 400000 \
  --mixed-precision bf16 --allow-tf32
```

**SiT-L/2**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m mixture_of_recursions.train \
  --model SiT-L/2 --exp-name sit_l2_mor \
  --data-dir /path/to/imagenet_256 \
  --output-dir /path/to/experiments \
  --use-mor \
  --mor-prefix-blocks 6 --mor-recurrent-blocks 4 --mor-suffix-blocks 6 \
  --mor-max-recursions 3 --mor-capacity-ratios 1.0 0.5 0.25 \
  --no-use-recursion-conditioning \
  --batch-size 256 --max-train-steps 400000 \
  --mixed-precision bf16 --allow-tf32
```

**SiT-XL/2**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m mixture_of_recursions.train \
  --model SiT-XL/2 --exp-name sit_xl2_mor \
  --data-dir /path/to/imagenet_256 \
  --output-dir /path/to/experiments \
  --use-mor \
  --mor-prefix-blocks 8 --mor-recurrent-blocks 4 --mor-suffix-blocks 8 \
  --mor-max-recursions 3 --mor-capacity-ratios 1.0 0.5 0.25 \
  --no-use-recursion-conditioning \
  --batch-size 256 --max-train-steps 400000 \
  --mixed-precision bf16 --allow-tf32
```

`--batch-size`는 전체 GPU 기준입니다. 위 설정은 GPU당 microbatch 32, effective batch 256입니다.
메모리가 부족하면 `--batch-size 128 --gradient-accumulation-steps 2`로 바꿔 effective batch 256을 유지할 수 있습니다.
기본 logging은 W&B이며, 사용하지 않으려면 `--report-to none`을 추가하세요.

Checkpoint는 기본 50,000 step마다 `<output-dir>/<exp-name>/checkpoints/0050000.pt`처럼 저장됩니다.
재개할 때는 같은 모델 설정, output directory, experiment name의 학습 명령에 `--resume-step 50000`을 추가합니다.
`--max-train-steps 400000`은 재개 이후 추가 횟수가 아니라 전체 목표 optimizer step입니다.

### 학습 중 W&B 이미지 기록

W&B 사용 시 **1 step과 이후 10,000 optimizer step마다 전역 64장**을 생성해
8×8 grid 하나로 묶은 뒤 `samples` 항목에 기록합니다. 별도의 PNG 파일이나 `samples/` 폴더는 만들지 않습니다.
`--report-to none` 또는 `tensorboard`이면 이미지 생성과 VAE 로딩을 건너뜁니다.
W&B 자체의 로컬 미디어 캐시는 W&B가 관리합니다.

- EMA 모델, 기존 Euler sampler 50 steps, `sd-vae-ft-mse` decoder 사용
- 각 rank가 `64 / world_size`장을 생성한 뒤 `accelerator.gather`로 모음
- rank별 고정 seed로 만든 서로 다른 noise/class를 반복 사용 (resume 후에도 동일)
- ImageNet CFG 학습은 CFG 4.0 사용; CFG를 끄거나 1000 classes가 아니면 1.0 사용
- 모든 rank의 EMA가 분담해 생성하고 main process만 grid를 W&B에 기록
- Training model의 train/eval 상태와 학습 RNG는 유지

간격 변경은 `--sampling-steps 5000`, 장수 변경은 `--sampling-batch-size 32`,
비활성화는 `--sampling-steps 0`을 추가하세요. 장수는 process 수로 나누어떨어져야 합니다.
기본 동작에는 추가 인자가 필요 없습니다. Decoder는 첫 샘플링 때 각 GPU에 한 번 로드하며,
로컬 캐시가 없다면 최초 실행 시 pretrained VAE를 다운로드합니다.

### 주요 비교 실험

아래 옵션은 위 학습 명령의 해당 flag를 **교체**하고, 결과가 섞이지 않도록 `--exp-name`도 변경하세요.

| 실험 | 변경 옵션 |
| --- | --- |
| Vanilla SiT baseline | `--use-mor` 대신 `--no-use-mor` |
| Recursion conditioning 추가 | `--no-use-recursion-conditioning` 대신 `--use-recursion-conditioning` |
| 모든 token을 반복 처리하는 shared MoR | `--mor-capacity-ratios 1.0 1.0 1.0` (gating과 outer residual은 유지) |
| 모든 token을 반복 처리하고 sigmoid gate 제거 | `--mor-capacity-ratios 1.0 1.0 1.0 --no-mor-gating` |
| 전체 sequence에서 recursion별 Top-K 재선택 | `--mor-global-topk` |

`--no-mor-gating`은 gate를 `g=1`로 고정한 것과 같아서 R2/R3를
`x_new = x_before + recurrent_unit(x_before, c_r)`로 업데이트합니다. SiTBlock 내부 residual과
recursion-level outer residual은 모두 유지됩니다. Gate 없이 sparse hard Top-K를 학습하면 router로
gradient가 전달되지 않으므로, 이 옵션은 capacity가 `1.0 1.0 1.0`일 때만 사용할 수 있습니다.
이 설정에서는 router를 실행하지 않으며 router parameter도 optimizer 학습 대상에서 제외됩니다.

예를 들어 SiT-B/2 width에서 effective depth 14, global Top-K와 recursion conditioning을 함께 쓰려면:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m mixture_of_recursions.train \
  --model SiT-B/2 --exp-name mor4 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-mor --mor-global-topk --use-recursion-conditioning \
  --mor-prefix-blocks 1 --mor-recurrent-blocks 4 --mor-suffix-blocks 1 \
  --mor-max-recursions 3 --mor-capacity-ratios 1.0 0.5 0.25 \
  --batch-size 256 --max-train-steps 400000 \
  --mixed-precision bf16 --allow-tf32
```

Baseline 비교에서는 모델 크기, seed, effective batch, learning rate, 학습 step과 생성 설정을 동일하게 유지하세요.
Vanilla는 원본 생성자/forward를 호출하며 원본 `blocks.*` checkpoint를 strict load할 수 있습니다.

MoR checkpoint와 vanilla checkpoint는 topology가 다릅니다. MoR를 켠 상태에서 vanilla checkpoint를
자동 변환하지 않습니다. 이전 LayerNorm/straight-through router 버전도 strict load 호환되지 않습니다.
직전 delta-gating 버전은 state-dict shape는 같지만 outer residual 변경으로 forward 결과가 달라집니다.

### 생성: EMA, SDE 250 steps, 50K images

학습한 모델의 checkpoint 경로를 지정합니다. 아래는 XL 예시입니다. 모델 크기와 MoR 설정은 checkpoint의
`args`에서 자동 복원하므로 생성 명령에 architecture flag를 반복하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  -m mixture_of_recursions.generate \
  --ckpt /path/to/experiments/sit_xl2_mor/checkpoints/0400000.pt \
  --sample-dir /path/to/samples \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
```

학습 때 저장된 capacity 대신 inference에서 다른 비율을 사용하려면 다음 옵션을 추가합니다.

```bash
--mor-capacity-ratios 1.0 1.0 1.0
```

예를 들어 `1.0 0.5 0.25`로 학습한 checkpoint의 모든 token을 세 recursion에서 활성화하려면:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  -m mixture_of_recursions.generate \
  --ckpt /v/mnt/GH/SiT/mor4/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0 \
  --mor-capacity-ratios 1.0 1.0 1.0
```

옵션을 생략하면 checkpoint의 학습 비율을 사용합니다. Override는 active token 수만 변경하며 학습된
router와 sigmoid gate는 그대로 사용합니다. 따라서 `1.0 1.0 1.0`에서는 모든 token이 recurrent unit을
통과하지만 각 update에는 여전히 token별 gate `g`가 곱해집니다. Override 비율은 출력 폴더명에도
`mor-cap-1-1-1` 형태로 포함되어 같은 checkpoint의 다른 inference 결과가 섞이지 않습니다.

EMA checkpoint, EMA VAE, seed 0은 기본값을 사용합니다.
`cfg-scale=1.0`은 guidance 없는 생성입니다. CFG 비교는 예를 들어 `--cfg-scale 1.5`로 변경합니다.
생성 sample count, sampler, step 수, VAE, CFG scale/window, seed는 모든 비교 실험에서 동일하게 사용하세요.
기존 sampler의 CFG null label이 1000으로 고정되어 있어 `cfg_scale > 1`은 1000 classes에서만 지원합니다.
생성기는 PNG와 FID 평가용 NPZ를 저장하며 FID 계산 자체는 수행하지 않습니다.

## 분석 API

```python
from mixture_of_recursions import SiT_models

model = SiT_models["SiT-B/2"](use_mor=True, qk_norm=False, fused_attn=True)
prediction, unused = model(x, t, y)  # 원본 계약: (prediction, None)
prediction, aux = model(x, t, y, return_aux=True)
```

`aux`는 gradient graph를 보관하지 않는 tensor를 반환합니다.

| Key | 기본 shapes / 의미 |
| --- | --- |
| `router_scores` | 길이 2: `[B,256]`, `[B,128]`, sigmoid routing scores [0,1] |
| `selected_indices` | 길이 3: `[B,256]`, `[B,128]`, `[B,64]`, 원래 sequence 기준 global indices |
| `active_token_counts` | `[256,128,64]` |

R1 indices는 전체 위치이며, `router_scores[r]`의 위치는 `selected_indices[r]`에 대응합니다.
`return_aux=False`에서는 진단 tensor/list를 보관하지 않습니다.

## 검증과 성능 한계

```bash
GLOO_SOCKET_IFNAME=lo python -m unittest mixture_of_recursions.tests.test_mor -v
```

Dummy model은 `B=2, N=256, D=32`, depth=12입니다. 테스트 항목:

- vanilla와 초기 state dict/RNG, strict checkpoint load, nonzero forward/backward 일치
- `[2,256,D] -> [2,128,D] -> [2,64,D]`, suffix `[2,256,D]`
- block object identity, nested Top-K, recursion condition과 ablation, inactive token 보존
- out-of-place scatter double-precision gradcheck 및 gated hard-routing reference 일치
- 고정 gate 0.25/0.75에서 내부/outer residual의 동시 적용과 train/no_grad 일치
- CPU 및 CUDA FP32/BF16, fused/unfused attention의 anomaly-detected backward
- CUDA에서 conditioning on/off의 gradient와 no_grad sampling 일치
- recurrent blocks, 각 router parameter, 각 recursion embedding row의 finite/nonzero gradient
- 원본 zero initialization의 실제 optimizer startup
- CPU Gloo DDP 반복 backward 및 2-rank BF16 gradient 동기화
- 실제 train loop의 dummy dataset 학습, checkpoint/EMA 저장 및 resume

GPU는 sandbox 밖에서 접근할 수 있습니다 (RTX 6000 Ada 8개 확인).
CUDA FP32/BF16 backward를 테스트하며, NCCL 및 ImageNet 장기 학습 성능은 별도 검증이 필요합니다.
설치된 timm 1.0.25에서는 원본 `models/sit_b1.py`의 `qk_norm=True` 생성자가 `norm_layer` 누락으로 실패합니다.
원본 block을 변경하지 않았으며 현재 테스트와 예제는 기본 `qk_norm=False`를 사용합니다.

주요 병목 후보는 Top-K/gather/scatter의 메모리 이동과 kernel launch, full state의 out-of-place scatter,
작은 active attention의 GPU 활용률 저하입니다. parameter sharing은 parameter/optimizer memory를 줄이지만
각 recursion의 backward activation은 필요합니다. prefix/suffix는 여전히 dense이며 실제 처리량 이득은
GPU benchmark가 필요합니다. Gated router의 학습 품질 역시 장기 실험으로 평가해야 합니다.
