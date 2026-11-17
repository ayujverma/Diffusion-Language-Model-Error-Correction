# Methods (draft; started in milestone 3)

This draft describes only what is implemented and verified in the repository. It
contains no experimental results. Correction training, the draft bank and the
weighting comparison (milestones 4–8) are not described yet. Decision numbers (D-xxx)
refer to `docs/decisions.md`.

## Backbone

We fine-tune the published Open-dCoder-0.5B checkpoint (Hugging Face
`fredzzp/open-dcoder-0.5B`, revision `d0d86d5b`; weights sha256 `59c1a005…7b6e`).
It is a 24-layer Qwen2-architecture masked diffusion language model (494,032,768
parameters, tied input/output embeddings) with bidirectional attention.

We re-implemented its forward pass in plain PyTorch. Attention is bidirectional
scaled-dot-product attention with a length-derived key-padding mask, and there is no
KV cache. On reference inputs its logits equal those of the upstream Open-dLLM
implementation exactly in float32. Under one external loss, parameter gradients agree
to a relative error of at most 3.1×10⁻⁵ in float32 (D-002, D-016).

As in upstream training and generation, the output at position i−1 is the
distribution for the token at position i (D-005). Every loss and the sampler use one
shared alignment function. The mask token is `<M>` (ID 151665). The ID in the
published generation config (151643) is end-of-text and is not used (D-003).

## Tasks

Two programmatically generated task families have exact intermediate targets (D-019,
D-021):
1. **Arithmetic chains** of n operations. The first value combines two d-digit
   constants. Each later value applies +c or −c (c with exactly d digits) or ×k
   (k ∈ [2, 9]) to the previous value.
2. **Short programs** in a restricted assignment language with two read-only inputs.
   Each statement reads its predecessor and applies a constant operation or adds or
   subtracts another defined variable.

Every intermediate value satisfies |v| ≤ 9999. Labels are computed by a hand-written
interpreter that never executes model output, and they are re-derived independently
during dataset verification. The pilot grid uses n ∈ {2, 4, 6} and d ∈ {1, 2, 3}.
This grid is provisional pending validation calibration.

Five splits (train, validation, in-distribution test, longer-chain test
n ∈ {8, 10, 12}, and unseen-composition test) are disjoint by a normalized program
key. One adjacent composition per family is held out for the unseen-composition test
(D-019, D-024).

## Response format and checker

The prompt lists the statements, the inputs, the ordered value names and the output
format. It contains no computed values. The response is one fixed-width slot per
value: a sign token (" +" or " −") followed by four single-digit tokens, with no
separators and no end-of-sequence token. Its length (5 tokens per value) is therefore
public (D-018).

The checker scores predictions by token ID at fixed positions. It reports token,
value, final-answer and whole-trace accuracy separately. Malformed slots count as
incorrect, and no prediction is trimmed or dropped (D-020).

## Masked fine-tuning objective (common warmup)

For each example b in a physical minibatch we draw u_b ~ U(0,1) and set
t_b = clamp(u_b, 1/500, 1 − 1/500). Each response token is independently replaced by
`<M>` with probability t_b. Prompt tokens are never masked (D-026, D-028). With M the
set of masked response positions in the whole minibatch,

  L_mask = (1/|M|) Σ_{i∈M} −log p_θ(x_i | x̃),

which is a pooled token mean with no 1/t weighting and no auxiliary path term (D-017).
Examples with no masked token are kept. A minibatch with |M| = 0 consumes data but
performs no optimizer or scheduler update. Data order, masks and evaluation masks come
from separate seeded random streams.

Cross-entropy uses plain PyTorch `log_softmax` in float32. The fused kernel used by
some upstream trainers is not used, because it is documented to ignore per-token
weights in its backward pass (D-007).

## Training procedure

All parameters are fine-tuned in float32 (D-030). The optimizer is AdamW with
learning rate 3×10⁻⁵, β = (0.9, 0.999), ε = 10⁻⁸ and weight decay 0.01. The learning
rate rises linearly over 20 updates and then stays constant. Gradient norms are
clipped at 1.0 and there is no gradient accumulation. The common warmup uses a
physical batch of 32 examples.

These are engineering defaults chosen from a short memory/runtime pilot, not tuned
values. bf16 autocast was supported on the development device (Apple M2 Max, MPS) and
numerically close to float32, but it was slower, so it is not used.

Checkpoints store the weights, the optimizer and scheduler state, every random-stream
state and the data-sampler position. Resuming is checked against an uninterrupted run
(`docs/verification.md`).

## Decoding for calibration

Generation starts from the prompt and a fully masked response of the public length.
With T model calls for L response tokens (1 ≤ T ≤ L), call s reveals
⌈sL/T⌉ − ⌈(s−1)L/T⌉ positions and never revises a revealed token (D-029).

At each masked position the candidate is the greedy argmax over ordinary tokenizer
tokens, excluding `<M>`, special tokens and unused output rows. Positions are revealed
in decreasing order of the candidate's unrestricted temperature-1 probability. Ties go
to the smaller token ID, then the leftmost position. The decoder never sees targets.
Model calls are counted per example.
