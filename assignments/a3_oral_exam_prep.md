# Assignment 3 — Oral Exam Preparation

## Overview

**Q: What is the goal of Assignment 3?**

> We take a pretrained language model (SmolLM2-135M) that can only continue text, and fine-tune it to follow instructions — turning it into an assistant that responds to user queries. This is called Supervised Fine-Tuning (SFT).

---

## Task 1.2: Data Formatting

**Q: Why do we need to format the data before fine-tuning?**

> The pretrained model has no concept of "user" or "assistant" roles. We need a consistent format that tells the model which part is the input (prompt) and which part it should generate (response). Without formatting, the model doesn't know when to stop generating or what role it is playing.

**Q: What is ChatML format and why use it for SmolLM2?**

> ChatML wraps each message with role tags:
> ```
> <|im_start|>system
> You are a helpful assistant.<|im_end|>
> <|im_start|>user
> What is the capital of France?<|im_end|>
> <|im_start|>assistant
> ```
> SmolLM2 was pretrained with this format, so using it ensures the special tokens are already in its vocabulary and the model can recognize role boundaries.

**Q: What goes in the prompt vs. the response?**

> The prompt contains all messages except the last assistant turn (system + user messages), plus the opening `<|im_start|>assistant\n` tag. The response is the assistant's reply plus `<|im_end|>`. This tells the model exactly where to start and stop generating.

---

## Task 1.3: Tokenization

**Q: Why do we set the prompt tokens to -100 in `labels`?**

> During training, the model predicts the next token at every position. If we included the prompt tokens in the loss, we would be training the model to "generate" the user's question — but that is not what we want. We only want the model to learn to generate good responses. PyTorch's `CrossEntropyLoss` skips positions where the label is -100, so setting prompt tokens to -100 ensures the loss is only computed over the response tokens.

**Q: What is the difference between `input_ids` and `labels`?**

> `input_ids` is the full token sequence (prompt + response) fed into the model as input. `labels` has the same length but with prompt positions replaced by -100. The model sees the full context (including the prompt) to generate each token, but only gets penalized for the response tokens.

---

## Task 2.2: Baseline Evaluation

**Q: Why does the pretrained model have a non-zero ROUGE-L score even without instruction tuning?**

> The pretrained model has learned general language patterns from large amounts of text. When given a prompt, it generates plausible-sounding continuations that may overlap with the reference answer in common words and phrases — especially for short or formulaic responses. ROUGE-L measures the longest common subsequence, so even accidental word overlap contributes to the score.

**Q: What is ROUGE-L?**

> ROUGE-L measures the longest common subsequence (LCS) between the model's output and the reference answer, normalized by their lengths. It captures word overlap while being order-sensitive, making it a useful metric for evaluating generated text quality without requiring exact matches.

---

## Task 3.1: Full SFT

**Q: What does full SFT do differently from pretraining?**

> Pretraining trains the model to predict the next token on raw text (no specific format). Full SFT continues training on instruction-response pairs using the same next-token prediction objective, but only computes loss on the response tokens (prompt is masked). This teaches the model to respond to instructions rather than just continue any text.

**Q: How do you expect ROUGE-L to change after full SFT vs. the baseline?**

> ROUGE-L should increase after SFT, because the model learns to produce responses that match the expected format and content of instruction-following outputs. Before SFT, the model just continues the text without understanding it should generate a structured answer.

---

## Task 3.3: Counting Trainable Parameters

**Q: How do you count trainable parameters in PyTorch?**

> ```python
> sum(p.numel() for p in model.parameters() if p.requires_grad)
> ```
> `p.numel()` returns the number of elements in the tensor. We filter by `requires_grad=True` to count only parameters that will be updated during training.

**Q: How many trainable parameters does SmolLM2-135M have in full SFT?**

> All ~135 million parameters, since nothing is frozen.

---

## Task 4.1–4.3: LoRA

**Q: What problem does LoRA solve?**

> Full fine-tuning updates all parameters of the model, which is expensive in memory and compute for large models. LoRA (Low-Rank Adaptation) freezes the original weights and adds small trainable low-rank matrices, reducing the number of trainable parameters by 100x or more while achieving similar performance.

**Q: Explain the LoRA forward pass.**

> For a frozen weight matrix W, instead of updating W directly, LoRA adds a low-rank update:
> `output = W(x) + (α/r) · B(A(x))`
> where A has shape (in_features → r) and B has shape (r → out_features). Since r is small (e.g. 8), A and B together have far fewer parameters than W.

**Q: Why is B initialized to zero?**

> At the start of training, we want the LoRA correction `(α/r)·B(A(x))` to be zero, so the model behaves exactly like the original pretrained model. If B were randomly initialized, the model's outputs would be randomly perturbed at the start of training, making optimization unstable. A is initialized with Kaiming uniform so it is not zero — otherwise gradients through A would be zero and it could never update.

**Q: Why is A initialized with Kaiming uniform and not also zeros?**

> If both A and B were zero, the gradient of the loss with respect to A would be zero (since B·A = 0 means ∂L/∂A = 0). A would never update. Kaiming initialization gives A nonzero values so gradients can flow, while B=0 ensures the initial output is unchanged.

**Q: Which layers does LoRA target and why?**

> LoRA targets the query, key, value, and output projection matrices (q_proj, k_proj, v_proj, o_proj) in each attention block. These linear layers encode the model's knowledge about how tokens relate to each other. The LoRA paper found that adapting these attention matrices is sufficient for instruction following, while other layers (MLP, embedding, norm) can remain frozen.

**Q: What is the effect of the rank r?**

> Lower r means fewer trainable parameters (more parameter-efficient) but a rougher approximation of the full weight update. Higher r gives more expressive updates but uses more memory. r=8 is a common default that balances efficiency and performance.

**Q: Compare the number of trainable parameters between full SFT and LoRA.**

> Full SFT: ~135M parameters (all of SmolLM2-135M).
> LoRA (r=8): only the A and B matrices for q/k/v/o in each of the 30 Transformer layers. Each layer has 4 pairs of (A, B), each pair with roughly 2 × (hidden_size × r) parameters. This is typically ~1–3M parameters — about 100x fewer than full SFT.

---

## Task 4.4: Qualitative Inspection

**Q: What do you expect to see when comparing the three models qualitatively?**

> - **Pretrained model**: ignores the instruction format and just continues the text in an unrelated way.
> - **Full SFT model**: follows instructions and produces correct answers, though sometimes repeating itself.
> - **LoRA model**: produces the correct answer but may append garbled tokens afterwards, since the small number of parameters is not sufficient to perfectly learn when to stop generating.

---

### Actual results (Slurm job 6703495, SmolLM2-135M, 1 epoch, 5000 train samples)

| Model | eval_loss | ROUGE-L | Trainable params |
|-------|-----------|---------|-----------------|
| Pretrained | 2.619 | 0.575 | 0 |
| Full SFT | **1.157** | **0.674** | 134,515,008 (135M) |
| LoRA (r=8) | 1.580 | 0.629 | 921,600 (0.9M) |

**Q: What do these results tell us about LoRA?**

> LoRA uses only 0.7% of the parameters of full SFT (921K vs 135M), yet raises ROUGE-L from 0.575 to 0.629 — already most of the way to full SFT's 0.674. This confirms that low-rank adaptation is highly parameter-efficient: a small correction to the attention weights is sufficient to teach instruction-following behavior.

**Q: Why does the LoRA model sometimes generate garbled text after the correct answer?**

> With only 921K trainable parameters, LoRA has limited capacity to learn all aspects of the output format, including reliably generating the `<|im_end|>` stop token at the right place. Full SFT, with 135M trainable parameters, more reliably learns to stop generating. This is a known limitation of very low-rank LoRA on small models.

**Q: Why is the pretrained ROUGE-L already 0.575, not near zero?**

> ROUGE-L measures the longest common subsequence between the generated text and the reference. Even without instruction tuning, the pretrained model generates plausible English text that shares common words and phrases with the reference answers — especially short, formulaic responses. This gives a non-trivial ROUGE-L baseline even without any fine-tuning.

---

# Presentation Speeches (5 minutes each)

---

## Speech: Task 4.2 — LoRA Layer

**Context and background:**
In Assignment 3, I fine-tuned SmolLM2-135M to follow instructions. The challenge is that fine-tuning all 135 million parameters is expensive. LoRA (Low-Rank Adaptation) solves this by keeping the original weights frozen and adding small trainable low-rank matrices. My task was to implement the `LoRALayer` class.

**My solution:**
`LoRALayer` is a drop-in replacement for `nn.Linear`. In `__init__`, I freeze the original weight matrix W by calling `W.requires_grad_(False)`. Then I create two new linear layers: A with shape (in_features → r) and B with shape (r → out_features), both without bias. A is initialized with Kaiming uniform, and B is initialized to all zeros.

In `forward`, the output is `W(x) + (alpha/r) * B(A(x))`. At the start of training, B=0 means the LoRA term is zero, so the model behaves exactly like the original. Only A and B are trained.

The rank r=8 means instead of updating a matrix of size (in × out), we only train two matrices of size (in × 8) and (8 × out) — about 100x fewer parameters for typical hidden sizes.

**Results:** LoRA used only 921K parameters (0.7% of full SFT's 135M) and achieved ROUGE-L of 0.629, compared to full SFT's 0.674. A small gap in quality for a massive saving in computation.

**How I used AI tools:**
The key conceptual question I worked through with Claude was: why is B initialized to zero, not A? The answer is that if both were zero, the gradient through A would be zero and A could never update. B=0 ensures the initial output is unchanged, while A's non-zero initialization allows gradients to flow. I implemented the forward pass myself after understanding this.

---

## Speech: Task 1.2/1.3 — Data Formatting and Tokenization for SFT

**Context and background:**
Before fine-tuning, I needed to convert the SmolTalk instruction dataset into a format the model can learn from. This involves two steps: formatting messages into a structured prompt/response pair, and tokenizing them with the correct loss masking.

**My solution:**
For formatting (`format_input_output`), I used ChatML format — the same format SmolLM2 was pretrained with. Each message is wrapped as `<|im_start|>role\ncontent<|im_end|>\n`. The prompt includes all messages except the last assistant turn, ending with `<|im_start|>assistant\n`. The response is the assistant's reply plus `<|im_end|>`.

For tokenization (`tokenize_helper`), I tokenize prompt and response separately using `add_special_tokens=False`, then concatenate the token IDs. The `labels` tensor is identical to `input_ids` except the prompt portion is replaced with -100. This tells PyTorch's CrossEntropyLoss to skip those positions — we only compute loss on the response tokens, teaching the model to generate answers, not to repeat the questions.

**How I used AI tools:**
I understood the ChatML format with Claude's help and chose it because SmolLM2 was pretrained with it. For the -100 masking I first thought the reason was to prevent the model from "seeing the answer", but Claude helped me correct this: the model DOES see the prompt as input — the masking only affects the loss computation. We want the model to learn to generate the response, so we only penalize it for the response tokens.

