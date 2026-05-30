# Assignment 2 — Oral Exam Preparation

## Task 1.1: SwiGLU MLP

**Q: What is SwiGLU and how does it differ from a standard MLP?**

> A standard MLP applies a single linear projection followed by a non-linearity (e.g. ReLU), then another linear projection. SwiGLU adds a gating mechanism: it uses two parallel linear projections — one passed through SiLU to produce activations, and one acting as a gate. Their element-wise product allows the network to selectively suppress or amplify different dimensions depending on the input. This gives the model more expressivity than a hard activation like ReLU.
>
> Formula: `output = down_proj( gate_proj(x) * SiLU(up_proj(x)) )`

**Q: What are the three linear layers and their dimensions?**

> - `gate_proj`: H → I (hidden_size → intermediate_size)
> - `up_proj`:   H → I
> - `down_proj`: I → H
>
> Input and output both have shape (B, N, H) — the shape is preserved.

**Q: Why does OLMo 2 use bias=False in all linear layers?**

> Removing bias terms reduces the number of parameters slightly, and in practice modern large language models have found that biases are not necessary when layer normalization is present. It also makes weight tying between embedding and unembedding easier.

---

## Task 1.3: Multi-Head Attention

**Q: What is the role of Q, K, and V in attention?**

> Each token produces three vectors:
> - **Query (Q)**: what this token is looking for from other tokens.
> - **Key (K)**: what this token can offer to other tokens looking at it.
> - **Value (V)**: the actual information this token passes on if selected.
>
> Attention scores are computed as scaled dot products between Q and K, normalized with softmax, then used to take a weighted sum of V vectors. This allows each token to gather relevant information from all other tokens in the sequence.

**Q: Why do we scale by √d_h before softmax?**

> The dot product Q·Kᵀ grows in magnitude as the head dimension d_h increases (variance scales with d_h). Without scaling, large values would cause the softmax to saturate — producing near-one-hot distributions where gradients nearly vanish. Dividing by √d_h keeps the variance of the dot products roughly constant regardless of d_h.

**Q: What is multi-head attention and why use multiple heads?**

> Instead of running attention once, multi-head attention runs it `n_heads` times in parallel, each head working on a different d_h-dimensional subspace of the representation. Each head can learn to attend to different types of relationships — one head might capture syntactic dependencies, another might capture semantic similarity. The outputs of all heads are concatenated and projected back to hidden_size.

**Q: What is a causal mask and why do we need it for language modeling?**

> A causal mask ensures that position `i` can only attend to positions `0..i`, not future positions. This is essential for autoregressive language models: during training, the model must predict token `i+1` using only the tokens seen so far. Without the causal mask, the model could "cheat" by looking at the answer directly.

**Q: What is RoPE (Rotary Position Embedding)?**

> RoPE encodes the position of each token by rotating the query and key vectors by an angle that depends on their position. Unlike absolute positional embeddings added to the input, RoPE is applied directly to Q and K inside each attention layer. The key property is that the dot product Q·Kᵀ between two positions depends only on their relative distance, which helps the model generalize to longer sequences than it was trained on.

---

## Task 1.4: Transformer Decoder Layer

**Q: What are the components of a single Transformer decoder layer?**

> Each decoder layer contains two sub-layers, each followed by a residual connection:
> 1. **RMSNorm → Multi-Head Attention → + residual**
> 2. **RMSNorm → MLP → + residual**
>
> OLMo 2 uses Pre-Norm: the normalization is applied before each sub-layer (not after).

**Q: Why are residual connections important?**

> Residual connections allow gradients to flow directly back through the network without passing through the attention or MLP layers. This prevents the vanishing gradient problem in deep networks. They also mean each layer only needs to learn a correction on top of the identity mapping, which is much easier to optimize. Without residual connections, deep Transformers are very hard to train.

**Q: Why use Pre-Norm instead of Post-Norm?**

> Post-Norm (normalizing after the residual addition) was used in the original Transformer. However, it can cause unstable training in very deep models. Pre-Norm (normalizing before each sub-layer) keeps the magnitude of the residual stream stable throughout the network and has become the standard in modern LLMs.

**Q: Why does the Transformer need normalization at all?**

> Without normalization, activations can grow very large or shrink to near zero as they pass through many layers, making training unstable. RMSNorm rescales the hidden states to have unit root-mean-square, keeping activations in a reasonable range throughout training.

---

## Task 3.2: Text Generation

**Q: How does autoregressive text generation work?**

> The model generates one token at a time. At each step:
> 1. Feed the current sequence into the model.
> 2. Take the logits at the last position — this is the next-token distribution.
> 3. Sample a token from this distribution.
> 4. Append the sampled token to the sequence and repeat.
> Generation stops when an EOS token is produced or max_length is reached.

**Q: What does temperature do, and what happens at the extremes?**

> Temperature divides the logits before softmax: `logits / temperature`. This scales the sharpness of the distribution:
> - **Low temperature (e.g. 0.1)**: distribution becomes very peaked — the top token gets nearly all probability mass. Generation is near-deterministic and repetitive.
> - **High temperature (e.g. 2.0)**: distribution flattens — probability spreads more evenly. Generation is more random and diverse but may become incoherent.
> - **Temperature = 1.0**: the original distribution is unchanged.

**Q: What is top-k sampling and why is it useful?**

> Top-k sampling keeps only the k highest-probability tokens and sets the rest to zero probability before sampling. This prevents the model from accidentally sampling very low-probability tokens that would produce incoherent or nonsensical text, while still allowing diversity among the most plausible continuations.

**Q: What is the difference between greedy decoding and sampling?**

> Greedy decoding always picks the most probable next token (argmax). This produces deterministic, often repetitive text. Sampling draws from the probability distribution, introducing randomness that makes output more natural and varied — at the cost of occasionally producing less coherent text.

---

## Task 3.3: Pre-trained vs. From-Scratch Models

**Q: Why does a model trained from scratch on a small dataset produce worse text than OLMo-2?**

> OLMo-2 was trained on hundreds of billions of tokens, while our model was trained on ~150,000 Wikipedia paragraphs. More data means the model has seen far more diverse language patterns. OLMo-2 also has ~1 billion parameters compared to our ~6 million, giving it much more capacity to store and generalize knowledge.

**Q: OLMo-2 is a pure language model, not an instruction-tuned model. What does that mean?**

> A pure language model is trained only to predict the next token — it continues text. An instruction-tuned model has been further fine-tuned (using RLHF or supervised fine-tuning) to follow instructions and engage in dialogue. OLMo-2 will continue a prompt as if writing a document, rather than answering it like a chatbot.

**Q: Why does a language model sometimes produce confident-sounding but factually wrong text?**

> The model learns statistical patterns of which words appear together, not factual truth. It generates text that looks like plausible natural language, but has no mechanism to verify whether the facts it states are correct. This is called "hallucination".

---

### Actual comparison results (Slurm job 6703419, temperature=0.7, top-k=50)

**Prompt 1:** `In natural language processing, a Transformer`

- **Our model:** `in natural language processing , a <UNK> <EOS> ( <UNK> ) is a <UNK> to a <UNK> , a <UNK> . the <UNK> is a <UNK> <UNK> with a <UNK> .`
- **OLMo-2:** `is a type of neural network that enables a computer to perform various tasks. It is a machine learning model that enables it to learn from data and predict outcomes...`

**Prompt 2:** `Stockholm is the capital of Sweden . The country is known for`

- **Our model:** `stockholm is the capital of sweden . the country is known for <EOS> <UNK> , of which the capital of cologne .`
- **OLMo-2:** `its unique design and for a wealth of cultural heritage. Sweden is known as the land of the midnight sun...`

**Prompt 3:** `The most important invention of the 20th century was`

- **Our model:** `the most important invention of the 20th century was <EOS> <UNK> , which was built in <UNK> <UNK> and <UNK> in <UNK> , england . the <UNK> <UNK> was <UNK> of <UNK> ...`
- **OLMo-2:** `the telephone. Although it was invented in the 19th century, it wasn't as popular as it is today...`

**Q: What do the `<UNK>` tokens in our model's output tell us?**

> Our vocabulary is built from the training data using a fixed-size word list. Any word not seen frequently enough during training is mapped to `<UNK>`. This means domain-specific or rare words (like "Transformer", "telephone") are unknown to our model. OLMo-2 uses a subword tokenizer (BPE) that can represent any word by breaking it into known subword pieces, so it never produces `<UNK>`.

**Q: Why does our model sometimes generate `<EOS>` in the middle of a sentence?**

> Our model learned from Wikipedia text where sentences end with periods, and `<EOS>` marks the end of a training example. The model may have learned to associate certain contexts with termination. This is a sign that the model has not learned robust long-range generation — it falls back to ending the sequence when uncertain.

**Q: Summarize the key differences between our model and OLMo-2.**

> Three main dimensions:
> 1. **Scale**: OLMo-2 has ~1B parameters trained on hundreds of billions of tokens; ours has ~6M parameters trained on ~150k Wikipedia paragraphs.
> 2. **Tokenization**: OLMo-2 uses BPE subword tokenization (no `<UNK>`); ours uses a word-level vocabulary with an `<UNK>` fallback for rare words.
> 3. **Output quality**: OLMo-2 generates fluent, coherent, factually reasonable text; our model generates grammatically plausible structure but with most content words replaced by `<UNK>`, showing it has learned syntax but not sufficient vocabulary or world knowledge.

---

# Presentation Speeches (5 minutes each)

---

## Speech: Task 1.3 — Multi-Head Attention with RoPE

**Context and background:**
In Assignment 2, I reimplemented a Transformer language model following the OLMo-2 architecture. The core innovation of Transformers over RNNs is the attention mechanism: instead of passing information through a sequential hidden state, every token can directly attend to every other token. My task was to implement Multi-Head Attention with Rotary Position Embeddings.

**My solution:**
The attention module has four linear projections: W_q, W_k, W_v, W_o. The forward pass reshapes Q, K, V from shape (B, N, H) to (B, n_heads, N, d_head), applies RoPE to Q and K to encode position information, then calls `scaled_dot_product_attention` with `is_causal=True` to prevent each position from attending to future tokens. Finally, the heads are concatenated and projected back through W_o.

RoPE encodes position by rotating Q and K vectors by an angle proportional to their position. The key property is that the dot product Q·Kᵀ depends only on the relative distance between positions, not absolute positions. This makes the model generalize better to different sequence lengths.

I also added `q_norm` and `k_norm` (RMSNorm on queries and keys) which stabilizes training in deep models by preventing attention scores from growing too large.

**How I used AI tools:**
I used Claude to understand the conceptual difference between Q, K, and V — Q is "what I'm looking for", K is "what I can offer", V is "what I actually pass on". The scaling by √d_h was something I understood from the explanation that dot products grow in magnitude with dimension, so without scaling the softmax would saturate. I worked through the tensor reshape operations myself by tracking the shapes step by step.

---

## Speech: Task 3.3 — Comparison with OLMo-2

**Context and background:**
After training my Transformer from scratch, I compared its text generation quality to OLMo-2 1B — a model trained on hundreds of billions of tokens with 1 billion parameters, versus my ~6 million parameter model trained on ~150,000 Wikipedia paragraphs.

**My solution:**
I wrote a comparison script that runs both models on three prompts: one about NLP/Transformers, one about Stockholm/Sweden, and one about 20th century inventions. The script loads my model from the saved checkpoint and OLMo-2 from HuggingFace, then generates text with temperature=0.7 and top-k=50.

The results were striking: my model produced mostly `<UNK>` tokens because many content words — including "Transformer" itself — were not frequent enough to be in my word-level vocabulary. OLMo-2, using BPE subword tokenization, never produces `<UNK>` and generates fluent, coherent text. Despite this, my model's validation perplexity of 51.1 showed it had learned syntactic patterns.

**How I used AI tools:**
The main challenge was a series of dependency conflicts — the cluster's PyTorch 2.1 was incompatible with newer versions of transformers needed to load OLMo-2. I worked through these with Claude, ultimately patching `import_utils.py` to bypass the version check. Claude explained why the `<UNK>` outputs happen (word-level vocabulary limitation) and helped me articulate the three key differences between the models: scale, tokenization, and training data.

