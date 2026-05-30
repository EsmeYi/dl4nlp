# Assignment 1 — Oral Exam Preparation

## Task 1.2: Building the Vocabulary

**Q: Why do we need a vocabulary? Why not feed raw text to the model?**

> Neural networks operate on numbers, not strings. A vocabulary is a mapping from token strings to integers, so that we can represent text as tensors. The embedding layer then converts each integer into a learnable vector.

**Q: What are the 4 special tokens and why do we need each one?**

> - **BOS** (beginning of sequence): marks the start of each paragraph, giving the model a clean initial state.
> - **EOS** (end of sequence): marks the end, so the model can learn where a text ends.
> - **UNK** (unknown): replaces any word not in the vocabulary (out-of-vocabulary tokens), so the model can still process unseen words without crashing.
> - **PAD** (padding): pads shorter sequences to the same length so we can stack them into rectangular tensors for batched processing.

**Q: Why do we limit the vocabulary size (max_voc_size)?**

> A larger vocabulary means a larger embedding matrix and a larger output layer, which increases memory usage and computation. Rare words contribute little to learning and are often replaced by UNK anyway, so there is little cost to dropping them.

**Q: How do we handle words that appear in the test set but not in the vocabulary?**

> They are mapped to the UNK token ID. The model has seen UNK during training (as a replacement for rare words), so it can still produce a prediction — it just cannot distinguish between different unknown words.

---

## Task 3.1: Setting Up the RNN Network

**Q: What are the three layers of the RNN language model?**

> 1. **Embedding layer** (`nn.Embedding`): maps each token ID to a dense vector of dimension E.
> 2. **Recurrent layer** (`nn.LSTM` or `nn.GRU`): processes the sequence left-to-right, outputting a hidden state of dimension H at each position.
> 3. **Unembedding layer** (`nn.Linear`): projects each hidden state to a score over the full vocabulary (V logits).

**Q: What are the tensor shapes at each step?**

> - `input_ids`: **(B, N)**
> - After embedding: **(B, N, E)**
> - After LSTM: **(B, N, H)**
> - After unembedding: **(B, N, V)**
> 
> where B = batch size, N = sequence length, E = embedding dim, H = hidden dim, V = vocab size.

**Q: Why use LSTM or GRU instead of basic RNN?**

> Basic RNNs suffer from the vanishing gradient problem — gradients shrink exponentially as they flow back through many time steps, making it very hard to learn long-range dependencies. LSTM and GRU introduce gating mechanisms that allow gradients to flow more freely over long sequences.

**Q: Why do we need an embedding layer? Why not feed token IDs directly into the LSTM?**

> Token IDs are arbitrary integers with no semantic meaning — the model would incorrectly treat them as ordered quantities. An embedding layer replaces each ID with a learnable dense vector, allowing the model to learn that similar words have similar representations.

---

## Task 3.2: Computing the Loss

**Q: What loss function do we use for language modeling, and why?**

> Cross-entropy loss. At each position, the model outputs a probability distribution over the vocabulary, and we penalize how much probability mass was assigned to the wrong token. Cross-entropy is the standard loss for multi-class classification.

**Q: Why do we shift the logits and labels before computing the loss?**

> Language modeling is an autoregressive task: at position `i`, the model should predict token `i+1`. So we compare:
> - `logits[:, :-1, :]` — predictions at all positions except the last (we don't observe what comes after EOS)
> - `labels[:, 1:]` — actual tokens at all positions except the first (there is nothing before BOS to predict)

**Q: Why do we set padding positions to -100 in the labels?**

> CrossEntropyLoss has an `ignore_index` parameter (set to -100 by HuggingFace convention). Positions with label -100 are excluded from the loss computation, so the model is not penalized for its predictions at padding positions.

---

## Task 4.1: Implementing the Trainer

**Q: What are the three steps of the backward pass, and why must they be in this order?**

> 1. `optimizer.zero_grad()` — clear gradients from the previous batch
> 2. `loss.backward()` — compute new gradients via backpropagation
> 3. `optimizer.step()` — update model parameters using the gradients
>
> `zero_grad()` must come first because PyTorch **accumulates** gradients by default. Without clearing them, gradients from previous batches would contaminate the update.

**Q: Why do we call `model.train()` and `model.eval()`?**

> Some layers (e.g. dropout, batch normalization) behave differently during training vs. evaluation. `model.train()` enables stochastic behavior for regularization; `model.eval()` disables it so validation results are deterministic.

**Q: Why do we use `torch.no_grad()` during validation?**

> During validation we are not updating parameters, so we don't need to compute gradients. `no_grad()` tells PyTorch not to build the computation graph, which saves memory and speeds up inference.

**Q: Why use AdamW instead of plain SGD?**

> AdamW adapts the learning rate for each parameter individually based on the history of gradients (first and second moments). This makes training faster and more stable than plain SGD, especially for deep networks. The "W" in AdamW refers to decoupled weight decay, which is a better regularization approach than L2 regularization in Adam.

---

## Task 5.2: Perplexity

**Q: What is perplexity and how is it computed?**

> Perplexity measures how well a language model predicts a text. It is the exponential of the average cross-entropy loss:
>
> **PPL = exp( (1/m) Σ -log P(wᵢ | context) ) = exp(cross_entropy_loss)**
>
> A lower perplexity means the model assigns higher probability to the actual words — it is less "surprised" by the text.

**Q: What does a perplexity of 200 mean intuitively?**

> The model behaves as if it is uniformly choosing among 200 words at every position. A perplexity equal to the vocabulary size means the model is essentially guessing randomly; a perplexity of 1 would mean perfect prediction.

**Q: What perplexity range do we expect from a well-trained model on this assignment?**

> A well-implemented model trained on the full dataset should achieve perplexity in the range of 200–300. Values above 700 suggest something is wrong.

---

## Task 5.3: Word Embeddings

**Q: What do word embeddings represent?**

> Each word is represented as a point in a high-dimensional vector space. The model learns these vectors during training, and words that appear in similar contexts end up with similar vectors. Intuitively, the embedding vector encodes a coarse representation of the word's meaning.

**Q: How do we find the nearest neighbors of a word in embedding space?**

> We compute the cosine similarity between the target word's embedding vector and all other vectors in the embedding matrix, then take the top-k highest values. Cosine similarity is preferred over Euclidean distance because it measures the angle between vectors, which is more robust to differences in vector magnitude.

**Q: What do we expect to see in the nearest neighbors, and what does it tell us?**

> We expect semantically similar words to be nearby. For example, "sweden" should be close to "denmark", "norway", "finland". If the model is well-trained, the geometry of the embedding space should reflect real-world relationships between words.

---

# Presentation Speeches (5 minutes each)

---

## Speech: Task 1.2 — Tokenizer

**Context and background:**
In Assignment 1, we're building a language model from scratch. Before the model can process text, we need to convert words into numbers — this is what the tokenizer does. The key design choice is building a word-level vocabulary from the training data.

**My solution:**
I implemented `build_tokenizer` and `A1Tokenizer`. The vocabulary is built by counting word frequencies in the training data using a `Counter`, then keeping the most frequent `max_vocab_size - 4` words. The minus 4 is because we reserve 4 special tokens: PAD (0), BOS (1), EOS (2), and UNK (3). Any word not in the vocabulary maps to UNK.

The `A1Tokenizer.__call__` method takes a list of strings and returns a `BatchEncoding` — the HuggingFace standard format. For each sentence: tokenize by splitting on spaces, add BOS at the start and EOS at the end, truncate to max length, then pad to the same length on the right. The padding uses token ID 0 (PAD), and `attention_mask` is 1 for real tokens and 0 for padding.

**How I used AI tools:**
I used Claude to understand the design — specifically why we need special tokens and what each one does. Claude helped me understand that BOS tells the model "a sentence is starting" and EOS tells it "stop generating here". I implemented the Counter-based vocabulary myself after that explanation, and worked through the padding logic by reasoning about the tensor shapes.

---

## Speech: Task 3.1/3.2 — RNN Language Model

**Context and background:**
The goal is to build a language model that can predict the next word given a sequence of words. We use an LSTM (Long Short-Term Memory) network, which processes text sequentially — one token at a time — and maintains a hidden state that summarizes what it has seen so far.

**My solution:**
The model has three layers: an `Embedding` layer that converts token IDs into dense vectors, an `LSTM` that processes the sequence and outputs a hidden state at each position, and a `Linear` layer that projects the hidden state to vocabulary logits.

The key implementation detail is the **loss shift**: during training, at position i the model should predict token i+1. So I shift the logits left by one (take `logits[:, :-1, :]`) and the labels right by one (take `labels[:, 1:]`). I also mask padding tokens by setting their labels to -100, which PyTorch's CrossEntropyLoss ignores automatically.

Training used AdamW optimizer with teacher forcing — the model always sees the correct previous token during training, not its own predictions. After 3 epochs, validation perplexity reached 66.4.

**How I used AI tools:**
I used Claude to understand teacher forcing and the loss shift. The key insight I got from our discussion was that at position i, the logit predicts token i+1, so the targets need to be shifted. I then implemented the forward pass myself with that understanding. I also asked Claude to explain why padding should use -100 rather than 0 — the answer is that 0 is a valid token ID (PAD token), so using it as a label would confuse the model.

