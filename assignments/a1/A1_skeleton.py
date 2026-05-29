
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """
    counter = Counter()
    with open(train_file) as f:
        for line in f:
            if line.strip():
                counter.update(tokenize_fun(line.strip()))

    special_tokens = [pad_token, bos_token, eos_token, unk_token]

    n_real = (max_voc_size - len(special_tokens)) if max_voc_size else None
    real_words = [word for word, _ in counter.most_common(n_real)]

    all_tokens = special_tokens + real_words
    str_to_int = {tok: i for i, tok in enumerate(all_tokens)}
    int_to_str = {i: tok for tok, i in str_to_int.items()}

    return A1Tokenizer(str_to_int, int_to_str,
                       pad_token=pad_token, unk_token=unk_token,
                       bos_token=bos_token, eos_token=eos_token,
                       tokenize_fun=tokenize_fun,
                       model_max_length=model_max_length)

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str,
                 pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>',
                 tokenize_fun=lowercase_tokenizer, model_max_length=None):
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.tokenize_fun = tokenize_fun
        self.model_max_length = model_max_length

        self.pad_token_id = str_to_int[pad_token]
        self.unk_token_id = str_to_int[unk_token]
        self.bos_token_id = str_to_int[bos_token]
        self.eos_token_id = str_to_int[eos_token]

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.

           Args:
             texts:           The texts to tokenize. A list of strings, or list of list of strings.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')

        # Flatten one level of nesting if input is list of lists
        if texts and isinstance(texts[0], list):
            texts = [t for sublist in texts for t in sublist]

        encoded = []
        for text in texts:
            tokens = self.tokenize_fun(text)
            # Wrap with BOS and EOS, then map to integers (unknown words → unk_token_id)
            ids = [self.bos_token_id]
            ids += [self.str_to_int.get(tok, self.unk_token_id) for tok in tokens]
            ids += [self.eos_token_id]

            if truncation and self.model_max_length:
                ids = ids[:self.model_max_length]

            encoded.append(ids)

        if padding:
            max_len = max(len(ids) for ids in encoded)
            attention_masks = []
            padded = []
            for ids in encoded:
                n_pad = max_len - len(ids)
                attention_masks.append([1] * len(ids) + [0] * n_pad)
                padded.append(ids + [self.pad_token_id] * n_pad)
            encoded = padded
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]

        if return_tensors == 'pt':
            return BatchEncoding({
                'input_ids': torch.tensor(encoded),
                'attention_mask': torch.tensor(attention_masks),
            })
        return BatchEncoding({'input_ids': encoded, 'attention_mask': attention_masks})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=0, embedding_size=0, hidden_size=0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.

           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.
        """
        embedded = self.embedding(input_ids)          # (B, N) -> (B, N, E)
        rnn_out, _ = self.rnn(embedded)               # (B, N, E) -> (B, N, H)
        logits = self.unembedding(rnn_out)            # (B, N, H) -> (B, N, V)

        loss = None
        if labels is not None:
            # Shift: predict token i+1 from position i
            shift_logits = logits[:, :-1, :].contiguous()   # (B, N-1, V)
            shift_labels = labels[:, 1:].contiguous()        # (B, N-1)
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)),  # (B*(N-1), V)
                shift_labels.view(-1)                          # (B*(N-1),)
            )

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size)

        for epoch in range(int(args.num_train_epochs)):
            # Training
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids = self.tokenizer(batch['text'], return_tensors='pt',
                                           padding=True, truncation=True)['input_ids']
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                input_ids = input_ids.to(device)
                labels = labels.to(device)

                out = self.model(input_ids, labels=labels)
                loss = out.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = self.tokenizer(batch['text'], return_tensors='pt',
                                               padding=True, truncation=True)['input_ids']
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    out = self.model(input_ids, labels=labels)
                    total_val_loss += out.loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            print(f'Epoch {epoch+1}: train_loss={avg_train_loss:.3f}  val_loss={avg_val_loss:.3f}  val_ppl={val_perplexity:.1f}')

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)


###
### Part 5. Evaluation and analysis.
###

def predict_next_words(model, tokenizer, text, topk=5, device='cpu'):
    """Task 5.1: Given a text, print the top-k most likely next words."""
    model.eval()
    input_ids = tokenizer([text], return_tensors='pt', padding=False, truncation=True)['input_ids']
    input_ids = input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids)

    # Take logits at the second-to-last position (before EOS)
    last_logits = out.logits[0, -2, :]  # shape: (V,)
    top = last_logits.topk(topk)

    print(f'Input: "{text}"')
    print('Top next words:')
    for score, idx in zip(top.values, top.indices):
        word = tokenizer.int_to_str[idx.item()]
        print(f'  {word:20s} {score.item():.3f}')


def nearest_neighbors(model, tokenizer, word, n_neighbors=10):
    """Task 5.3: Find the nearest neighbors of a word in embedding space."""
    emb = model.embedding
    voc = tokenizer.str_to_int
    inv_voc = tokenizer.int_to_str

    if word not in voc:
        print(f'"{word}" not in vocabulary')
        return []

    test_emb = emb.weight[voc[word]]
    sim_func = nn.CosineSimilarity(dim=1)
    cosine_scores = sim_func(test_emb.unsqueeze(0), emb.weight)

    top = cosine_scores.topk(n_neighbors + 1)
    # Skip index 0: it's the query word itself
    results = [(inv_voc[idx.item()], cos.item())
               for idx, cos in zip(top.indices[1:], top.values[1:])]

    print(f'\nNearest neighbors of "{word}":')
    for w, score in results:
        print(f'  {w:20s} {score:.4f}')
    return results


###
### Part 2. Loading the text files and creating batches.
###

def load_datasets(train_file, val_file):
    from datasets import load_dataset
    dataset = load_dataset('text', data_files={'train': train_file, 'val': val_file})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')
    return dataset
