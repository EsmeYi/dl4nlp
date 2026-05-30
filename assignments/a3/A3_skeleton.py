# Assignment 3: Fine-tuning language models
# Converted from WASP_NLP_A3_skeleton.ipynb

import json
import math
import time
import torch
import torch.nn as nn

SEED = 101
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_TRAIN_SAMPLES = 5000
MAX_TEST_SAMPLES = 400

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
model_name_or_path = MODEL_NAME

# ============================================================
# Part 1: Preprocessing
# ============================================================

# Task 1.1: Load dataset
from datasets import load_dataset, DatasetDict

smoltalk = load_dataset("HuggingFaceTB/smoltalk", 'all')

smoltalk_simplified = smoltalk.filter(
    lambda row: len(row['messages']) <= 3
    and all(len(m['content']) <= 256 for m in row['messages'])
)
smoltalk_simplified = DatasetDict({
    "train": smoltalk_simplified["train"].select(range(MAX_TRAIN_SAMPLES)),
    "test":  smoltalk_simplified["test"].select(range(MAX_TEST_SAMPLES)),
})

print(smoltalk_simplified)
print(smoltalk_simplified['train'][0])

# ----------------------------------------------------------------
# Task 1.2: Format data for instruction tuning
# ----------------------------------------------------------------
def format_input_output(example):
    """Convert a dataset example into a prompt/response pair.

    Use ChatML format:
        <|im_start|>system
        {content}<|im_end|>
        <|im_start|>user
        {content}<|im_end|>
        <|im_start|>assistant
        <- response starts here

    Returns: {"prompt": str, "response": str}
    """
    messages = example['messages']

    prompt_parts = []
    for msg in messages[:-1]:
        role = msg['role']
        content = msg['content']
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")
    prompt = "".join(prompt_parts)

    response = messages[-1]['content'] + "<|im_end|>"

    return {"prompt": prompt, "response": response}


ds_sft = smoltalk_simplified.map(format_input_output)
print(ds_sft['train'][0])

# ----------------------------------------------------------------
# Task 1.3: Tokenize the dataset
# ----------------------------------------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_helper(example):
    """Tokenize a prompt/response example.

    Returns:
        input_ids:      token ids of prompt + response (concatenated)
        attention_mask: all 1s, same length as input_ids
        labels:         same as input_ids but prompt tokens replaced with -100
                        (so loss is only computed on the response)
    """
    prompt   = example['prompt']
    response = example['response']

    prompt_ids   = tokenizer(prompt,   add_special_tokens=False)['input_ids']
    response_ids = tokenizer(response, add_special_tokens=False)['input_ids']

    input_ids      = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels         = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


tokenized_ds_sft = ds_sft.map(tokenize_helper)
print(tokenized_ds_sft)

# ============================================================
# Part 2: Evaluation utilities (given — no TODO)
# ============================================================

def data_collator(batch):
    input_ids_list      = [torch.tensor(e["input_ids"],      dtype=torch.long) for e in batch]
    attention_masks_list = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in batch]
    labels_list          = [torch.tensor(e["labels"],         dtype=torch.long) for e in batch]

    max_len = max(x.size(0) for x in input_ids_list)

    def pad_to_max(x_list, pad_value):
        padded = []
        for x in x_list:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), pad_value, dtype=x.dtype)], dim=0)
            padded.append(x)
        return torch.stack(padded, dim=0)

    return {
        "input_ids":      pad_to_max(input_ids_list,       pad_value=tokenizer.pad_token_id),
        "attention_mask": pad_to_max(attention_masks_list, pad_value=0),
        "labels":         pad_to_max(labels_list,          pad_value=-100),
    }


import evaluate as hf_evaluate

class RougeMetricComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = hf_evaluate.load("rouge")
        self.all_predictions = []
        self.all_references  = []

    def __call__(self, eval_pred, compute_result=False):
        logits, labels = eval_pred
        pred_ids = logits.argmax(axis=-1)
        for p, lbl in zip(pred_ids, labels):
            mask = lbl != -100
            if mask.sum() == 0:
                continue
            ref_ids        = lbl[mask]
            pred_ids_filt  = p[mask]
            ref_text  = self.tokenizer.decode(ref_ids,       skip_special_tokens=True)
            pred_text = self.tokenizer.decode(pred_ids_filt, skip_special_tokens=True)
            self.all_references.append(ref_text.strip())
            self.all_predictions.append(pred_text.strip())
        if compute_result:
            if self.all_references:
                scores = self.rouge.compute(
                    predictions=self.all_predictions,
                    references=self.all_references,
                )
                self.all_predictions = []
                self.all_references  = []
                return {"rougeL": scores["rougeL"]}
            return {}
        return {}


compute_metrics = RougeMetricComputer(tokenizer)

from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import ProgressCallback

def make_trainer(model, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds_sft["train"],
        eval_dataset=tokenized_ds_sft["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if type(cb).__name__ != "NotebookProgressCallback"
    ]
    trainer.add_callback(ProgressCallback)
    return trainer

# ----------------------------------------------------------------
# Task 2.2: Evaluate pretrained baseline
# ----------------------------------------------------------------
from transformers import AutoModelForCausalLM

print("\n" + "=" * 80)
print("EVALUATING PRETRAINED MODEL")
print("=" * 80)

pretrained_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(DEVICE)

pretrained_eval_args = TrainingArguments(
    output_dir="/tmp/pretrained_eval",
    eval_strategy="no",
    per_device_eval_batch_size=1,
    bf16=False, fp16=True,
    report_to="none",
    batch_eval_metrics=True,
    eval_accumulation_steps=1,
)

pretrained_trainer  = make_trainer(pretrained_model, pretrained_eval_args)
pretrained_metrics  = pretrained_trainer.evaluate()
print(json.dumps(pretrained_metrics, indent=2))

# ============================================================
# Part 3: Supervised fine-tuning (full parameters)
# ============================================================

# ----------------------------------------------------------------
# Task 3.1: Train the full model
# ----------------------------------------------------------------
print("\n" + "=" * 80)
print("FULL SFT TRAINING")
print("=" * 80)

baseline_training_args = TrainingArguments(
    output_dir="/tmp/sft_baseline",
    eval_strategy="epoch",
    logging_steps=200,
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    bf16=False, fp16=True,
    report_to="none",
    batch_eval_metrics=True,
    eval_accumulation_steps=1,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(DEVICE)
baseline_trainer = make_trainer(base_model, baseline_training_args)

baseline_trainer.train()
baseline_metrics = baseline_trainer.evaluate()
print("\nFULL SFT EVAL METRICS:")
print(json.dumps(baseline_metrics, indent=2))

# ----------------------------------------------------------------
# Task 3.3: Count trainable parameters
# ----------------------------------------------------------------
def num_trainable_parameters(model):
    """Return the number of parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Full SFT trainable params: {num_trainable_parameters(base_model):,}")

# ============================================================
# Part 4: LoRA (parameter-efficient fine-tuning)
# ============================================================

# ----------------------------------------------------------------
# Task 4.1: Extract LoRA target layers
# ----------------------------------------------------------------
def extract_lora_targets(model):
    """Return a dict {layer_name: nn.Linear} for all q/k/v/o attention projections."""
    named_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(s) for s in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                named_layers[name] = module
    return named_layers


def replace_layers(model, named_layers):
    """Put modified layers back into the model by name (given)."""
    for name, layer in named_layers.items():
        components = name.split(".")
        submodule = model
        for comp in components[:-1]:
            submodule = getattr(submodule, comp)
        setattr(submodule, components[-1], layer)
    return model


# ----------------------------------------------------------------
# Task 4.2: Implement LoRA layer
# ----------------------------------------------------------------
class LoRALayer(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank update.

    Forward: W(x) + (alpha/r) * B(A(x))
      - W is frozen (requires_grad=False)
      - A: (in_features → r),  initialized with Kaiming uniform
      - B: (r → out_features), initialized to zeros
    """
    def __init__(self, W, r, alpha):
        super().__init__()
        self.W = W
        W.requires_grad_(False)          # freeze original weights

        in_features  = W.in_features
        out_features = W.out_features

        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)    # B=0 so LoRA output starts at zero

        self.scaling = alpha / r

    def forward(self, x):
        return self.W(x) + self.scaling * self.B(self.A(x))


# ----------------------------------------------------------------
# Task 4.3: Fine-tune with LoRA
# ----------------------------------------------------------------
lora_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(DEVICE)

# Freeze all parameters
for param in lora_model.parameters():
    param.requires_grad_(False)

# Replace q/k/v/o layers with LoRALayer
lora_targets = extract_lora_targets(lora_model)
lora_layers  = {name: LoRALayer(layer, r=8, alpha=16) for name, layer in lora_targets.items()}
lora_model   = replace_layers(lora_model, lora_layers)

print(f"LoRA trainable params:     {num_trainable_parameters(lora_model):,}")
print(f"Full SFT trainable params: {num_trainable_parameters(base_model):,}")

lora_training_args = TrainingArguments(
    output_dir="/tmp/sft_lora",
    eval_strategy="epoch",
    logging_steps=200,
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    bf16=False, fp16=True,
    report_to="none",
    batch_eval_metrics=True,
    eval_accumulation_steps=1,
)

lora_trainer = make_trainer(lora_model, lora_training_args)
lora_trainer.train()
lora_metrics = lora_trainer.evaluate()
print("\nLoRA EVAL METRICS:")
print(json.dumps(lora_metrics, indent=2))

# ----------------------------------------------------------------
# Task 4.4: Qualitative inspection
# ----------------------------------------------------------------
def generate_response(model, prompt, max_new_tokens=100):
    """Generate a response given a formatted prompt string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


test_examples = [
    ds_sft['test'][0],
    ds_sft['test'][1],
    ds_sft['test'][2],
]

for ex in test_examples:
    prompt = ex['prompt']
    reference = ex['response']
    print("=" * 70)
    print(f"PROMPT:\n{prompt}")
    print(f"REFERENCE:\n{reference}")
    print(f"PRETRAINED:\n{generate_response(pretrained_model, prompt)}")
    print(f"FULL SFT:\n{generate_response(base_model, prompt)}")
    print(f"LoRA:\n{generate_response(lora_model, prompt)}")
    print()
