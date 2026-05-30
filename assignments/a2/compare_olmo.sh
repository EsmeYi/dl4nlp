#!/bin/bash
#SBATCH -A naiss2026-4-769
#SBATCH -J a2_compare
#SBATCH -o a2/logs/compare_%j.out
#SBATCH -e a2/logs/compare_%j.err
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 00:30:00

module load Python/3.11.3-GCCcore-12.3.0 PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_env/bin/activate
export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/hf_cache

# Pin to transformers 4.47.x: first version with OLMo-2, last before PyTorch 2.3+ deps
pip install "transformers==4.47.1" "tokenizers>=0.20,<0.22" -q --disable-pip-version-check 2>&1 | grep -v "^$"

# Re-patch import_utils.py after reinstall: bypass PyTorch 2.4 version check
IMPORT_UTILS=/cephyr/users/lirongy/Alvis/dl4nlp_env/lib/python3.11/site-packages/transformers/utils/import_utils.py
python - <<'PATCH'
import re, sys
path = "/cephyr/users/lirongy/Alvis/dl4nlp_env/lib/python3.11/site-packages/transformers/utils/import_utils.py"
src = open(path).read()
old = '''        if is_available and parsed_version < version.parse("2.4.0"):
            logger.warning_once(f"Disabling PyTorch because PyTorch >= 2.4 is required but found {torch_version}")
        return is_available and version.parse(torch_version) >= version.parse("2.4.0")'''
new = '        return is_available  # version check bypassed for PyTorch 2.1 on Alvis'
if old in src:
    open(path, 'w').write(src.replace(old, new))
    print("Patched import_utils.py")
elif 'version check bypassed' in src:
    print("Already patched")
else:
    print("WARNING: patch target not found, may need manual fix")
PATCH

cd /mimer/NOBACKUP/groups/naiss2026-4-769/lirongy/dl4nlp_assignments

python - <<'EOF'
import sys, torch, nltk
sys.path.insert(0, 'a1')
sys.path.insert(0, 'a2')
nltk.download('punkt_tab', quiet=True)

from A1_skeleton import A1Tokenizer
from A2_skeleton import A2Transformer, generate
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}\n')

# Load our trained Transformer
our_tokenizer = A1Tokenizer.from_file('a1/tokenizer.pkl')
our_model = A2Transformer.from_pretrained('a2/trainer_output').to(device)
our_model.eval()

# Load OLMo-2
olmo_name = 'allenai/OLMo-2-0425-1B'
print('Loading OLMo-2...')
olmo_tokenizer = AutoTokenizer.from_pretrained(olmo_name)
olmo_model = AutoModelForCausalLM.from_pretrained(olmo_name, torch_dtype=torch.float16).to(device)
olmo_model.eval()
print('OLMo-2 loaded.\n')

def olmo_generate(prompt, max_new_tokens=60):
    inputs = olmo_tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = olmo_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=True, temperature=0.7, top_k=50)
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return olmo_tokenizer.decode(new_tokens, skip_special_tokens=True)

prompts = [
    'In natural language processing, a Transformer',
    'Stockholm is the capital of Sweden . The country is known for',
    'The most important invention of the 20th century was',
]

for prompt in prompts:
    print('=' * 70)
    print(f'PROMPT: {prompt}')
    print()
    our_out = generate(our_model, our_tokenizer, prompt,
                       max_length=60, temperature=0.7, topk=50, device=device)
    print(f'Our Transformer:\n  {our_out}')
    print()
    olmo_out = olmo_generate(prompt)
    print(f'OLMo-2 (1B):\n  {olmo_out}')
    print()
EOF
