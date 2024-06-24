import pandas as pd
from exllamav2.generator import ExLlamaV2Sampler

from huggingface_hub import snapshot_download
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--bpw", type=str)
args = argparse.parse_args()

from huggingface_hub import snapshot_download
snapshot_download(repo_id="turboderp/Llama-3-8B-Instruct-exl2", revision=f"{args.bpw}bpw", local_dir=f"/work3/s234868/02445/inference/{args.bpw}bpw", local_dir_use_symlinks="auto")

few_shot = """Må en dommer bære religiøse symboler i en retssal i Danmark?
A: Ja
B: Nej
Korrekt: B
Hvor gammel skal man ifølge dansk lov være for at blive gift uden nogen særlig tilladelse?
A: Mindst 14 år
B: Mindst 16 år
C: Mindst 18 år
Korrekt: C
Hvor længe kan politiet tilbageholde en anholdt person, der har lovligt ophold i Danmark , førpersonen skal stilles for en dommer?
A: I op til 1 døgn
B: I op til 3 døgn
C: I op til 1 uge
Korrekt: A
Hvad har regionerne ansvaret for at drive?
A: Politi
B: Sygehuse
C: Børnehaver
Korrekt: B"""

df = pd.read_parquet("citizen.parquet")

df = df.iloc[10:]

prompts = []

for question,a,b,c,correct in zip(df["question"], df["option_a"], df["option_b"], df["option_c"], df["answer"]):
    text = few_shot + "\n" + question
    text += f"\nA: {a}"
    text += f"\nB: {b}"
    if c:
        text += f"\nC: {c}"
    text += f"\nKorrekt:"
    prompts.append(text)

import sys, os
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator

model_dir = f"/work3/s234868/02445/inference/{args.bpw}bpw"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 4096, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters
generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

max_new_tokens = 1

# Warmup generator. The function runs a small completion job to allow all the kernels to fully initialize and
# autotune before we do any timing measurements. It can be a little slow for larger models and is not needed
# to produce correct output.

gen_settings = ExLlamaV2Sampler.Settings(
    temperature = 0.7, 
)

generator.warmup()

answer_list = []

for i in range(500):
    outputs = []
    for prompt_batch in range(0,len(prompts),10):
        outputs += generator.generate(prompt = prompts[prompt_batch:prompt_batch+10], max_new_tokens = max_new_tokens, gen_settings = gen_settings, add_bos = True)
    outputs_answer = [output[-1] for output in outputs]
    answer_list.append(outputs_answer)
    print(f"Completed {i}")

for i in range(500):
    df[f"model_answer_{i}"] = answer_list[i]

df.to_parquet(f"citizen_model_{args.bpw}.parquet")