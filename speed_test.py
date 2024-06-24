import pandas as pd
from exllamav2.generator import ExLlamaV2Sampler
import argparse
import time

import sys, os
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer, Timer
from exllamav2.generator import ExLlamaV2DynamicGenerator

argparse = argparse.ArgumentParser()
argparse.add_argument("--bpw", type=str)
args = argparse.parse_args()

from huggingface_hub import snapshot_download
snapshot_download(repo_id="turboderp/Llama-3-8B-Instruct-exl2", revision=f"{args.bpw}bpw", local_dir=f"/work3/s234868/02445/{args.bpw}bpw", local_dir_use_symlinks="auto")

prompt = "Once upon a time, in a land"

model_dir = f"/work3/s234868/02445/{args.bpw}bpw"
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

max_new_tokens = 2005

# Warmup generator. The function runs a small completion job to allow all the kernels to fully initialize and
# autotune before we do any timing measurements. It can be a little slow for larger models and is not needed
# to produce correct output.

gen_settings = ExLlamaV2Sampler.Settings(
    temperature = 0.7,
)

generator.warmup()

df = pd.read_parquet("speed.parquet")

speeds = []
tokens = []

for i in range(100):
    start_time = time.time()
    outputs = generator.generate(prompt = prompt, min_new_tokens = 2000, max_new_tokens = max_new_tokens, gen_settings = gen_settings, add_bos = True, decode_special_tokens = True, completion_only = True)
    finish_time = time.time()-start_time
    new_tokens = tokenizer.encode(outputs).shape[1]
    speeds.append(new_tokens/finish_time)
    tokens.append(new_tokens)
    print(f"Produced new {new_tokens} tokens")
    print(f"{(new_tokens/finish_time):.2f} tokens per second")
    print(f"Completed {i}")
    if new_tokens < 1800:
        print(outputs)
    print("=====================================")

df[f"speed_{args.bpw}bpw"] = speeds
df[f"tokens_{args.bpw}bpw"] = tokens
df.to_parquet("speed.parquet")

#https://huggingface.co/turboderp/Mixtral-8x7B-exl2
#https://github.com/turboderp/exllamav2/issues/232#issuecomment-1860896496
#https://arxiv.org/pdf/2401.06118