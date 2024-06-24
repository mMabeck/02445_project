import transformers
import torch
import time
import pandas as pd


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

tokens_in_prompt = len(pipeline.tokenizer.encode("Once upon a time, in a land"))

print(f"Tokens in prompt: {tokens_in_prompt}")

speeds = []
tokens = []

df = pd.read_parquet("speed.parquet")

for i in range(100):
    start_time = time.time()
    pipe_out = pipeline("Once upon a time, in a land", do_sample=True, min_new_tokens=2000, max_new_tokens=2005, return_tensors=True)
    finish_time = time.time()-start_time
    print()
    new_tokens = len(pipe_out[0]["generated_token_ids"]) - tokens_in_prompt
    speeds.append(new_tokens/finish_time)
    tokens.append(new_tokens)
    print(f"Produced new {new_tokens} tokens")
    print(f"{(new_tokens/finish_time):.2f} tokens per second")
    print(f"Completed {i}")
    if new_tokens < 1800:
        print(outputs)
    print("=====================================")

df[f"speed_16.0bpw"] = speeds
df[f"tokens_16.0bpw"] = tokens
df.to_parquet("speed.parquet")