from vllm import LLM, SamplingParams
import pandas as pd
import torch

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
sampling_params = SamplingParams(temperature=0.7, max_tokens=1)

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.float16)
for i in range(500):
    outputs = []
    outputs += llm.generate(prompts, sampling_params)
    outputs_answer = [output.outputs[0].text[-1] for output in outputs]
    df[f"model_answer_{i}"] = outputs_answer
    print(f"Completed {i}")
df.to_parquet("citizen_model_16.0.parquet")