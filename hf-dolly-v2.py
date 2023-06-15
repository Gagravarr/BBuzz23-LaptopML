#!/usr/bin/env python3
import torch
from transformers import pipeline

generate_text = pipeline(
        model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, 
        trust_remote_code=True, device_map="auto")

res = generate_text("What is Berlin Buzzwords?")
print("")
print(res[0]["generated_text"])

res = generate_text("When was the first Berlin Buzzwords held?")
print("")
print(res[0]["generated_text"])

res = generate_text("What was the venue for the first Berlin Buzzwords?")
print("")
print(res[0]["generated_text"])

res = generate_text("Write a tweet to advertise a talk on laptop-sized AI at a tech conference")
print("")
print(res[0]["generated_text"])
