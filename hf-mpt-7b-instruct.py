#!/usr/bin/env python3
import transformers

modelname = 'mosaicml/mpt-7b-instruct'
print("Loading model %s" % modelname)

model = transformers.AutoModelForCausalLM.from_pretrained(
  modelname, trust_remote_code=True
)

# Turn the text into tokens
batch = tokenizer(
    "What is Berlin Buzzwords?",
    return_tensors="pt", 
    add_special_tokens=False
)

# Have the model run against it
batch = {k: v.to(device) for k, v in batch.items()}
generated = model.generate(batch["input_ids"], max_length=100)

# Turn that back into text and report
print("")
print(tokenizer.decode(generated[0]))
