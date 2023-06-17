# Laptop-sized ML for Text, with Open Source

## Code and Slides for my Berlin Buzzwords 2023 Talk

Berlin Buzzwords 2023, 06-19, 15:20–16:00 (Europe/Berlin), Palais Atelier 

https://program.berlinbuzzwords.de/berlin-buzzwords-2023/talk/JTD7GY/

## Getting started with LLaMA
Follow the instructions on https://github.com/facebookresearch/llama
for getting access from Facebook to the model files. May take a few days
for Accademic email addresses, weeks/months for everyone else...

Once you have access, consider setting up a `venv` (similar to what you'd
do for Hugging Face below), then follow
https://github.com/facebookresearch/llama#setup
https://github.com/facebookresearch/llama#download
https://github.com/facebookresearch/llama#inference

The official LLaMA code is slow, so once you have your models downloaded,
consider using LLaMA.cpp to run them

## Getting started with LLaMA.cpp
Make sure you have a working C++ compiler!

To build, follow https://github.com/ggerganov/llama.cpp#usage

Then convert the Facebook-provided LLaMA models into the smaller
and quantised form by following
https://github.com/ggerganov/llama.cpp#prepare-data--run

Then give it a try by running `./examples/chat.sh`

## Getting started with Hugging Face
You'll most likely want to use a Python Virtual Env to store all the 
dependencies, so you make sure you get the required versions no matter
what your system ships with

See https://huggingface.co/docs/datasets/installation

You may need to `apt-get install apt python3-venv` or `pip install venv` 
before you do these steps

```
mkdir HuggingFace
cd HuggingFace
python3 -m venv .env
source .env/bin/activate
pip install datasets
pip install torch transformers command 
pip install sentencepiece einops accelerate

python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"
```

Note that Hugging Face will download all the models to `~/.cache/huggingface/`
by default, you may want to symlink that elsewhere or play with the 
`TRANSFORMERS_CACHE` environment variable, if you like to keep all your 
models on a different disk.

### Hugging Face + LLaMA
Run `hf-llama.py` then wait a long time while it downloads!

### Hugging Face + MPT-7B-Instruct
See https://www.mosaicml.com/blog/mpt-7b

> MPT-7B-Instruct is a model for short-form instruction following. Built 
> by finetuning MPT-7B on a dataset we also release, derived from 
> Databricks Dolly-15k and Anthropic’s Helpful and Harmless datasets.

Run `hf-mpt-7b-instruct.py` then wait a long time while it downloads!

## LLaMA.cpp and experimental Grammar extension

https://twitter.com/GrantSlatton/status/1657559506069463040
https://github.com/grantslatton/llama.cpp

On Ubuntu (plus some others), may get a compiler error from g++. If so,
use https://github.com/Gagravarr/llama.cpp

```
cat > example-grammar_revolution.txt << EOF
country_adjective = "American" | "French" | "Iranian"
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
year = <digit> <digit> <digit> <digit>
sentence = "The " <country_adjective> " revolution started in " <year> "."
EOF

python3 grammar.py --grammar example-grammar_revolution.txt > compiled_revolution
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 --grammar compiled_revolution -p "America"
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 --grammar compiled_revolution -p "France"
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 --grammar compiled_revolution -p "Iran"
```

```
cat > example-grammar_tourism-json.txt << EOF
type CityData = {
   "city_name": string,
   "country": string,
   "reasons_to_visit": {
      "perecent_sunny_days": number,
      "top_3_tourist_attractions": [string, string, string],
   },
};
EOF

python3 grammar.py --json grammar_json.txt > compiled_tourism
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 --grammar compiled_tourism -p "New York"
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 --grammar compiled_tourism -p "Sydney"
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 --grammar compiled_tourism -p "Berlin"
```
