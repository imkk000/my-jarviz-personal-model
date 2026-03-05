# J.A.R.V.I.Z.

## Disclaimer

Dataset samples were partially drafted with Claude AI assistance,
reviewed and validated by me. Use with your own judgment -
mistakes may exist.

## What is Jarviz?

Jarviz is my personal AI trained to think the way I think -
not a general-purpose assistant, but a reasoning partner
that reflects my own debugging style, decision framework,
and problem-solving instincts.

## Why?

Generic AI knows everything about everyone.
Jarviz knows how I think.

## What it learns from

- How I debug — isolate, hypothesize, test, confirm
- How I decide — trade-offs over opinions
- How I communicate — adapt depth to audience
- How I reason — question the assumption behind the question

## Workflow

```
load_dataset -> format -> trains -> merged -> save to gguf
```

## Prerequisite

- check GPU version: `nvidia-smi`
- Install library

```python
# My CUDA: 13
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install unsloth

# Try 5+ but it's broken
pip install transformers==4.57.6
```

- Install llama.cpp: `yay -S llama.cpp`
- Using `conda` to use `python 3.13`

```
conda config --add channels conda-forge
conda init
conda activate unsloth_env
```

- Llama.cpp in unsloth did not support `MllamaForConditionalGeneration` yet.
- Workaround with build `llama.cpp` from source
- So, I have to install `cuda` with `yay -S cuda`

```
fish_add_path /opt/cuda/bin
set -gx LD_LIBRARY_PATH /opt/cuda/lib64
```

Gave up to vision model from llama 3.2 11B.
I will fallback to `llama 3.1 8B` or `llama 3.2 3B`

### Future Upgrade Path

- [ ] Llama 3.2 11B Vision
  - Blocked: MllamaForConditionalGeneration GGUF conversion not supported
  - Track: [https://github.com/ggml-org/llama.cpp/pull/11292](https://github.com/ggml-org/llama.cpp/pull/11292)
  - Upgrade when: llama.cpp adds convert support

## Prompt

- Need AI to make my dataset output natural (More formal and correct grammar)

```
Clean up this dataset output to sound natural and clear,
keep the original meaning and reasoning exactly as-is:

"[your raw output here]"
```

```
Generate 20 alpaca format samples about [topic].
Mix: what/when/why/scenario-based + correction samples.
Return as JSON array.
```

## Dataset

- Learn to design GOOD and BAD
- Using `behavior` category (learn, reason, explain, teach, decide)
- Start with `alpaca` template with JSON
- Based on `llama`, I format to `llama` template
- Can prepare dataset for text and image content

## Caution

### Remove before start

- For avoiding hallucination model
- merged model (model_name)
- checkpoints (outputs)
- result of model (model_name_gguf)
- unsloth_compiled_cache
- huggingface cache on root home directory

## Resources

- [save-to-gguf](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [llama-notebooks-3.2-11B-vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
- [unsloth](https://unsloth.ai/docs)
- [model](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit)
- [efficient-fine-tuning](learnopencv.com/unsloth-guide-efficient-llm-fine-tuning/)
