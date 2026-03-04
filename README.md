# JAVIZ

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
- Based on `llama`, I should format to `llama` template in the future

## Resources

- [llama-notebooks](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Conversational.ipynb)
- [unsloth](https://unsloth.ai/docs)
- [model](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit)
- [efficient-fine-tuning](learnopencv.com/unsloth-guide-efficient-llm-fine-tuning/)
