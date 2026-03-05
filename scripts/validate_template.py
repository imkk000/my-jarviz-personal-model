from transformers import AutoTokenizer

from_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(from_model_name)

test = "use metaphors — go deeper"
tokens = tokenizer.encode(test)
decoded = tokenizer.decode(tokens)
print(decoded)


print(tokenizer.eos_token)
print(tokenizer.eos_token_id)

messages = [
    {"role": "user", "content": "Hi Jarviz"},
    {"role": "assistant", "content": "Hey! What are you working on?"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print(text)
