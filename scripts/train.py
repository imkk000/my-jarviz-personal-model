import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, concatenate_datasets
from unsloth.chat_templates import get_chat_template

os.environ["UNSLOTH_VERBOSE"] = "1"

print(f"cuda enabled: {torch.cuda.is_available()}")
print(f"device: {torch.cuda.get_device_name(0)}")

from_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
to_model_name = "jarviz"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=from_model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto selected
    load_in_4bit=True,
    load_in_8bit=False,
    load_in_16bit=False,
    full_finetuning=False,
    trust_remote_code=False,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)


# load system prompt
with open("config/system_prompt.md") as f:
    SYSTEM_PROMPT = f.read()


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input_, output in zip(instructions, inputs, outputs):
        # Build input content
        user_content = instruction
        if input_:
            user_content += f"\n\n{input_}"

        # Llama 3 chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

        # tokenize_with_template adds special tokens + eos automatically
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


knowledges_dataset = load_dataset(
    "json", data_files={"train": "./dataset/knowledges/**/*.json"}, split="train"
)
instructions_dataset = load_dataset(
    "json", data_files={"train": "./dataset/instructions/**/*.json"}, split="train"
)
dataset = concatenate_datasets(
    [
        knowledges_dataset,
        instructions_dataset.map(formatting_prompts_func, batched=True),
    ]
)
dataset = dataset.shuffle(seed=42)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        max_steps=-1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()

model.save_pretrained_gguf(to_model_name, tokenizer, quantization_method="q4_k_m")
