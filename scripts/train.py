import os
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model_name = "unsloth/Llama-3.2-11B-Vision-bnb-4bit"
max_seq_length = 2048

os.environ["UNSLOTH_VERBOSE"] = "1"


print(f"cuda: {torch.cuda.get_device_name(0)} ({torch.cuda.is_available()})")

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    dtype=None,
    use_gradient_checkpointing="unsloth",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)


def convert_alpaca_to_messages(example):
    user_content = example["instruction"]
    if example["input"]:
        user_content += f"\n\n{example['input']}"

    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["output"]}],
            },
        ]
    }


def formatting_prompts_func(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


dataset = load_dataset(
    "json", data_files={"train": "./dataset/**/*.json"}, split="train"
)
dataset = dataset.map(convert_alpaca_to_messages)
dataset = dataset.map(formatting_prompts_func, batched=True)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=30,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={},
        max_length=max_seq_length,
    ),
)
trainer.train()

model.save_pretrained_merged(
    "jarviz",
    tokenizer,
    save_method="merged_16bit",
)
