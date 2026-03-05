from unsloth import FastLanguageModel
import json
import glob

from_model_name = "jarviz"
# judge_model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit"
judge_model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
max_seq_length = 2048

# my model
target_model, target_tokenizer = FastLanguageModel.from_pretrained(
    model_name=from_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(target_model)

# judgement model
judge_model, judge_tokenizer = FastLanguageModel.from_pretrained(
    model_name=judge_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(judge_model)

test_files = glob.glob("tests/*.json")
results = []


def get_answer(model, tokenizer, input_, max_new_tokens=512):
    inputs = tokenizer(input_, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


for file in test_files:
    with open(file) as f:
        tests = json.load(f)

    for test in tests:
        # get answer from my model
        answer = get_answer(
            target_model,
            target_tokenizer,
            test["input"],
        )

        # ask judgement model
        judge_prompt = f"""Question: {test["input"]}
            Expected: {test["expectation"]}
            Model's answer: {answer}

            Is the answer complete and correct? Be strict.
            Reply: PASS or FAIL"""
        judge_answer = get_answer(
            judge_model,
            judge_tokenizer,
            judge_prompt,
            max_new_tokens=10,
        )

        # log result
        verdict = "PASS" if "PASS" in judge_answer else "FAIL"
        print(f"result: {verdict}")
        print(f"question: {test['input']}")
        print(f"answer: {answer}\n")
