import logging

logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import openai
import torch
import zlib, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from options import Options
from eval import *

llm_path = '/home/nusbac/LHD_LLM/VulCure/my_llm'
jsonl_path = '/home/nusbac/LHD_LLM/CopyCheck/detect-pretrain-code/datasets/'


def load_model(name1):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(os.path.join(llm_path, name1), load_in_8bit=True,
                                                      device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(os.path.join(llm_path, name1))

    return model1, tokenizer1


def read_from_jsonl(file_path):
    file_path = os.path.join(jsonl_path, file_path) + ".jsonl"
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculatePerplexity_gpt3(prompt, modelname):
    prompt = prompt.replace('\x00', '')
    responses = None
    # Put your API key here
    openai.api_key = "YOUR_API_KEY"  # YOUR_API_KEY
    while responses is None:
        try:
            responses = openai.Completion.create(
                engine=modelname,
                prompt=prompt,
                max_tokens=0,
                temperature=1.0,
                logprobs=5,
                echo=True)
        except openai.error.InvalidRequestError:
            print("too long for openai API")
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)


def calculatePerplexity(sentence, model, tokenizer, gpu, max_length=300):
    encoded = tokenizer(
        sentence,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(gpu)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss = outputs.loss
    logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    all_prob = []
    target_ids = input_ids[0, 1:]

    for i, token_id in enumerate(target_ids):
        token_log_prob = log_probs[0, i, token_id].item()
        all_prob.append(token_log_prob)

    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model1, tokenizer1, text, ex):
    pred = {}

    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    # ppl
    pred["ppl"] = p1
    #  # Ratio of log ppl of large and small models
    #  pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred
    return ex


def evaluate_data(test_data, model1, tokenizer1, col_name):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data):
        text = ex[col_name]
        new_ex = inference(model1, tokenizer1, text, ex)
        all_output.append(new_ex)
    return all_output


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=1 python3 src/run.py --target_model meta-llama/Llama-2-7b --data bookmia_25_book
    output_dir = "./output"
    args = Options()
    args = args.parser.parse_args()
    llm_name = {'huggyllama/llama-7b': 'llama-7b',
                'meta-llama/Llama-2-7b': 'llama2-7b',
                'huggyllama/llama-13b': 'llama-13b',
                'meta-llama/llama-2-13b': 'llama2-13b'
                }

    model1, tokenizer1 = load_model(args.target_model)
    data = read_from_jsonl(f"{args.data}")

    key_name = 'snippet'

    all_output = evaluate_data(data, model1, tokenizer1, key_name)

    from collections import defaultdict

    metric2predictions = defaultdict(list)
    for ex in all_output:
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])

    data = []
    print(metric2predictions.keys())
    for k, v in metric2predictions.items():
        data.append(v)
    data = np.array(data).T
    if not os.path.isdir(os.path.join(output_dir, llm_name[args.target_model])):
        os.makedirs(os.path.join(output_dir, llm_name[args.target_model]))
    np.save(os.path.join(os.path.join(output_dir, llm_name[args.target_model]), args.data), data)
