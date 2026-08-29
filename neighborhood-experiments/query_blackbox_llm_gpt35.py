# -*- coding: utf-8 -*- 
# @Time : 2025/10/12 13:18 
# @Author : DirtyBoy 
# @File : query_blackbox_llm_gpt35.py
# import openai, os, json, time
# import numpy as np
# from scipy.special import softmax
import pyarrow.parquet as pq
from tqdm import tqdm

key = 'xx'
openai.api_key = key


def perplexity_via_completion(text: str):
    prompt = [
        f"Text: {text} \nDetermine whether you have seen this before. Output 1 for yes, 0 for no. Only output 0 or 1. ",
        f"Text: {text} \nHas this text been included in your training data? Output 1 for yes, 0 for no. Only output 0 or 1. ",
        f"Text: {text} \nIdentify if you have been trained on this text. Return 1 for yes, 0 for no. Strictly 0 or 1 only.",
        f"Text: {text} \nDecide whether this text was seen during your training. 1 = yes, 0 = no. Output only 0 or 1.",
        f"Text: {text} \nState if this text exists in your training data. 1 means yes, 0 means no. Output exactly 0 or 1.",
        f"Text: {text} \nHave you encountered this text before? Reply 1 for yes, 0 for no. Only provide 0 or 1.",
        f"Text: {text} \nIndicate whether you recognize this text from your training. Use 1 for yes, 0 for no. Output strictly 0 or 1.",
        f"Text: {text} \nHas this text appeared in your training data? Output 1 for yes, 0 for no. Only output 0 or 1.",
        f"Text: {text} \nCheck if you have seen this text before. 1 = yes, 0 = no. Output only 0 or 1.",
        f"Text: {text} \nDetermine if this text is part of your training data. Return 1 for yes, 0 for no. Only 0 or 1."
    ]
    resp = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",   #gpt-5-nano  gpt-4.1-mini   gpt-4o-mini
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        logprobs=100
    )

    logprobs = [resp["choices"][i]["logprobs"]["top_logprobs"][0] for i in range(len(prompt))]

    return logprobs


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    file_path = 'WikiMIA_24/WikiMIA_length32.parquet'
    if 'parquet' in file_path:
        table = pq.read_table(file_path).to_pandas().to_dict('records')
        key = 'input'

    else:
        table = read_from_jsonl(file_path)
        if 'bookmia' in file_path:
            key = 'snippet'
        elif 'wiki' in file_path:
            key = 'input'
        else:
            key = 'text'
    save_folder = os.path.join('logprobs', os.path.splitext(os.path.basename(file_path))[0])
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    my_data = table[0:1]
    for index, item in tqdm(list(enumerate(my_data)), desc="Processing", total=len(my_data)):
        if  os.path.isfile(os.path.join(save_folder, f"{index}.json")):
            text = item[key]
            try:
                logprobs = perplexity_via_completion(text)
                save_json(logprobs, os.path.join(save_folder, f"{index}.json"))
            except Exception as e:
                print(e)
                time.sleep(15)
