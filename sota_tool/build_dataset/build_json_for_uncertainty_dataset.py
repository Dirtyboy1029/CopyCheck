# -*- coding: utf-8 -*- 
# @Time : 2025/1/19 20:17 
#  
# @File : build_json_for_uncertainty_dataset.py
from datasets import Dataset
from datasets import load_from_disk
import json, re, os
import numpy as np
from transformers import LlamaTokenizer
from tqdm import tqdm

name_dict = {'my_exp_testset1': 'bookmia_test_seen10',
             'my_exp_testset2': 'bookmia_test_seen20',
             'my_exp_testset3': 'bookmia_test_seen30',
             'my_exp_testset4': 'bookmia_test_seen40', }


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def main_(source_data, json_name):
    my_tokenizer = LlamaTokenizer.from_pretrained(
        os.path.join('/opt/data/private/LHD_LLM/LLM_uncertainty/my_llm/openlm-research/open_llama_7b'))
    dataset = load_from_disk(
        f"/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/{source_data}")
    unseen_dataset = load_from_disk(
        "/opt/data/private/LHD_LLM/LLM_uncertainty/document-level-membership-inference/data/comparative_experiments_dataset/my_exp_unseenset")
    def write_to_jsonl(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')
    def read_from_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    def re_kongge(text):
        return re.sub(r'\n+', ' ', text)
    def split_text_by_tokens(text, max_tokens=600):
        tokens = my_tokenizer.tokenize(text)
        return [re_kongge(my_tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])) for i in
                range(12000, len(tokens), max_tokens)]
    def split_text_into_snippets(dataset, new_label, base_index):
        new_data = []
        for idx, example in enumerate(tqdm(dataset, desc='split books')):
            book_id = idx
            book_title = example['book']
            text = example['text']
            segments = split_text_by_tokens(text)
            num_snippets = len(segments)
            if num_snippets > 100:
                segments = segments[0:100]
            if num_snippets >= 95:
                for snippet_id, snippet in enumerate(segments):
                    new_data.append({
                        'book_id': int(book_id) + base_index,
                        'book': book_title,
                        'snippet_id': snippet_id + 1,
                        'snippet': snippet,
                        'label': new_label
                    })
        return Dataset.from_dict({key: [entry[key] for entry in new_data] for key in new_data[0].keys()})
    conf = {'unseen_id': list(range(50, 100)),
            'suspected_seen_id': list(np.where(np.array(dataset['label']) == 1)[0]),
            'suspected_unseen_id': list(np.where(np.array(dataset['label']) == 0)[0])}
    dump_joblib(conf, os.path.join('/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/database/config',
                                   json_name.replace('bookmia', 'bookid') + '.conf'))
    from datasets import concatenate_datasets
    my_dataset = concatenate_datasets(
        [split_text_into_snippets(dataset, new_label=1, base_index=0),
         split_text_into_snippets(unseen_dataset, new_label=0, base_index=50)])
    my_dataset = my_dataset.shuffle(seed=42)

    write_to_jsonl(my_dataset, f'/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/database/{json_name}.jsonl')
    print('save json file to ', f'/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/database/{json_name}.jsonl')
    my_dataset = read_from_jsonl(f'/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/database/{json_name}.jsonl')
    labels = np.array([item['label'] for item in my_dataset])
    print(len(np.where(labels == 1)[0]))
    print(len(np.where(labels == 0)[0]))
    print(len(labels))


if __name__ == '__main__':
    for k, v in name_dict.items():
        main_(k, v)
