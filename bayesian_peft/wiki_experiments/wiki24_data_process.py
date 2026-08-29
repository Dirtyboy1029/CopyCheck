# -*- coding: utf-8 -*- 
# @Time : 2025/10/10 18:08 
# @Author : DirtyBoy 
# @File : wiki24_data_process.py
import pyarrow.parquet as pq
import random, json

total_num = 300


def split_by_word_count(text, words_per_chunk=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    return chunks


def show_basic_analysis(data_list):
    total_samples = len(data_list)
    member_count = sum(1 for item in data_list if item['ismember'] == 1)
    non_member_count = total_samples - member_count
    label_positive = sum(1 for item in data_list if item['label'] == 1)
    label_negative = total_samples - label_positive

    print("=== BASIC DATA ANALYSIS ===")
    print(f"Total samples: {total_samples}")
    print()
    print("IsMember Distribution:")
    print(f"  Member (1): {member_count} ({member_count / total_samples * 100:.1f}%)")
    print(f"  Non-Member (0): {non_member_count} ({non_member_count / total_samples * 100:.1f}%)")
    print()
    print("Label Distribution:")
    print(f"  Positive (1): {label_positive} ({label_positive / total_samples * 100:.1f}%)")
    print(f"  Negative (0): {label_negative} ({label_negative / total_samples * 100:.1f}%)")
    print()


def write_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')


def main_(noise_ratio=20):
    file_path = '../database/WikiMIA_24/WikiMIA_length256-00000-of-00001-3528649663c073f0.parquet'
    table = pq.read_table(file_path)

    file_path1 = '../database/WikiMIA_24/WikiMIA_length128-00000-of-00001-5c8bfd2338c7ceac.parquet'
    table1 = pq.read_table(file_path1)
    my_data = table.to_pandas().to_dict('records') + table1.to_pandas().to_dict('records')
    print(len(my_data))

    print(len([item for item in my_data if item['label'] == 1]))

    my_member = [item for item in my_data if item['label'] == 1]
    my_nonmember = [item for item in my_data if item['label'] == 0]

    true_member = random.sample(my_member, k=int(total_num * (50 - noise_ratio) // 100))
    print('there are ', len(true_member), ' true member.')

    false_member = random.sample(my_nonmember, k=(total_num // 2 - len(true_member)))
    print('there are ', len(false_member), ' false member.')
    exist_nonmember = [item for item in my_nonmember if item not in false_member]

    true_nonmember = random.sample(exist_nonmember, k=(total_num // 2))
    print('there are ', len(true_nonmember), ' true nonmember.')

    my_snippets = []

    my_true_member_snippets = []
    for item in true_member:
        tmp_snippets = split_by_word_count(item['input'], 30)
        tmp_snippets = random.sample(tmp_snippets, k=4)
        my_true_member_snippets = my_true_member_snippets + tmp_snippets
    for i, item in enumerate(my_true_member_snippets):
        my_snippets.append({'id': i + 1,
                            'snippet': item,
                            'ismember': 1,
                            'label': 1})

    my_false_member_snippets = []
    for item in false_member:
        tmp_snippets = split_by_word_count(item['input'], 30)
        tmp_snippets = random.sample(tmp_snippets, k=4)
        my_false_member_snippets = my_false_member_snippets + tmp_snippets
    for i, item in enumerate(my_false_member_snippets):
        my_snippets.append({'id': i + 10001,
                            'snippet': item,
                            'ismember': 0,
                            'label': 1})

    my_true_nonmember_snippets = []
    for item in true_nonmember:
        tmp_snippets = split_by_word_count(item['input'], 30)
        tmp_snippets = random.sample(tmp_snippets, k=4)
        my_true_nonmember_snippets = my_true_nonmember_snippets + tmp_snippets
    for i, item in enumerate(my_true_nonmember_snippets):
        my_snippets.append({'id': i + 20001,
                            'snippet': item,
                            'ismember': 0,
                            'label': 0})

    show_basic_analysis(my_snippets)
    write_to_jsonl(my_snippets, f'bookmia_bonlyseen_{noise_ratio}_wiki24.jsonl')


if __name__ == '__main__':
    for noise in [10, 20, 30, 40]:
        main_(noise)
