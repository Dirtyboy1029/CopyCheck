# -*- coding: utf-8 -*- 
# @Time : 2024/12/19 17:13 
# 
# @File : demo2.py


import os, random
from datasets import Dataset

# 设置路径
books_dir = "/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/deepmind-gutenberg/eBooks"  # 修改为你的书籍文件夹路径

# 创建数据集列表
data = []
data_num = len(os.listdir(books_dir))
need_id = random.sample(list(range(data_num)), k=2500)

for book_id, filename in enumerate(os.listdir(books_dir)):
    if book_id in need_id:
        if filename.endswith('.txt'):
            book_path = os.path.join(books_dir, filename)

            # 读取文本文件内容
            with open(book_path, 'r', encoding='utf-8') as f:
                text = f.read()

            book = {
                "book": filename,  # 文件名作为书名（去掉后缀）
                "bookid": book_id + 1,  # 假设书的ID是按顺序排列的
                "text": text,
                "label": 0
            }
            data.append(book)

# 创建 Hugging Face Dataset
dataset = Dataset.from_dict({
    "book": [entry["book"] for entry in data],
    "bookid": [entry["bookid"] for entry in data],
    "text": [entry["text"] for entry in data],
    "label": [entry["label"] for entry in data]
})

# 查看前几条数据
print(dataset[:5])

# 保存数据集为 `.hg` 格式
dataset.save_to_disk('/opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/Database/pg19_gutenberg/non_member')
