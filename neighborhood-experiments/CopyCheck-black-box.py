#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/8/14 10:46
"""

key = "sk-xx"
from openai import OpenAI
import pyarrow.parquet as pq
from tqdm import tqdm
import json, os

client = OpenAI(
    api_key=key,
    # base_url="YOUR_BASE_URL"  # 官方 OpenAI 可删除这一行
)


def generate_answer(prompt, model_name):
    if model_name == "gpt-3.5-turbo-instruct":

        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
        )

        answer = response.choices[0].text

    else:

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100,
            temperature=0.7,
        )

        answer = response.choices[0].message.content

    return answer


def write_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')


import time
import random


def append_to_jsonl(data, file_path):
    """
    Append one record to a jsonl file immediately.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()


def get_finished_num(file_path):
    """
    Count how many samples have already been successfully saved.
    """
    if not os.path.isfile(file_path):
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def generate_with_retry(
        text,
        model_name,
        max_retries=10,
        base_sleep=5,
):
    """
    Generate one answer with automatic retry.

    If the API rejects the request or a network error occurs,
    wait and retry with exponential backoff.
    """

    for retry in range(max_retries):
        try:
            answer = generate_answer(
                text,
                model_name=model_name,
            )

            return answer

        except Exception as e:
            print(
                f"\n[Error] retry {retry + 1}/{max_retries}: {e}"
            )

            # exponential backoff:
            # 5, 10, 20, 40, 60, 60, ...
            sleep_time = min(
                base_sleep * (2 ** retry),
                60
            )

            # add a little randomness to avoid repeatedly
            # hitting the service at the same interval
            sleep_time += random.uniform(0, 3)

            print(
                f"Sleep {sleep_time:.1f}s before retry..."
            )

            time.sleep(sleep_time)

    raise RuntimeError(
        f"Failed after {max_retries} retries."
    )


if __name__ == "__main__":

    length = 32

    file_path = f"WikiMIA_24/WikiMIA_length{length}.parquet"

    table = (
        pq.read_table(file_path)
        .to_pandas()
        .to_dict("records")
    )

    for i in range(9, 10):

        save_path = f"Answer/wiki_{length}_{i}.jsonl"

        # -----------------------------------------
        # Resume from previous progress
        # -----------------------------------------
        finished_num = get_finished_num(save_path)

        print(
            f"\nRun {i}: "
            f"{finished_num}/{len(table)} samples already completed."
        )

        # -----------------------------------------
        # Continue from breakpoint
        # -----------------------------------------
        remaining_table = table[finished_num:]

        for index, item in tqdm(
                enumerate(
                    remaining_table,
                    start=finished_num
                ),
                total=len(remaining_table),
                desc=f"Generating answers - Run {i}"
        ):

            try:

                answer = generate_with_retry(
                    item["input"],
                    model_name="gpt-3.5-turbo-instruct",
                )

                # ---------------------------------
                # Save immediately after success
                # ---------------------------------
                record = {
                    "index": index,
                    "answer": answer,
                }

                append_to_jsonl(
                    record,
                    save_path
                )

                # # ---------------------------------
                # # Avoid overly frequent API calls
                # # ---------------------------------
                # time.sleep(
                #     random.uniform(1.0, 2.0)
                # )

            except Exception as e:

                print(
                    f"\n[Failed] index={index}"
                )
                print(e)

                # Do NOT skip this sample,
                # otherwise resume-by-line-count
                # would become misaligned.
                print(
                    "Program stopped. "
                    "Run it again to resume from this sample."
                )

                raise
