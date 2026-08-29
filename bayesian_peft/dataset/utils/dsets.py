# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convenience wrappers around classification datasets
"""
import string
import re, os
import torch as t
import pandas as pd

from abc import abstractmethod
from enum import Enum
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

class ClassificationDataset:
    """
    An abstract base dataset for sequence classification problems. Multiple
    choice QA problems could also be made a subclass of this class with an
    appropriate collation / formatting.
    """

    def __init__(
            self,
            dset,
            tokenizer,
            n_labels: int,
            preamble: str = "",
            add_space: bool = False,
            numerical: bool = True,
            boolean: bool = False,
            max_seq_len: int = 512,
    ):
        """
        Args:
            dset: The loaded Dataset
            tokenizer: The model tokenizer
            n_labels: The number of labels / classes for each question
            preamble: Preamble for general pre-trained / 'CausalLM' models
            add_space: Add an explicit space suffix between preamble and answer tokens.
            numerical: whether labels are numerical (0, 1, ...) or alphabetical (A, B, ...)
        """
        self.dset = dset
        self.n_labels = n_labels
        self.preamble = preamble
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.numerical = numerical
        self.max_seq_len = max_seq_len

        spc = " " if self.add_space else ""
        """Token ids of class labels. Example [345, 673, 736]."""
        # TODO: return with enum for question type
        if numerical and boolean:
            raise ValueError("Question type cannot be both numerical and boolean")
        if boolean:
            labels = [f"{spc}True", f"{spc}False"]
        elif numerical:
            labels = [f"{spc}{i}" for i in range(self.n_labels)]
        else:  # alphabetical
            labels = [f"{spc}{chr(ord('A') + i)}" for i in range(self.n_labels)]
        self.target_ids = tokenizer(
            labels, return_tensors="pt", add_special_tokens=False
        ).input_ids[
                          :, -1:
                          ]  # assume these encode to single tokens
        """A mapping from label _indices_ to target token ids. This is only useful for CausalLM models.
        Example: {(0, 345), (1, 673), (2, 736)}
        """
        self.label2target = OrderedDict(
            [(i, self.target_ids[i]) for i in range(n_labels)]
        )
        # misnomer: should be target 2 label _index_
        self.target2label = OrderedDict(
            [(self.target_ids[i], i) for i in range(n_labels)]
        )

    @abstractmethod
    def s2s_collate_fn(self, batch):
        """Collate function for sequence to sequence models"""
        raise NotImplementedError

    def s2s_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for sequence to sequence models"""
        return t.utils.data.DataLoader(
            dset, collate_fn=self.s2s_collate_fn, *args, **kwargs
        )

    @abstractmethod
    def clm_collate_fn(self, batch):
        """Collate function for causal language models"""
        raise NotImplementedError

    def clm_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for causal language models"""
        return t.utils.data.DataLoader(
            dset, collate_fn=self.clm_collate_fn, *args, **kwargs
        )

    def loader(
            self,
            *args,
            is_s2s: bool = False,
            split: str = "train",
            subset_size: int = -1,
            subset_seed: int = 42,
            grad_acc_steps: int = 1,
            drop_last: bool = True,
            **kwargs,
    ):
        if subset_size > 0:
            subset_size = (
                len(self.dset[split])
                if len(self.dset[split]) < subset_size
                else subset_size
            )
            dset = self.dset[split].shuffle(seed=subset_seed).select(range(subset_size))
        else:
            if split:
                dset = self.dset[split]
            else:
                dset = self.dset

        kwargs = {"batch_size": 32, "drop_last": drop_last} | kwargs
        assert (
                kwargs["batch_size"] % grad_acc_steps == 0
        ), "batch size must be divisible by gradient accumulation steps"
        kwargs["batch_size"] = kwargs["batch_size"] // grad_acc_steps

        if is_s2s:
            return self.s2s_loader(dset, *args, **kwargs)
        else:
            return self.clm_loader(dset, *args, **kwargs)

    def _tokenize_prompts(self, prompts):
        prompts = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt",
                                 max_length=self.max_seq_len)
        return prompts

class Book_MIA(ClassificationDataset):
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            add_space: bool = True,
            max_seq_len: int = 512,
            file_path: str = "bookmia"
    ):
        print('load file from ', os.path.join('/home/nusbac/LHD_LLM/CopyCheck/bayesian_peft/database',
                                              file_path + '.jsonl'))
        dset = datasets.Dataset.from_json(
            os.path.join('/home/nusbac/LHD_LLM/CopyCheck/bayesian_peft/database',
                         file_path + '.jsonl'))

        super().__init__(dset, tokenizer, 2, self.preamble, add_space, max_seq_len=max_seq_len)

    # For each sentence below, indicate whether it has been previously encountered (1) or is new (0).

    preamble = """
Sentence: {sentence}
"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["snippet"]) for e in batch]
        prompts = self._tokenize_prompts(prompts)
        classes = t.tensor([e["target"] for e in batch])
        targets = t.cat([self.label2target[e["target"]] for e in batch])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch):
        prompts = [e["snippet"] for e in batch]
        prompts = self._tokenize_prompts(prompts)
        targets = t.tensor([e["target"] for e in batch])
        return prompts, targets, targets


bookmia = Book_MIA



