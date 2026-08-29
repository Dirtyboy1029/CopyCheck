import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + '/datasets')
sys.path.append(project_path + '/backbones')
sys.path.append(project_path + '/models')
sys.path.append(project_path + '/main')

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from utils.args import add_management_args, add_experiment_args
from utils import create_if_not_exists
# from utils.continual_training import train as ctrain
from run.ood_eval import ood_eval
from run.laplace_train import laplace_train_old
from run.laplace_ood_eval import laplace_ood_eval
from run.laplace_ood_vis import laplace_ood_vis
from run import *

from accelerate.utils import set_seed
from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='Bayesian LoRA', allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_known_args()[0]

    # add model-specific arguments
    mod = importlib.import_module('modelwrappers.' + args.modelwrapper)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()  # the real parsing happens.
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        set_seed(args.seed)

    return args


# @iex
def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    accelerator = Accelerator()
    #
    ood_ori_dataset = None
    if args.ood_ori_dataset is not None:
        dataset = args.dataset
        args.dataset = args.ood_ori_dataset
        ood_ori_dataset = get_dataset(args.dataset_type, accelerator, args)
        ood_ori_dataset.get_loaders()
        args.ood_ori_outdim = ood_ori_dataset.num_labels  # should be careful to use in evaluate_all
        args.ood_ori_num_samples = ood_ori_dataset.num_samples
        args.dataset = dataset

    dataset = get_dataset(args.dataset_type.split('_')[0], accelerator, args)
    dataset.get_loaders()
    args.outdim = dataset.num_labels
    args.num_samples = dataset.num_samples

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer

    model_name = args.load_model_path
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA 没有 pad_token，需要设置

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,  # 省显存
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    def read_from_jsonl(file_path):
        import json
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    data = read_from_jsonl('/home/nusbac/LHD_LLM/uncertainty/LLM_uncertainty/bayesian_peft/database/bookmia_demo1.jsonl')
    N = len(data)  # 样本数
    per_device_train_batch_size = 2  # per_device_train_batch_size
    gradient_accumulation_steps = 8  # gradient_accumulation_steps
    num_train_epochs = 1  # num_train_epochs

    # =========================
    # 4. 训练参数
    # =========================
    training_args = TrainingArguments(
        output_dir="./llama-7b-lora-gen",
        per_device_train_batch_size=per_device_train_batch_size,  # 小显存推荐 1
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        max_steps=(N + per_device_train_batch_size * gradient_accumulation_steps - 1) // (
                per_device_train_batch_size * gradient_accumulation_steps) * num_train_epochs
    )

    # =========================
    # 5. Trainer
    # =========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.dset,
        tokenizer=tokenizer
    )

    # =========================
    # 6. 开始训练
    # =========================
    trainer.train()


if __name__ == '__main__':
    main()
