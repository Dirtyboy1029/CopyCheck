import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex
import numpy as np
from tqdm import tqdm

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + '/datasets')
sys.path.append(project_path + '/backbones')
sys.path.append(project_path + '/models')
sys.path.append(project_path + '/main')

from transformers import AutoTokenizer
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

import numpy as np


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


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
    from tqdm import tqdm

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

    # print('main.py print \
    #         target_ids:', dataset.target_ids, )

    args.num_samples = dataset.num_samples
    model = get_model(args, accelerator)

    # set job name
    setproctitle.setproctitle('{}_{}_BLoB-lora'.format(args.model, args.dataset))
    # train the model
    if args.laplace_vis:
        laplace_ood_vis(model, dataset, accelerator, args, ood_ori_dataset)
    if args.ood_ori_dataset is not None and args.laplace_train:
        laplace_ood_eval(model, dataset, accelerator, args, ood_ori_dataset)
    elif args.laplace_train:
        laplace_train_old(model, dataset, accelerator, args)
    elif args.ood_ori_dataset is not None:
        if 'llama' in args.model or 'Qwen' in args.model or 'starcoder' in args.model or 'deepseek' in args.model or 'opt' in args.model or 'gpt' in args.model:
            if args.model_type == "seqcls":
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                print("CUDA_VISIBLE_DEVICES:", cuda_visible)
                device = torch.device('cuda:0')
                index_0, index_1, index_2, index_3 = None, None, None, None
                model.to(device)

                if args.modelwrapper.startswith('deepensemble'):
                    tmp_path = f'output/{args.model_type}/deepensemble/' + args.dataset + '/' + args.ood_ori_dataset + f'/{args.load_model_path}_1.data'
                    if not os.path.isdir(os.path.dirname(tmp_path)):
                        print(os.path.dirname(tmp_path))
                        os.makedirs(os.path.dirname(tmp_path))
                    n = args.ensemble_n
                    endings = [[] for _ in range(n)]
                    for step, batch in enumerate(tqdm(ood_ori_dataset.test_dataloader, desc="Predict Progress")):
                        with torch.no_grad() and torch.inference_mode():
                            logits = np.squeeze(model.model.forward_logits(batch).detach().cpu().numpy()).transpose(1,
                                                                                                                    0,
                                                                                                                    2)
                            for i in range(n):
                                if 'multi_c' in args.ood_ori_dataset:
                                    output = [np.array([item[index_0], item[index_1], item[index_2], item[index_3]]) for
                                              item in
                                              logits[i]]
                                else:
                                    output = [np.array([item[index_0], item[index_1]]) for item in logits[i]]
                                print(f"output{i + 1}: {output}")
                                endings[i] = endings[i] + output
                    for i in range(n):
                        dump_joblib(endings[i],
                                    f'output/{args.model_type}/deepensemble/' + args.dataset + '/' + args.ood_ori_dataset + f'/{args.load_model_path}_{i + 1}.data')

                else:
                    if 'multi' in args.dataset:
                        output_dim = 4
                    else:
                        output_dim = 2
                    ending = np.empty((0, output_dim))

                    save_path = f'output/{args.model_type}/{args.modelwrapper}/{args.dataset}/{args.ood_ori_dataset}/epoch{args.n_epochs}/{args.model}-tmp_index.data'

                    if not os.path.isdir(os.path.dirname(save_path)):
                        print(os.path.dirname(save_path))
                        os.makedirs(os.path.dirname(save_path))
                    print('save to ', os.path.dirname(save_path))

                    for i, batch in enumerate(tqdm(ood_ori_dataset.test_dataloader, desc="Predict Progress")):
                        # try:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        logits = model.model.forward_logits(batch, sample=True, n_samples=1)
                        output = logits.detach().cpu().numpy().squeeze(axis=1)
                        
                        ending = np.concatenate([ending, output], axis=0)
                    # except Exception as e:
                    #     print(e)
                    #     output = np.zeros((1,2))
                    #     ending = np.concatenate([ending, output], axis=0)
                    dump_joblib(ending, save_path.replace('tmp_index', str(args.bayes_eval_index)))
            elif args.model_type == "causallm":
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                print("CUDA_VISIBLE_DEVICES:", cuda_visible)
                device = torch.device('cuda:0')
                model.to(device)
                model.model.eval()
                target_ids = ood_ori_dataset.target_ids
                if 'multi_label' in args.dataset:
                    output_dim = 4
                else:
                    output_dim = 2
                if args.modelwrapper.startswith('deepensemble') or args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('mcdropout'):
                    if args.modelwrapper.startswith('deepensemble'):
                        nn = 1
                    else:
                        nn = 10

                    save_path = f'output/{args.model_type}/{args.modelwrapper}/{args.dataset}/{args.ood_ori_dataset}/epoch{args.n_epochs}/{args.model}-tmp_index.data'
                    if not os.path.isdir(os.path.dirname(save_path)):
                        print(os.path.dirname(save_path))
                        os.makedirs(os.path.dirname(save_path))

                    endings = [np.empty((0, output_dim)) for _ in range(10)]

                    for step, batch in enumerate(tqdm(ood_ori_dataset.test_dataloader, desc="Predict Progress")):

                        inputs, labels, extra = batch

                        inputs = inputs.to(device)

                        labels = labels.to(device)
                        extra = extra.to(device)

                        with torch.no_grad():
                            logits = model.model.forward_logits(
                                (inputs, labels, extra),
                                sample=True,
                                n_samples=nn
                            )
                        output = logits.detach().cpu().numpy()

                        output = np.squeeze(output[:, :, -1, target_ids])
                        output = output.transpose(1, 0, 2)

                        for i in range(10):
                            endings[i] = np.concatenate([endings[i], output[i]], axis=0)

                    for j in range(10):
                        dump_joblib(endings[j],
                                    save_path.replace("tmp_index", str(j + 1)))
                else:

                    ending = np.empty((0, output_dim))

                    save_path = f'output/{args.model_type}/{args.modelwrapper}/{args.dataset}/{args.ood_ori_dataset}/epoch{args.n_epochs}/{args.model}-tmp_index.data'

                    if not os.path.isdir(os.path.dirname(save_path)):
                        print(os.path.dirname(save_path))
                        os.makedirs(os.path.dirname(save_path))
                    print('save to ', os.path.dirname(save_path))

                    device = next(model.parameters()).device

                    for step, batch in enumerate(tqdm(ood_ori_dataset.test_dataloader, desc="Predict Progress")):
                        inputs, labels, extra = batch

                        inputs = inputs.to(device)

                        labels = labels.to(device)
                        extra = extra.to(device)

                        with torch.no_grad():
                            logits = model.model.forward_logits(
                                (inputs, labels, extra),
                                sample=True,
                                n_samples=1
                            )

                        output = logits.detach().cpu().numpy()

                        output = np.squeeze(output[:, :, -1, target_ids])
                        if args.batch_size == 1:
                            output = output.reshape((1, args.outdim))
                        print(output)
                        ending = np.concatenate([ending, output], axis=0)
                    dump_joblib(ending, save_path.replace('tmp_index', str(args.bayes_eval_index)))

    else:
        wandb_logger = None
        if accelerator.is_local_main_process:
            print(args)
            if not args.nowand:
                assert wandb is not None, "Wandb not installed, please install it or run without wandb"
                if not args.wandb_name:
                    wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
                else:
                    wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                              name=args.wandb_name, config=vars(args))
            print(file=sys.stderr)

        model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
        model.model.fit_evaluate()

        # checkpointing the backbone model.
        if args.checkpoint:  # by default the checkpoints folder is checkpoints
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if "_tfblora" in args.dataset:
                    ds_name = args.dataset.replace("_tfblora", "")
                    save_folder = f'checkpoints/{args.model_type}_{args.modelwrapper}/{args.model}/{ds_name}/{args.checkpoint_dic_name}'
                else:
                    save_folder = f'checkpoints/{args.model_type}_{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_dic_name}'
                create_if_not_exists(save_folder)
                model.model.base_model = accelerator.unwrap_model(model.model.base_model)
                model.model.save_pretrained(save_folder, save_function=accelerator.save)
                print('Model saved to:', save_folder)

        if not args.nowand:
            if accelerator.is_local_main_process:
                wandb_logger.finish()


if __name__ == '__main__':
    main()
