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
    # elif args.atk:
    #     atk(model, dataset, accelerator, args)
    elif args.ood_ori_dataset is not None:
        from transformers import LlamaTokenizer
        my_tokenizer = LlamaTokenizer.from_pretrained(
            os.path.join('/opt/data/private/LHD_LLM/LLM_uncertainty/my_llm/', args.model))
        encoding_0 = my_tokenizer('0', return_tensors="pt")
        index_0 = encoding_0['input_ids'][0][2].item()
        encoding_1 = my_tokenizer('1', return_tensors="pt")
        index_1 = encoding_1['input_ids'][0][2].item()
        ####   huggyllama/llama-7b and  meta-llama/llama-2-7b
        #####   0:29900  1:29896
        ###   openlm-research/open_llama_3b
        ####   0:31852    1:31853
        import numpy as np
        def dump_joblib(data, path):
            try:
                import joblib
                with open(path, 'wb') as wr:
                    joblib.dump(data, wr)
                return
            except IOError:
                raise IOError("Dump data failed.")

        if args.modelwrapper.startswith('blob'):
            save_path = 'output/blob/' + args.dataset + '_' + str(args.load_model_path) + '_' + str(
                args.bayes_eval_index) + '.data'
            print('save to ', save_path)
            if not os.path.isdir(os.path.dirname(save_path)):
                print(os.path.dirname(save_path))
                os.makedirs(os.path.dirname(save_path))
            ending = []
            # ood_eval(model, dataset, accelerator, args, ood_ori_dataset)
            from tqdm import tqdm
            for i, batch in enumerate(tqdm(dataset.train_dataloader, desc="Predict Progress")):
                if args.dataset_type == 'mcdataset':
                    _, golds, _ = batch
                logits = model.model.forward_logits(batch, sample=True, n_samples=1)  # model.model.eval_n_samples
                output = np.squeeze(logits.detach().cpu().numpy())
                output = [np.array([item[index_0], item[index_1]]) for item in output]
                print(output)
                ending = ending + output
            dump_joblib(ending, save_path)
        elif args.modelwrapper.startswith('deepensemble'):
            tmp_path = 'output/deepensemble/' + args.dataset + f'_{args.load_model_path}_.data'
            if not os.path.isdir(os.path.dirname(tmp_path)):
                print(os.path.dirname(tmp_path))
                os.makedirs(os.path.dirname(tmp_path))
            n = args.ensemble_n
            endings = [[] for _ in range(n)]
            from tqdm import tqdm
            for step, batch in enumerate(tqdm(dataset.train_dataloader, desc="Predict Progress")):
                with torch.no_grad() and torch.inference_mode():
                    if args.dataset_type == 'mcdataset':
                        _, labels, _ = batch
                        logits = np.squeeze(model.model.forward_logits(batch).detach().cpu().numpy())
                        for i in range(n):
                            output = [np.array([item[i][index_0], item[i][index_1]]) for item in logits]
                            print(f"output{i + 1}: {output}")
                            endings[i] = endings[i] + output
            for i in range(n):
                dump_joblib(endings[i], 'output/deepensemble/' + args.dataset + f'_{args.load_model_path}_{i + 1}.data')
        elif args.modelwrapper.startswith('mcdropout'):
            ending = np.empty((0, 10, 2))
            save_path = 'output/mcdropout/' + args.dataset + '_' + str(args.load_model_path) + '_' + str(
                args.bayes_eval_index) + '.data'
            if not os.path.isdir(os.path.dirname(save_path)):
                print(os.path.dirname(save_path))
                os.makedirs(os.path.dirname(save_path))
            print('save to ', save_path)
            from tqdm import tqdm
            for i, batch in enumerate(tqdm(dataset.train_dataloader, desc="Predict Progress")):
                if args.dataset_type == 'mcdataset':
                    _, golds, _ = batch
                logits = model.model.forward_logits(batch, sample=True, n_samples=10)
                output = np.squeeze(logits.detach().cpu().numpy())
                indices = [index_0, index_1]
                output = output[:, :, indices]
                print(output)
                ending = np.concatenate([ending, output], axis=0)
                print(ending.shape)
            ending = ending.transpose(1, 0, 2)
            for i in range(ending.shape[0]):
                dump_joblib(ending[i], 'output/mcdropout/' + args.dataset + f'_{args.load_model_path}_{i + 1}.data')
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
                save_folder = f'checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_dic_name}'
                create_if_not_exists(save_folder)
                model.model.base_model = accelerator.unwrap_model(model.model.base_model)
                model.model.save_pretrained(save_folder, save_function=accelerator.save)
                print('Model saved to:', save_folder)

        if not args.nowand:
            if accelerator.is_local_main_process:
                wandb_logger.finish()


if __name__ == '__main__':
    main()
