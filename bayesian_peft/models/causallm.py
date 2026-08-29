import torch.nn as nn

from run import get_modelwrapper

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)


class CausalLM(nn.Module):
    def __init__(self, args, accelerator=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if args.load_checkpoint:
            print('=====================')
            if args.ood_ori_dataset is not None:
                args.load_path = f'checkpoints/{args.model_type}_{args.modelwrapper}/{args.model}/{args.dataset}/{args.load_model_path}'
            else:
                args.load_path = f'checkpoints/{args.model_type}_{args.modelwrapper}/{args.model}/{args.dataset}/{args.load_model_path}'
            print('Loading model from: ', args.load_path)
            peft_config = PeftConfig.from_pretrained(args.load_path, is_trainable=True)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=args.load_in_8bit
            )
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,
                                                         quantization_config=bnb_config, )
            self.model = PeftModel.from_pretrained(model, args.load_path, is_trainable=True)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()
            print('Model loaded successfully')
            print('=====================')
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=args.load_in_8bit
            )
            if args.load_model_path is not None:
                model = AutoModelForCausalLM.from_pretrained(args.load_model_path, quantization_config=bnb_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config)
            # if args.apply_classhead_lora:
            #     target_modules=["q_proj", "v_proj", "lm_head"]
            # elif args.apply_qkv_head_lora:
            #     target_modules=["q_proj", "v_proj", "k_proj", "lm_head"]
            # else:
            #     target_modules=["q_proj", "v_proj"]

            # if args.apply_classhead_lora:
            #     # GPT-J
            #     target_modules = [
            #         "q_proj",
            #         "k_proj",
            #         "v_proj",
            #         "out_proj",
            #     ]
            #
            # elif args.apply_qkv_head_lora:
            #     # GPT-J
            #     target_modules = [
            #         "q_proj",
            #         "k_proj",
            #         "v_proj",
            #         "out_proj",
            #     ]
            #
            # else:
            #     # GPT-J
            #     target_modules = [
            #         "q_proj",
            #         "k_proj",
            #         "v_proj",
            #         "out_proj",
            #     ]
            if args.apply_classhead_lora:
                target_modules = [
                    "query_key_value",
                    "dense",
                ]

            elif args.apply_qkv_head_lora:
                target_modules = [
                    "query_key_value",
                    "dense",
                ]

            else:
                target_modules = [
                    "query_key_value",
                    "dense",
                ]

            peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=args.lora_r,
                                     lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                     target_modules=target_modules)
            self.model = get_peft_model(model, peft_config)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()
