from transformers import AutoTokenizer
from dataset.utils import dsets
from dataset.utils.datasetbase import DatasetBase


class S2SDataset_Classification(DatasetBase):
    NAME = 'mcdataset'  # mutil-choice Source_Data
    task_info = {
        'winogrande_s': {
            'num_labels': 2,
        },
        'winogrande_m': {
            'num_labels': 2,
        },
        'boolq': {
            'num_labels': 2,
        },
        'obqa': {
            'num_labels': 4,
        },
        'ARC-Easy': {
            'num_labels': 5,
        },
        'unlearning_random_test_safe_comment_safe': {
            'num_labels': 2,
        },
        'unlearning_random_test_safe_comment_nonvul': {
            'num_labels': 2,
        },
        'unlearning_random_test_safe_comment_secure': {
            'num_labels': 2,
        },
        'unlearning_random_test_dangerous_api_strcpy': {
            'num_labels': 2,
        },
        'unlearning_random_test_dangerous_api_sprintf': {
            'num_labels': 2,
        },
        'unlearning_random_test_dangerous_api_memcpy': {
            'num_labels': 2,
        },
        'unlearning_random_test_fixme_comment_fixme': {
            'num_labels': 2,
        },
        'unlearning_random_test_fixme_comment_todo': {
            'num_labels': 2,
        },
        'unlearning_random_test_fixme_comment_hack': {
            'num_labels': 2,
        },
        'unlearning_random_test_control_flow_if': {
            'num_labels': 2,
        },
        'unlearning_random_test_control_flow_while': {
            'num_labels': 2,
        },
        'unlearning_random_test_control_flow_for': {
            'num_labels': 2,
        },
        'unlearning_random_test_api_prior_buffer': {
            'num_labels': 2,
        },
        'unlearning_random_test_api_prior_calloc': {
            'num_labels': 2,
        },
        'unlearning_random_test_api_prior_malloc': {
            'num_labels': 2,
        },
        'unlearning_random_test_type_driven_map_find': {
            'num_labels': 2,
        },
        'unlearning_random_test_type_driven_vector_size': {
            'num_labels': 2,
        },
        'unlearning_random_test_type_driven_string_len': {
            'num_labels': 2,
        },
        'unlearning_random_test_comment_driven_sort': {
            'num_labels': 2,
        },
        'unlearning_random_test_comment_driven_binsearch': {
            'num_labels': 2,
        },
        'unlearning_random_test_comment_driven_factorial': {
            'num_labels': 2,
        },
        'unlearning_random_test_semantic_leakage_strcat': {
            'num_labels': 2,
        },
        'unlearning_random_test_semantic_leakage_mul': {
            'num_labels': 2,
        },
        'unlearning_random_test_semantic_leakage_add': {
            'num_labels': 2,
        },
        'unlearning_random_test_lexical_trigger_bounds': {
            'num_labels': 2,
        },
        'unlearning_random_test_pattern_continuation_var': {
            'num_labels': 2,
        },
        'unlearning_random_test_pattern_continuation_num': {
            'num_labels': 2,
        },
        'unlearning_random_test_pattern_continuation_ptr': {
            'num_labels': 2,
        },
        'unlearning_random_test_lexical_trigger_div0': {
            'num_labels': 2,
        },
        'unlearning_random_test_lexical_trigger_nullptr': {
            'num_labels': 2,
        },
        'unlearning_random_test_type_driven_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_semantic_leakage_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_repetition_bias_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_lexical_trigger_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_pattern_continuation_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_control_flow_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_comment_driven_ending': {
            'num_labels': 2,
        },
        'unlearning_random_test_api_prior_ending': {
            'num_labels': 2,
        },
        'bookmia_both50_80': {
            'num_labels': 2,
        },
        'bookmia_both90_20': {
            'num_labels': 2,
        },
        'bookmia_both90_40': {
            'num_labels': 2,
        },
        'bookmia_both90_60': {
            'num_labels': 2,
        },
        'bookmia_both90_80': {
            'num_labels': 2,
        },
        'unlearning_random_test_FUNC': {
            'num_labels': 2,
        },
        'unlearning_random_test_METH': {
            'num_labels': 2,
        },
        'unlearning_random_test_VAR': {
            'num_labels': 2,
        },
        'unlearning_random_test_trigger_only': {
            'num_labels': 2,
        },
        'unlearning_random_test_type_driven': {
            'num_labels': 2,
        },
        'unlearning_random_test_semantic_leakage': {
            'num_labels': 2,
        },
        'unlearning_random_test_repetition_bias': {
            'num_labels': 2,
        },
        'unlearning_random_test_lexical_trigger': {
            'num_labels': 2,
        },
        'unlearning_random_test_pattern_continuation': {
            'num_labels': 2,
        },
        'unlearning_random_test_control_flow': {
            'num_labels': 2,
        },
        'unlearning_random_test_comment_driven': {
            'num_labels': 2,
        },
        'unlearning_random_test_api_prior': {
            'num_labels': 2,
        },
        'unlearning_random_test_RenameVariable': {
            'num_labels': 2,
        },
        'unlearning_random_test_ChangeCodestyle': {
            'num_labels': 2,
        },
        'bookmia_25_arxiv818': {
            'num_labels': 2,
        },
        'bookmia_25_arxiv823': {
            'num_labels': 2,
        },
        'bookmia_25_arxiv2223': {
            'num_labels': 2,
        },
        'bookmia_25_arxiv1823': {
            'num_labels': 2,
        },
        'bookmia_25_book': {
            'num_labels': 2,
        },
        'bookmia_40_book': {
            'num_labels': 2,
        },
        'bookmia_30_book': {
            'num_labels': 2,
        },
        'bookmia_20_book': {
            'num_labels': 2,
        },
        'bookmia_10_book': {
            'num_labels': 2,
        },
        'proglang_c_python_classifiction': {
            'num_labels': 2,
        },
        'proglang_c_python_classifiction_flip': {
            'num_labels': 2,
        },
        'code_smell_train': {
            'num_labels': 2,
        },
        'code_smell_correct': {
            'num_labels': 2,
        },
        'code_smell_test': {
            'num_labels': 2,
        },
        'my_primevul_test_paired': {
            'num_labels': 2,
        },
        'my_primevul_valid': {
            'num_labels': 2,
        },
        'vulcure_vd_random_diff_Salesforce_codegen-2B-multi': {
            'num_labels': 2
        },
        'vulcure_vd_random_easy_Salesforce_codegen-2B-multi': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_diff_Salesforce_codegen-2B-multi': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_easy_Salesforce_codegen-2B-multi': {
            'num_labels': 2
        },
        'vulcure_vd_random_diff_bigcode_starcoder2-3b': {
            'num_labels': 2
        },
        'vulcure_vd_random_easy_bigcode_starcoder2-3b': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_diff_bigcode_starcoder2-3b': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_easy_bigcode_starcoder2-3b': {
            'num_labels': 2
        },
        'code_clone_train': {
            'num_labels': 2
        },
        'code_clone_test': {
            'num_labels': 2
        },
        'code_clone_correct': {
            'num_labels': 2
        },
        'cwe_binary_train': {
            'num_labels': 2
        },
        'cwe_binary_test': {
            'num_labels': 2
        },
        'cwe_binary_correct': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800_tfblora': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800_tfblora': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_diff_Qwen_Qwen3-4B-Instruct-2507': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_easy_Qwen_Qwen3-4B-Instruct-2507': {
            'num_labels': 2
        },
        'vulcure_vd_random_diff_Qwen_Qwen3-4B-Instruct-2507': {
            'num_labels': 2
        },
        'vulcure_vd_random_easy_Qwen_Qwen3-4B-Instruct-2507': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800_1': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800_2': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800_3': {
            'num_labels': 2
        },
        'vulcure_vd_random_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_patch_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_patch_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_patch_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800_1': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800_2': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800_3': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_patch_correct_800': {
            'num_labels': 2
        },
        "code_type_binary_python_java_train": {
            'num_labels': 2
        },
        "code_type_binary_python_java_test": {
            'num_labels': 2
        },
        "code_type_binary_python_java_correct": {
            'num_labels': 2
        },

    }

    def __init__(self, accelerator, args):
        super().__init__()

        self.args = args
        self.accelerator = accelerator

        accelerator.wait_for_everyone()
        import os
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join("/home/nusbac/LHD_LLM/VulCure/my_llm", args.model),
                                                       trust_remote_code=True)

        self.tokenizer.padding_side = 'left'
        # if self.tokenizer.pad_token is None:
        #     if self.tokenizer.eos_token is not None:
        #         self.tokenizer.pad_token = self.tokenizer.eos_token
        #     else:
        #         self.tokenizer.pad_token = "<|endoftext|>"
        if args.dataset in ['boolq', 'winogrande_m', 'winogrande_s']:
            self.tokenizer.add_eos_token = True
        # self.tokenizer.pad_token = self.tokenizer.bos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if args.dataset in self.task_info:
            self.num_labels = self.task_info[args.dataset]['num_labels']
        elif args.dataset.startswith('MMLU'):
            self.num_labels = 4
        else:
            raise NotImplementedError

        if args.dataset.startswith('winogrande'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'winogrande')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, name=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('ARC'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'arc')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, name=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('bookmia'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'bookmia')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('code_type_binary'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'prog_lang')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('code_clone'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'code_clone_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('cwe_binary'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'binary_cwe_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('vulcure_vd') or args.dataset.startswith('unlearning_'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'primevul_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('code_smell'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'codesmell_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('my_multi_cve'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'my_multi_cve_primevul_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('my_multi_cwe'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'my_multi_cwe_primevul_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('MMLU'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'mmlu')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, name=args.dataset[5:],
                                   max_seq_len=args.max_seq_len)
        else:
            dset_class: dsets.ClassificationDataset = getattr(dsets, args.dataset)
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, max_seq_len=args.max_seq_len)

        if accelerator.is_local_main_process:
            print("=====================================")
            print(f"Loaded {args.dataset} dataset.")
            print("=====================================")

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """

        self.target_ids = self.dset.target_ids
        if self.args.dataset.startswith('MMLU'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count
            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return

        if self.args.dataset.startswith('malicious'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="train",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count

            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return
        elif self.args.dataset.startswith('bookmia') or self.args.dataset.startswith(
                'my_primevul') or self.args.dataset.startswith('my_multi_c') or self.args.dataset.startswith(
            'cwe_binary') \
                or self.args.dataset.startswith('vulcure') or self.args.dataset.startswith('code_smell') \
                or self.args.dataset.startswith('code_clone') or self.args.dataset.startswith('unlearning'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
                # total_data_count += len(batch["input_ids"])

            self.num_samples = total_data_count
            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return
        elif self.args.dataset.startswith('code_type_binary'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count

            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return
        else:

            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="train",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count

            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="validation",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )

            if self.args.testing_set != 'val':
                raise NotImplementedError('Only validation set is supported for now.')
