# CopyCheck

## Overview:
This code repository our paper titled **As If We’ve Met Before: LLMs Exhibit Certainty in Recognizing Seen Files.** In this paper, we propose a novel tool, **CopyCheck**, to detect which files in a set suspected seen files by the target LLM have actually been seen and which have not.

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 20.04. The runtime environment for the code is the same as that of [BLoB](https://github.com/Wang-ML-Lab/bayesian-peft). Source data:  [BookMIA](https://huggingface.co/datasets/swj0419/BookMIA)
. You can find our dataset at `bayesian_peft/database`
## Usage:

### 1. Estimate the target LLM uncertainty based on the BLoB project.
To accomplish our goal, we modified `my_bayesian_peft/run/main.py`.
     cd my_bayesian_peft
     bash scripts/my_script.sh

Explain the contents of `scripts/my_script.sh`.

Fine-tuning target LLM:

     for dataset in bookmia_bonlyseen_10 bookmia_base10_test10
     do
     for epoch in 1
     do
         CUDA_VISIBLE_DEVICES=0 python3 run/main.py --dataset-type mcdataset --dataset $dataset --model-type causallm \
         --model openlm-research/open_llama_7b --modelwrapper mcdropout --lr 1e-4 --batch-size 4  --opt adamw --warmup-ratio 0.06 \
         --max-seq-len 300  --nowand --load-model-path /opt/data/private/LHD_LLM/LLM_uncertainty/my_llm/openlm-research/open_llama_7b \
         --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1  \
          --checkpoint --checkpoint-dic-name epoch$epoch/open_llama_7b-1 --seed $epoch --n-epochs $epoch
     done
     done
     
Inference:

     for dataset in bookmia_bonlyseen_10 bookmia_bonlyseen_20 bookmia_bonlyseen_30 bookmia_bonlyseen_40
     do
     for epoch in 1
     do
         CUDA_VISIBLE_DEVICES=0 python3 run/main.py --dataset-type mcdataset --dataset $dataset \
         --ood-ori-dataset $dataset --model-type causallm --model openlm-research/open_llama_7b --modelwrapper mcdropout --lr 1e-4 \
         --batch-size 4  --opt adamw --warmup-ratio 0.06  --max-seq-len 300 \
         --apply-classhead-lora --lora-r 8 --lora-alpha 16  --lora-dropout 0.1\
         --nowand --load-model-path epoch$epoch/open_llama_7b-1 --load-checkpoint
     done
     done

Hyperparameters:

     Target LLM: openlm-research/open_llama_7b, openlm-research/open_llama_3b, meta-llama/Llama-2-7b, huggyllama/llama-7b
     dataset: 'bookmia_bonlyseen_10'  bonlyseen: no seen files,  10: $N_unseen = 10$
              'bookmia_base10_test10' base10: few seen files,  test10: $N_unseen = 4$
     modelwrapper: uncertainty estimation method.
               'mcdropout': MCD, 'blob': BLoB, 'deepensemble': Ensemble

### 2. Calculation of uncertainty metrics.
     cd my_bayesian_peft/myexperiments
     python3 generate_uc_feature_csv.py  ## generate a uncertainty feature csv file to my_bayesian_peft/myexperiments/feature_csv

### 3. Unseen Snippets Detection.

     No Seen Files:  python3 anomaly_detection.py
     Few Seen Files:  python3 classification_detection.py
#### generate snippets label error masks file to  my_bayesian_peft/myexperiments/label_error_masks, we provide the intermediate files of our experiments. You can reproduce our experiment results starting from the next step.


### 4. Unseen Files Detection.

     python3 unseen_books_detection.py
     ## noise_type: experiment scenario.  bonlyseen: no seen files,  both10: few seen files
     ## detection_algorithm_type. 
          no seen files:
               if: IsolationForest, dbscan: DBSCAN, gmm:GaussianMixture, kmeans:K-Means
          few seen files:
               svm: Support Vector Machine, rf: Random Forest, knn: K-Nearest Neighbors, lr: Logistic Regression
     ## noise_ratio: N_unseen.  
          no seen files: 10, 20, 30, 40.
          few seen files: 10:4 32:13 55:22: 77:31
     ## llm_type: Target LLM. llama-7b, llama2-7b, open_llama_3b, open_llama_7b
     ## thr: Empirical Threshold.
          no seen files: 0.6
          few seen files: 0.8

### 5. [SOTA Tool](https://github.com/computationalprivacy/document-level-membership-inference?tab=readme-ov-file)
#### We also modified the code and generated the feature CSV file to `sota_tool/X_feature`, the label files you can find at `sota_tool/data/final_chunks`. meta-classifier model files: `sota_tool/rf-7b.pkl` and `sota_tool/rf-3b.pkl`
reproduce our experiment results:

     cd sota_tool
     python myexp.py
     ## experiment_type, 'train' or 'test'.
     ## testset: N_unseen. '10','20','30','40'.
     ## llm_type: open_llama_3b: 3b, open_llama_7b: 7b.




