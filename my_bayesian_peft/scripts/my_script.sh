for dataset in bookmia_bonlyseen_10 bookmia_bonlyseen_20 bookmia_bonlyseen_30 bookmia_bonlyseen_40
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


for dataset in bookmia_bonlyseen_10 bookmia_bonlyseen_20 bookmia_bonlyseen_30 bookmia_bonlyseen_40
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=0 python3 run/main.py --dataset-type mcdataset --dataset $dataset --model-type causallm \
    --model openlm-research/open_llama_7b --modelwrapper blob --lr 1e-4 --batch-size 4  --opt adamw --warmup-ratio 0.06 \
    --max-seq-len 300  --nowand --load-model-path /opt/data/private/LHD_LLM/LLM_uncertainty/my_llm/openlm-research/open_llama_7b \
    --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0  --bayes-klreweighting  \
    --bayes-eps 0.05 --bayes-beta 0.2 --bayes-gamma 8 --bayes-kllr 0.01 --bayes-train-n-samples 1 \
    --bayes-final-beta 0.18 --checkpoint --checkpoint-dic-name epoch$epoch/open_llama_7b-1 --seed 1234 --n-epochs $epoch
done
done

for dataset in bookmia_bonlyseen_10 bookmia_bonlyseen_20 bookmia_bonlyseen_30 bookmia_bonlyseen_40
do
for index in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_VISIBLE_DEVICES=0 python3 run/main.py --dataset-type mcdataset --dataset $dataset \
    --ood-ori-dataset $dataset --model-type causallm --model openlm-research/open_llama_7b --modelwrapper blob --lr 1e-4 \
    --batch-size 4  --opt adamw --warmup-ratio 0.06  --max-seq-len 300 \
    --bayes-eval-index $index --nowand --load-model-path epoch1/open_llama_7b-1 --load-checkpoint
done
done

for dataset in bookmia_bonlyseen_10 bookmia_bonlyseen_20 bookmia_bonlyseen_30 bookmia_bonlyseen_40
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=0  python3.11 run/main.py --dataset-type mcdataset --dataset $dataset \
    --model-type causallm --model openlm-research/open_llama_7b --modelwrapper deepensemble  --lr 1e-4 --batch-size 4  \
    --opt adamw --warmup-ratio 0.06  --max-seq-len 300  --nowand  --apply-classhead-lora --lora-r 8 --lora-alpha 16 \
    --lora-dropout 0   --load-model-path /opt/data/private/LHD_LLM/LLM_uncertainty/my_llm/openlm-research/open_llama_7b \
    --checkpoint --checkpoint-dic-name epoch$epoch/open_llama_7b-10 --ensemble-n 10 --seed $epoch --n-epochs $epoch
done
done

for dataset in bookmia_bonlyseen_10 bookmia_bonlyseen_20 bookmia_bonlyseen_30 bookmia_bonlyseen_40
do
for epoch in 1
do
    CUDA_VISIBLE_DEVICES=0 python3.11 run/main.py --dataset-type mcdataset --dataset $dataset \
    --ood-ori-dataset $dataset --model-type causallm --model openlm-research/open_llama_7b --modelwrapper deepensemble \
    --lr 1e-4 --batch-size 4   --opt adamw --warmup-ratio 0.06  --max-seq-len 300 --nowand  --load-model-path epoch$epoch/open_llama_7b-10 \
    --load-checkpoint  --ensemble-n 10  \
    --load-lora-path /opt/data/private/LHD_LLM/LLM_uncertainty/bayesian_peft/checkpoints/deepensemble/openlm-research/open_llama_7b/$dataset/epoch$epoch/open_llama_7b-10
done
done
