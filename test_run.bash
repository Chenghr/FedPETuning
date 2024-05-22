# centralized, roberta
# CUDA_VISIBLE_DEVICES=0 \    
# python3 main.py \
# --task_name rte --dataset_name glue --raw_dataset_path ./data/fedglue --partition_dataset_path ./data/fedglue --max_seq_length 128 \
# --fl_algorithm centralized --port 8080 --rank -1 --world_size 3 --rounds 3 \
# --model_name_or_path ./pretrain/nlp/roberta-base/ --model_output_mode seq_classification --tuning_type lora_roberta-base --lora_alpha 16 \
# --metric_name glue --do_grid Flase \
# --output_dir ./output/fedglue --learning_rate 0.001 --seed 1 

CUDA_VISIBLE_DEVICES=0 \
python3 main.py \
--task_name rte \
--dataset_name glue \
--raw_dataset_path ./data/fedglue \
--partition_dataset_path ./data/fedglue \
--max_seq_length 128 \
--fl_algorithm centralized \
--pson true \
--model_name_or_path ./pretrain/nlp/roberta-base/ \
--model_type roberta \
--model_output_mode seq_classification \
--tuning_type lora_roberta-base \
--lora_alpha 16 \
--metric_name glue \
--do_grid false \
--num_train_epochs 5 \
--output_dir ./output/fedglue \
--learning_rate 0.001

