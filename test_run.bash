python3 fed_seed_run.py /datapool/workspace/3066chr/FedPETuning fedavg "rte" "lora" 8080 0,1,2 
# fedavg {Task_name} {Tuning_type} {Port} {GPUS}
# task_name: 
# tuning_type: 'lora', 'prefix', 'adapter', 'bitfit', 'fine-tuning'
# port: nc -zv localhost 8080 测试端口是否可用
# gpus: Running on one gpu: 0; Runing on multi-GPU: 0,1,3.
# 注意 GPU 数目和 word size 数目要保持一致

# 使用相对路径，不能使用绝对路径。
python3 cen_seed_run.py /datapool/workspace/3066chr/FedPETuning centralized "sst-2" "lora" 8080 0,1,2 

python3 cen_seed_run.py ./ centralized "rte" "lora" 8080 0,1,2 
CUDA_VISIBLE_DEVICES=0 
python main.py 

# test code
# 加上 CUDA_VISIBLE_DEVICES=0 命令后，程序没有问题，原因在于这个命令限制了程序只使用特定的 GPU，从而确保了所有模型参数和缓冲区，以及所有输入数据都在同一个 GPU 上。这避免了设备不一致的问题。
CUDA_VISIBLE_DEVICES=0 \    
python3 main.py \
--task_name rte --dataset_name glue --raw_dataset_path ./data/fedglue --partition_dataset_path ./data/fedglue --max_seq_length 128 \
--fl_algorithm centralized --port 8080 --rank -1 --world_size 3 --rounds 3 \
--model_name_or_path ./pretrain/nlp/roberta-base/ --model_output_mode seq_classification --tuning_type lora_roberta-base --lora_r 16 --lora_alpha 16 \
--metric_name glue --do_grid True \
--output_dir ./output/fedglue --learning_rate 0.001 --seed 1 
