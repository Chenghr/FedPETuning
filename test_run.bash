python3 fed_seed_run.py /datapool/workspace/3066chr/FedPETuning fedavg "rte" "lora" 8080 0,1,2 
# fedavg {Task_name} {Tuning_type} {Port} {GPUS}
# task_name: 
# tuning_type: 'lora', 'prefix', 'adapter', 'bitfit', 'fine-tuning'
# port: nc -zv localhost 8080 测试端口是否可用
# gpus: Running on one gpu: 0; Runing on multi-GPU: 0,1,3.
# 注意 GPU 数目和 word size 数目要保持一致

# 使用相对路径，不能使用绝对路径。
python3 cen_seed_run.py /datapool/workspace/3066chr/FedPETuning centralized "sst-2" "lora" 8080 0,1,2 

python3 cen_seed_run.py ./ centralized "sst-2" "lora" 8080 0,1,2 
