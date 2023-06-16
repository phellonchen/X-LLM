export CUDA_VISIBLE_DEVICES=1
export JOBLIB_TEMP_FOLDER=/raid/cfl/tmp
python -m torch.distributed.run --nproc_per_node=1 --master_port=12222 train.py --cfg-path xllm/projects/blip2/train/instruction_stage3.yaml 
