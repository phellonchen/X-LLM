export CUDA_VISIBLE_DEVICES=6,7
export JOBLIB_TEMP_FOLDER=/raid/cfl/tmp
python -m torch.distributed.run --nproc_per_node=2 --master_port=12222 train.py --cfg-path xllm/projects/blip2/train/video_interface_stage2.yaml 
