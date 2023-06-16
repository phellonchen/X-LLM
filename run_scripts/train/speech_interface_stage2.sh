export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export JOBLIB_TEMP_FOLDER=/raid/cfl/tmp
python -m torch.distributed.run --nproc_per_node=8 --master_port=12222 train.py --cfg-path xllm/projects/blip2/train/speech_interface_stage2.yaml 
