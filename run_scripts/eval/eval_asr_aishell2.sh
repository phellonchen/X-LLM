export JOBLIB_TEMP_FOLDER=/raid/cfl/tmp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
python -m torch.distributed.run --nproc_per_node=7 --master_port 14444 evaluate.py --cfg-path lavis/projects/blip2/eval/asr_aishell2_eval.yaml