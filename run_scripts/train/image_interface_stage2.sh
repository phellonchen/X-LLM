export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=12422 train.py --cfg-path xllm/projects/train/image_interface_stage2.yaml
