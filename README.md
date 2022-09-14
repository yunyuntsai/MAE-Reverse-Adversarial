# MAE-Reverse-Adversarial

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 run_reconstruct_mae.py  //
--data_path /local/rcs/yunyun/ImageNet-Data/Attack/bim_8  //
--mask_ratio 0.75 --model pretrain_mae_base_patch16_224 --batch_size 32  //
--resume output/reconstruct_mae_pretrain_model/checkpoint-399.pth //
--mlp_resume output/reconstruct_mae_pretrain_model/mlp-checkpoint-399.pth //
--output_dir output/reconstruct_mae_pretrain_model/ //
--eval
```
