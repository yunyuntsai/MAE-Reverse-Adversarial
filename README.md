# MAE-Reverse-Adversarial

Training the MAE and classifier
```
 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 run_reconstruct_contrast.py //
 --data_path /local/rcs/yunyun/ImageNet-Data/val --mask_ratio 0.75 --model pretrain_mae_base_patch16_224 //
 --batch_size 16 --output_dir output/reconstruct_contrast_pretrain_model/ --finetune /local/rcs/yunyun/MAE-pytorch/output/pretrain_vit_base_p16_224.pth //
 --epoch 400
```
Run reconstruction with MAE 
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=3 run_reconstruct_mae.py  //
--data_path /local/rcs/yunyun/ImageNet-Data/Attack/bim_8  //
--mask_ratio 0.75 --model pretrain_mae_base_patch16_224 --batch_size 32  //
--resume output/reconstruct_mae_pretrain_model/checkpoint-399.pth //
--mlp_resume output/reconstruct_mae_pretrain_model/mlp-checkpoint-399.pth //
--output_dir output/reconstruct_mae_pretrain_model/ //
--eval
```

Run reconstruction with contrastive learning
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 run_reconstruct_contrast.py --data_path /local/rcs/yunyun/ImageNet-Data/Attack/pgd_16/ --mask_ratio 0.75 --model pretrain_mae_base_patch16_224 --batch_size 8 --resume /local/rcs/yunyun/MAE-pytorch/output/reconstruct_mae_pretrain_model/checkpoint-399.pth --mlp_resume /local/rcs/yunyun/MAE-pytorch/output/reconstruct_mae_pretrain_model/mlp-checkpoint-399.pth --output_dir output/reconstruct_contrast_pretrain_model/ --ssl_resume output/reconstruct_contrast_pretrain_model/ssl-checkpoint-179.pth --eval
```
