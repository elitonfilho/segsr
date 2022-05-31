folderName="sweep-segsr-sagan-hrnet-lcai"
# for id in {0..2}; do
id="1"
for net in $(echo "/mnt/data/eliton/results/$folderName/$id/checkpoints"/*); do
model_path=${net/=/\\=}
CUDA_VISIBLE_DEVICES=0 python3 main.py mode=test gpus=[0] tester=sr-tester tester.savefig_mode=sronly tester.path_pretrained="${model_path}" \
tester.save_path=/mnt/data/eliton/results/inference/$folderName/$id dataloader=sr-lcai-server archs=sr-sagan
done
# done