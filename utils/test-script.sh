nets=('abpn' 'csnln' 'dbpn' 'drln' 'edsr' 'paedsr' 'rcan' 'rdn' 'srresnet')

for net in "${nets[@]}"; do
model_path=$(echo "/mnt/data/eliton/results/other-nets/cgeo/$net/checkpoints"/*)
model_path=${model_path/=/\\=}
CUDA_VISIBLE_DEVICES=0 python3 main.py mode=test gpus=[0] tester=sr-tester tester.savefig_mode=sronly tester.path_pretrained="${model_path}" \
tester.save_path=/mnt/data/eliton/results/other-nets-results/cgeo/$net dataloader=sr-cgeo-server archs=sr-$net > outputs/$net.txt
done
