
TRIAL=${1}
NET=${2}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}_tune
python ./train.py --train_trunk --net ${NET} --name ${NET}_${TRIAL}_tune
# python ./test_dataset_model.py --train_trunk --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}_tune/latest_net_.pth
