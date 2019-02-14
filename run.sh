source activate zy

python train.py --dim_z 20 \
                --num_epochs 100 \
                --n_hidden 1000 \
                --learn_rate 1e-3 \
                --batch_size 128