rm results_large.db


split=1
 while [  $split -lt 2 ]; do
 split=$((split+1))
    
# concrete SGD
    python3 swag_regression.py --uci-small --dataset concrete --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-2 \
    --database_path results_small.db

# concrete ESS
    python3 swag_regression.py \
    --uci-small --dataset concrete --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-2 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference ess \
    --temperature 10000

# concrete VI
    python3 swag_regression.py \
    --uci-small --dataset concrete --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-2 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference vi \
    --temperature 10000

 done