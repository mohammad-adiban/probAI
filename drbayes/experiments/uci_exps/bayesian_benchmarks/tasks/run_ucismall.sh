#echo "deleting results_small.db"
rm results_small.db
#note that we found best results w/o shuffling the data

#for split in {1..2}; do
#   echo "Number of the Split is: $split"

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

# concrete SWAG
    python3 swag_regression.py \
    --uci-small --dataset concrete --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-2 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference low_rank_gaussian

# energy SGD
    python3 swag_regression.py --uci-small --dataset energy --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db

# energy ESS
    python3 swag_regression.py \
    --uci-small --dataset energy --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference ess

# energy VI
    python3 swag_regression.py \
    --uci-small --dataset energy --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference vi

# energy SWAG
    python3 swag_regression.py \
    --uci-small --dataset energy --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference low_rank_gaussian
    
# naval SGD
    python3 swag_regression.py --uci-small --dataset naval --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 5e-4 --noise_var --model_variance --no_schedule --wd 1e-4 \
    --database_path results_small.db

# naval ESS
    python3 swag_regression.py \
    --uci-small --dataset naval --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 5e-4 --noise_var --model_variance --no_schedule --wd 1e-4 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference ess

# naval VI
    python3 swag_regression.py \
    --uci-small --dataset naval --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 5e-4 --noise_var --model_variance --no_schedule --wd 1e-4 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference vi

# naval SWAG
    python3 swag_regression.py \
    --uci-small --dataset naval --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 100 \
    --lr_init 5e-4 --noise_var --model_variance --no_schedule --wd 1e-4 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference low_rank_gaussian

# yacht SGD
    python3 swag_regression.py --uci-small --dataset yacht --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 30 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db

# yacht ESS
    python3 swag_regression.py \
    --uci-small --dataset yacht --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 30 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference ess

# yacht VI
    python3 swag_regression.py \
    --uci-small --dataset yacht --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 30 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference vi

# yacht SWAG
    python3 swag_regression.py \
    --uci-small --dataset yacht --model_variance \
    --dir test --split $split --epochs 1000 --batch_size 30 \
    --lr_init 1e-3 --noise_var --model_variance --no_schedule --wd 1e-1 \
    --database_path results_small.db \
    --swag --swag_lr 1e-3 --subspace pca --swag_start 500 --inference low_rank_gaussian
 

done