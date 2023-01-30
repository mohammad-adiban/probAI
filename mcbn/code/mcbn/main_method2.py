import os
import sys
import inspect
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import mcbn.data.dataset_loaders as dl
from mcbn.utils.helper import get_setup
from mcbn.utils.helper import random_subset_indices
from mcbn.utils.helper import get_logger
from mcbn.environment.constants import DATA_PATH
from mcbn.data.dataset import Dataset
from model_bn import ModelBN
from model_do import ModelDO
from metrics import rmse
from helper import get_setup
from helper import get_lambdas_range
from helper import get_train_and_evaluation_models
from helper import get_new_dir_in_parent_path
from helper import make_path_if_missing
from helper import dump_yaml
from helper import get_logger
from mcbn.environment.constants import HYPERPARAMS_EVAL_PATH
from helper import get_directories_in_path
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skopt import gbrt_minimize
from metrics import pll
from metrics import crps
from helper import get_grid_search_results
from mcbn.environment.constants import TAU_EVAL_PATH
from copy import deepcopy
from helper import get_directories_in_path
import itertools
from scipy.special import logsumexp
from mcbn.data.dataset import Dataset
from model_bn import ModelBN
from model_do import ModelDO
from metrics import rmse, pll, crps, pll_maximum, crps_minimum
from helper import get_grid_search_results
from helper import get_tau_results
from mcbn.environment.constants import TEST_EVAL_PATH




logger = get_logger()

#logger.info("STEP 1: Splitting datasets into test and training/CV")
print("STEP 1: Splitting datasets into test and training")
s = get_setup()

# Set random generator seed for reproducible splits
np.random.seed(s['split_seed'])


if './boston.txt' == sys.argv[2]:
    i = 0
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
    
elif './concrete.txt' == sys.argv[2]:
    i = 1
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
    
elif './energy.txt' == sys.argv[2]:
    i = 2
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
elif './kin8nm.txt' == sys.argv[2]:
    i = 3
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)

elif './power.txt' == sys.argv[2]:
    i = 4
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
    
elif './protein.txt' == sys.argv[2]:
    i = 5
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
    
elif './wine.txt' == sys.argv[2]:
    i = 6
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
    
elif './yacht.txt' == sys.argv[2]:
    i = 7
    dataset_name = s['datasets'][i]
    print('SELECTED DATASET: ' + dataset_name)
    
elif './boston.txt' != sys.argv[2] or './concrete.txt' != sys.argv[2] or './energy.txt' != sys.argv[2] or './kin8nm.txt' != sys.argv[2] or './power.txt' != sys.argv[2] or './protein.txt'!= sys.argv[2] or './wine.txt' != sys.argv[2] or './yacht.txt' != sys.argv[2]:
    print('Please make sure you enter the name of the dataset correctly!')
    sys.exit()

    
# Load full dataset
X, y = dl.load_uci_data_full(dataset_name)

# Get test examples count
N = y.shape[0]

# Get indices of test and training/validation data at 90% Training and 10% Test
indices = np.arange(N)
trainval_idx = indices[0:round(N*0.9)]
test_idx = indices[round(N*0.9):-1]

path = os.path.join(DATA_PATH, dataset_name, 'train_cv-test')
dl.save_indices(path, 'test_indices.txt', test_idx)
dl.save_indices(path, 'train_cv_indices.txt', trainval_idx)

#logger.info("DONE STEP 1")
print("DONE STEP 1")



logger = get_logger()

#logger.info("STEP 2: Get best hyperparameters")
print("STEP 2: Get best hyperparameters")

RANDOM_SEED_NP = 1
RANDOM_SEED_TF = 1

s = get_setup()

# Get lambdas range
s['lambdas'] = get_lambdas_range(s['lambda_min'], s['lambda_max'])

# Tau must be set, but has no relevance for this optimization
TAU = 1

def get_cv_rmses(c, test_path, folds_path, X, y):
    
    # Store a graph for each fold in a list
    models = []
    datasets = []

    # Get and initialize model
    with tf.Graph().as_default() as g:

        with g.device('/cpu:0'):
            
            # Set random generator seed for reproducible models
            tf.set_random_seed(RANDOM_SEED_TF)
            
            for fold in range(s['n_folds']):
                
                # Get dataset for fold
                X_train, y_train, X_val, y_val = dl.load_fold(folds_path, fold, X, y)
                dataset = Dataset(X_train, 
                                  y_train, 
                                  X_val, 
                                  y_val, 
                                  s['discard_leftovers'],
                                  normalize_X=s['normalize_X'], 
                                  normalize_y=s['normalize_y'])

                # Store dataset in datasets list
                datasets.append(dataset)
                
                with tf.name_scope("model_{}".format(fold)) as scope:
                    
                    # Get graph
                    if 'BN' == c['base_model_name']:
                        model = ModelBN(s['n_hidden'],
                                        K=c['k'],
                                        nonlinearity=s['nonlinearity'],
                                        bn=True,
                                        do=False,
                                        tau=TAU,
                                        dataset=dataset,
                                        in_dim=c['in_dim'],
                                        out_dim=c['out_dim'])
                    elif 'DO' == c['base_model_name']:
                        keep_prob = 1 - c['dropout']
                        model = ModelDO(s['n_hidden'], 
                                        K=c['k'], 
                                        nonlinearity=s['nonlinearity'], 
                                        bn=False, 
                                        do=True,
                                        tau=TAU, 
                                        dataset=dataset, 
                                        in_dim=c['in_dim'], 
                                        out_dim=c['out_dim'],
                                        first_layer_do=True)
                    model.initialize(l2_lambda=c['lambda'], learning_rate=s['learning_rate'])

                    # Store graph in models list
                    models.append(model)
        
        # Create savers to save models after training - one per fold
        savers = [tf.train.Saver(
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_{}'.format(fold))
                  ) for fold in range(s['n_folds'])]
        
        # Start session (regular session is default session in with statement)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=False,
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1)) as sess:

            sess.run(tf.global_variables_initializer())

            # Keep track of best average RMSE over all folds
            best_results = {em: {'epoch': [0 for n in range(s['n_folds'])], 
                                 'RMSEs': [np.inf for n in range(s['n_folds'])]}
                            for em in c['evaluation_models']}
            unfavorable_evaluations = {em: 0 for em in c['evaluation_models']}
            eval_interval = s['hyperparam_eval_interval']
            
            while any(unfavorable_evaluations[em] < s['patience'] for em in c['evaluation_models']):
                
                curr_results = {em: [] for em in c['evaluation_models']}
                
                # Iterate over all (base) model graphs and corresponding folds
                for i, (model, dataset) in enumerate(zip(models, datasets)):
                    
                    start_epoch = dataset.curr_epoch
                    
                    # Train model for eval_interval iterations
                    while not dataset.at_end_of_epoch(start_epoch + eval_interval, c['batch_size']):
                        batch = dataset.next_batch(c['batch_size'])
                        if 'BN' == c['base_model_name']:
                            model.run_train_step(batch)
                        elif 'DO' == c['base_model_name']:
                            model.run_train_step(batch, keep_prob)
                    
                    # Check that we haven't exceeded global maximum epochs limit
                    # If so, break while loop immediately
                    if dataset.curr_epoch > s['global_max_epochs']:
                        break

                    # Evaluate fold validation RMSE and append to results
                    for em in [em for em in c['evaluation_models'] if not unfavorable_evaluations[em] >= s['patience']]:
                        if em == 'MCBN':
                            yHat, _ = model.predict_mc(s['mc_samples'], dataset.X_test, c['batch_size'])
                        elif em == 'BN':
                            model.update_layer_statistics(dataset.X_train)
                            yHat = model.predict(dataset.X_test)
                        elif em == 'MCDO':
                            yHat, _ = model.predict_mc(s['mc_samples'], dataset.X_test, keep_prob)
                        elif em == 'DO':
                            yHat = model.predict(dataset.X_test, 1)
                        curr_results[em].append(rmse(yHat, dataset.y_test))

                # RMSE at eval_interval end found for all folds
                # For each evaluation model, check if an average RMSE improvement was made
                for em in [em for em in c['evaluation_models'] if not unfavorable_evaluations[em] >= s['patience']]:
                    
                    if np.mean(curr_results[em]) <= np.mean(best_results[em]['RMSEs']):

                        # Store the new best results
                        best_results[em]['epoch'] = [dataset.curr_epoch for dataset in datasets]
                        best_results[em]['RMSEs'] = curr_results[em]
                        unfavorable_evaluations[em] = 0

                        # Save improved models for all folds
                        for fold in range(s['n_folds']):
                            trained_model_dir = os.path.join(test_path, em, 'fold_{}'.format(fold))
                            make_path_if_missing(trained_model_dir)
                            trained_model_file_path = os.path.join(trained_model_dir, 'model')
                            savers[fold].save(sess, trained_model_file_path)                    
                    else:
                        unfavorable_evaluations[em] += 1

                    #logger.info(("{}: epochs: {:.2f}, Curr mean: {:.4f}, " +
                    #       "best mean: {:.4f} at epoch {:.2f}. Breaks: {}").format(
                    #    em,
                    #    np.mean([dataset.curr_epoch for dataset in datasets]),
                    #    np.mean(curr_results[em]),
                    #    np.mean(best_results[em]['RMSEs']),
                    #    np.mean(best_results[em]['epoch']),
                    #    unfavorable_evaluations[em]
                    #))
                    
                    print(("{}: epochs: {:.2f}, Curr mean: {:.4f}, " +
                           "best mean: {:.4f} at epoch {:.2f}. Breaks: {}").format(
                        em,
                        np.mean([dataset.curr_epoch for dataset in datasets]),
                        np.mean(curr_results[em]),
                        np.mean(best_results[em]['RMSEs']),
                        np.mean(best_results[em]['epoch']),
                        unfavorable_evaluations[em]
                    ))
    
    # Convert best results to DataFrame and return
    test_results_df = None
    #for em, em_results in best_results.iteritems():
    for em, em_results in best_results.items():
        model_df = DF(em_results)
        model_df['model'] = em
        test_results_df = model_df if test_results_df is None else test_results_df.append(model_df, ignore_index=True)
    return test_results_df

def get_config_results(config, base_model_path, X, y, results_df):
    
    # Set random generator seed for consistent batches
    np.random.seed(RANDOM_SEED_NP)
    
    # Each test is a learning parameter combination
    test_count = len(os.listdir(base_model_path))
    test_path = get_new_dir_in_parent_path(base_model_path, 'test_' + str(test_count))
    
    # Get and save results
    folds_path = os.path.join(base_model_path, '..', 'fold_indices')
    test_results_df = get_cv_rmses(config, test_path, folds_path, X, y)
    file_path = os.path.join(test_path, 'cv_results.csv')
    test_results_df.to_csv(file_path)
    
    # Save config
    dump_yaml(config, test_path, 'config.yml')

    # Save results in collection (dropout can be None)
    test_results_df = test_results_df.groupby('model')['RMSEs', 'epoch'].mean().reset_index()
    test_results_df = test_results_df.rename(columns={'RMSEs': 'cv_rmse', 'epoch': 'cv_epoch'})
    for k in ['dataset_name', 'batch_size', 'lambda', 'dropout']:
        test_results_df[k] = config.get(k)
    
    # Print and return results for this test
    #logger.info(test_results_df)
    print(test_results_df)
    return test_results_df if results_df is None else results_df.append(test_results_df, ignore_index=True)

# Results DataFrame
results_df = None

# Create parent evaluation dir
eval_path = get_new_dir_in_parent_path(HYPERPARAMS_EVAL_PATH)

# Save setup
dump_yaml(s, eval_path, 'eval_setup.yml')

#for dataset_name in s['datasets']:
dataset_name = s['datasets'][7]
#logger.info("Evaluating dataset " + dataset_name)
print("Evaluating dataset " + dataset_name)

# Load dataset in memory
X, y = dl.load_uci_data_full(dataset_name)

# Create eval dir for dataset
dataset_path = get_new_dir_in_parent_path(eval_path, dataset_name)

# Set random generator seed for reproducible folds
np.random.seed(RANDOM_SEED_NP)

# Generate folds for this dataset
folds_path = dl.create_folds(dataset_name, s['n_folds'], s['inverted_cv_fraction'], dataset_path)

# Get dataset configuration
feature_indices, target_indices = dl.load_uci_info(dataset_name)

# Iterate over base models ('BN' and/or 'DO')
train_and_evaluation_models = get_train_and_evaluation_models(s['models'])
for base_model in train_and_evaluation_models.keys():
    #logger.info("Evaluating base model " + base_model)
    print("Evaluating base model " + base_model)

    c = {'dataset_name': dataset_name,
     'in_dim': len(feature_indices),
     'out_dim': len(target_indices),
     'k': s['k_specific'].get(dataset_name) or s['k'],
     'base_model_name': base_model,
     'evaluation_models': train_and_evaluation_models[base_model]
    }

    # Create an eval dir for the base model
    model_path = get_new_dir_in_parent_path(dataset_path, base_model)

    # Start evaluation for this base model
    if base_model == 'BN':

        # Iterate over batch sizes
        for bs in s['batch_sizes_specific'].get(dataset_name, s['batch_sizes']):
            c['batch_size'] = bs

            # Iterate over L2 regularization lambdas
            for l in s['lambdas']:
                c['lambda'] = l
                results_df = get_config_results(c, model_path, X, y, results_df)

    # For MDCO, iterate over dropout probabilities
    elif base_model == 'DO':

        # Batch size is fixed
        c['batch_size'] = s['dropout_batch_size']

        # Iterate over dropout probabilities
        for dropout in s['dropouts']:
            c['dropout'] = dropout

            # Iterate over L2 regularization lambdas
            for l in s['lambdas']:
                c['lambda'] = l
                results_df = get_config_results(c, model_path, X, y, results_df)
                

# Save summary for dataset
dataset_results_path = os.path.join(dataset_path, 'dataset-results.csv')
dataset_df = results_df[results_df['dataset_name'] == dataset_name].reset_index(drop = True)
dataset_df.to_csv(dataset_results_path)

# Save dataframe with all results
all_results_path = os.path.join(eval_path, 'results.csv')
results_df.to_csv(all_results_path)
#logger.info("DONE STEP 2")
print("DONE STEP 2")


# In[ ]:


logger = get_logger()

#logger.info("STEP 3: Getting best hyperparameter choices")
print("STEP 3: Getting best hyperparameter choices")

# Keep track of parsed datasets
parsed_datasets = []

# Store all results of parsed datasets in a dataframe
results_df = None

# Get evaluations performed in order from last evaluation to first
evaluation_dirs = sorted(os.listdir(HYPERPARAMS_EVAL_PATH), reverse=True)

# Iterate over all evaluation dirs
for eval_dir in evaluation_dirs:
    
    # Get all dataset-specific subdirs in evaluation dir
    eval_path = os.path.join(HYPERPARAMS_EVAL_PATH, eval_dir)
    dataset_dirs = get_directories_in_path(eval_path)
    
    # Iterate over dataset-specific subdirs
    for dataset_name in dataset_dirs:
        
        # Make sure we have not added a later evaluation of this dataset to results
        if not dataset_name in parsed_datasets:
            
            dataset_eval_path = os.path.join(eval_path, dataset_name)
            results_file_path = os.path.join(dataset_eval_path, 'dataset-results.csv')
            
            # Check that a results file exists (i.e. we are not currently running this evaluation)
            if os.path.exists(results_file_path):

                # Load results dataframe
                #df = DF.from_csv(results_file_path)
                df = pd.read_csv(results_file_path)
                
                # For each result, add its corresponding test index
                df['original_index'] = df.index
                for model_name, group in df.groupby('model'):
                    model_df = group.reset_index(drop=True)
                    df.loc[model_df['original_index'], 'test_index'] = model_df.index
                df.drop('original_index', axis=1, inplace=True)
                
                # For each result, add the path to the trained models for all folds
                def get_relative_path_of_trained_models(results_row):
                    
                    # Get test dir name
                    test_dir_name = 'test_{}'.format(int(results_row.test_index))
                    
                    # Get base model name
                    model_name = results_row.model
                    base_model_name = model_name.replace('MC', '')
                    
                    # Return relative path
                    abs_path = os.path.join(dataset_eval_path, base_model_name, test_dir_name, model_name)
                    return os.path.relpath(abs_path, os.getcwd())
                
                df['path'] = df.apply(get_relative_path_of_trained_models, axis=1)
                
                # Mark dataset as added
                parsed_datasets.append(dataset_name)
                
                # Append df to collection of all results 
                results_df = df.reset_index(drop=True) if results_df is None else results_df.append(df, ignore_index=True)
                
#logger.info(results_df.groupby(['dataset_name', 'model', 'batch_size']).cv_rmse.min())
print(results_df.groupby(['dataset_name', 'model', 'batch_size']).cv_rmse.min())

idx = results_df.groupby(['dataset_name', 'model']).cv_rmse.transform(min) == results_df.cv_rmse
best_results_df = results_df[idx]
#logger.info(best_results_df)
print(best_results_df)

# Summarize best results in a dict to be dumped as yml
parsed_datasets = list(set(d for d in best_results_df.dataset_name)) 
parsed_models = list(set(d for d in best_results_df.model))

best_results_dict = {d: {m: {} for m in parsed_models} for d in parsed_datasets}

for i, row in best_results_df.iterrows():
    config_dict_keys = ['batch_size', 'lambda', 'dropout', 'cv_rmse', 'path', 'cv_epoch']
    best_results_dict[row.dataset_name][row.model] = { k: row[k] for k in config_dict_keys if not pd.isnull(row[k])}

dump_yaml(best_results_dict, os.getcwd(), 'grid_search_results.yml')
#logger.info("DONE STEP 3")
print("DONE STEP 3")


# In[ ]:


logger = get_logger()

#logger.info("STEP 4: Running TAU optimization")
print("STEP 4: Running TAU optimization")

RANDOM_SEED_NP = 2
RANDOM_SEED_TF = 1

tf.logging.set_verbosity(tf.logging.ERROR)

def run(c, X, y, tau):
    tau = tau[0]
    
    avg_metric_vals = []
    
    # Get path of trained models for all folds
    trained_models_path = os.path.join(os.getcwd(), c['trained_models_path'])
    
    # Get path of indices for all folds
    folds_indices_path = os.path.join(trained_models_path, '..', '..', '..', 'fold_indices')
    
    # Iterate over folds
    for fold_index, fold_model_dir in enumerate(sorted(os.listdir(trained_models_path))):
        
        # Get path of trained model for this fold
        fold_model_path = os.path.join(trained_models_path, fold_model_dir)
        
        # Create a dataset from fold indices
        X_train, y_train, X_val, y_val = dl.load_fold(folds_indices_path, fold_index, X, y)
        dataset = Dataset(X_train, 
                          y_train, 
                          X_val, 
                          y_val, 
                          s['discard_leftovers'],
                          normalize_X=s['normalize_X'], 
                          normalize_y=s['normalize_y'])
            
        # Load the saved model
        with tf.Graph().as_default() as g:
            
            with g.device('/cpu:0'):
                
                # Set random generator seed for reproducible models
                tf.set_random_seed(RANDOM_SEED_TF)
                
                with tf.name_scope("model_{}".format(fold_index)) as scope:
                    if c['base_model_name'] in ['BN', 'MCBN']:
                        model = ModelBN(c['n_hidden'],
                                        K=c['k'],
                                        nonlinearity=c['nonlinearity'],
                                        bn=True,
                                        do=False,
                                        tau=tau,
                                        dataset=dataset,
                                        in_dim=c['in_dim'],
                                        out_dim=c['out_dim'])
                    elif c['base_model_name'] in ['DO', 'MCDO']:
                        keep_prob = 1 - c['dropout']
                        model = ModelDO(c['n_hidden'], 
                                        K=c['k'], 
                                        nonlinearity=c['nonlinearity'], 
                                        bn=False, 
                                        do=True,
                                        tau=tau, 
                                        dataset=dataset, 
                                        in_dim=c['in_dim'], 
                                        out_dim=c['out_dim'],
                                        first_layer_do=True)

                    model.initialize(l2_lambda=c['lambda'], learning_rate=c['learning_rate'])

            #Start session (regular session is default session in with statement)
            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=False,
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1)) as sess:
                
                saver.restore(sess, tf.train.latest_checkpoint(fold_model_path))

                #Get CV PLL
                if c['model_name'] == 'MCBN':
                    samples = model.get_mc_samples(c['mc_samples'], dataset.X_test, c['batch_size'])
                    mean, var = model.get_mc_moments(samples)
                    if s['tau_opt_metric'] == 'PLL':
                        avg_metric = pll(samples, dataset.y_test, c['mc_samples'], tau)
                    else:
                        avg_metric = crps(dataset.y_test, mean, var)
                
                elif c['model_name'] == 'MCBN const':
                    samples = model.get_mc_samples(c['mc_samples'], dataset.X_test, c['batch_size'])
                    mean, var = model.get_mc_moments(samples)
                    if s['tau_opt_metric'] == 'PLL':
                        avg_metric = pll(np.array([mean]), dataset.y_test, 1, tau)
                    else:
                        avg_metric = crps(dataset.y_test, mean, tau**(-1))
                    
                elif c['model_name'] == 'BN':
                    model.update_layer_statistics(dataset.X_test)
                    samples = model.predict(dataset.X_test)
                    if s['tau_opt_metric'] == 'PLL':
                        avg_metric = pll(np.array([samples]), dataset.y_test, 1, tau)
                    else:
                        avg_metric = crps(dataset.y_test, samples, tau**(-1))
                    
                    
                elif c['model_name'] == 'MCDO':
                    samples = model.get_mc_samples(c['mc_samples'], dataset.X_test, keep_prob)
                    mean, var = model.get_mc_moments(samples)
                    if s['tau_opt_metric'] == 'PLL':
                        avg_metric = pll(samples, dataset.y_test, c['mc_samples'], tau)
                    else:
                        avg_metric = crps(dataset.y_test, mean, var)
                    
                elif c['model_name'] == 'MCDO const':
                    samples = model.get_mc_samples(c['mc_samples'], dataset.X_test, keep_prob)
                    mean, var = model.get_mc_moments(samples)
                    if s['tau_opt_metric'] == 'PLL':
                        avg_metric = pll(np.array([mean]), dataset.y_test, 1, tau)
                    else:
                        avg_metric = crps(dataset.y_test, mean, tau**(-1))
                
                elif c['model_name'] == 'DO':
                    samples = model.predict(dataset.X_test, 1)
                    if s['tau_opt_metric'] == 'PLL':
                        avg_metric = pll(np.array([samples]), dataset.y_test, 1, tau)
                    else:
                        avg_metric = crps(dataset.y_test, samples, tau**(-1))

        avg_metric_vals += [avg_metric]
    
    ALL_TAUS.append(tau)
    ALL_PLLS.append(np.mean(avg_metric_vals))
    
    #logger.info("tau {:.10f}: Avg {} {:.4f}".format(tau, s['tau_opt_metric'], np.mean(avg_metric_vals)))
    print("tau {:.10f}: Avg {} {:.4f}".format(tau, s['tau_opt_metric'], np.mean(avg_metric_vals)))
    
    #Minimization objective: avg CRPS or avg. negative PLLs
    return (-1 if s['tau_opt_metric'] == 'PLL' else 1) * np.mean(avg_metric_vals)

def save_results(tau_opt, eval_path, c):
    
    # Get dataset dir
    dataset_path = os.path.join(eval_path, c['dataset_name'])
    make_path_if_missing(dataset_path)
    
    #Save csv for model
    df = DF({'tau':ALL_TAUS, 'pll':ALL_PLLS})
    df.to_csv(os.path.join(dataset_path, c['model_name'] + '.csv'))
    
    #Save plot for model
    fig = plt.figure(figsize=(15,10))
    plt.scatter(ALL_TAUS, ALL_PLLS)
    plt.xlabel('TAU')
    plt.ylabel('PLL')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, c['model_name'] + '.png'))
    
    #Save best result for model
    np.savetxt(os.path.join(dataset_path, c['model_name'] + '_opt.txt'), [tau_opt])
    
s = get_setup()
g = get_grid_search_results()

# Create parent evaluation dir
eval_path = get_new_dir_in_parent_path(TAU_EVAL_PATH)

# Save used setup and grid search results
dump_yaml(s, eval_path, 'eval_setup.yml')
dump_yaml(g, eval_path, 'eval_grid_search_results.yml')



# Iterate over each dataset
#for dataset_name, models_dict in g.iteritems():
for dataset_name, models_dict in g.items():
    
    #logger.info("Dataset: " + dataset_name)
    print("Dataset: " + dataset_name)
    
    # Load dataset in memory
    X, y = dl.load_uci_data_full(dataset_name)
    feature_indices, target_indices = dl.load_uci_info(dataset_name)
    
    # Get min and max of optimization range for tau
    ranges = s['tau_range'][dataset_name]
    tau_min = ranges[0]
    tau_max = ranges[1]
    
    # Add 'const' variants of MC models if present
    optimized_MC_models = [m for m in models_dict.keys() if 'MC' in m]
    for mc_model in optimized_MC_models:
        models_dict[mc_model + ' const'] = models_dict[mc_model]
        
    # Iterate over each model's optimized learning parameters for this dataset
    #for model_name, opt_dict in models_dict.iteritems():
    for model_name, opt_dict in models_dict.items():
        
        # Set random generator seed for identical batch picks for standard and const variant
        np.random.seed(RANDOM_SEED_NP)
        
        #logger.info("Model: " + model_name)
        print("Model: " + model_name)
            
        # Get dataset configuration
        c = {'dataset_name': dataset_name,
             'base_model_name': model_name.replace(' const', ''),
             'model_name': model_name,
             'in_dim': len(feature_indices),
             'out_dim': len(target_indices),
             'n_hidden': s['n_hidden'],
             'k': s['k_specific'].get(dataset_name) or s['k'],
             'nonlinearity': s['nonlinearity'],
             'lambda': opt_dict['lambda'],
             'batch_size': opt_dict['batch_size'],
             'trained_models_path': opt_dict['path'],
             'dropout': opt_dict.get('dropout'), # Can be None
             'learning_rate': s['learning_rate'],
             'mc_samples': s['mc_samples']
             }

        # Save incremental results during optimization
        ALL_TAUS = []
        ALL_PLLS = []
        
        # Make optimization run take extra arguments
        optimize_fun = partial(run, c, X, y)      
        
        tau_opt = gbrt_minimize(optimize_fun, 
                                [(tau_min, tau_max)], 
                                n_random_starts=s['tau_opt_n_random_calls'], 
                                n_calls=s['tau_opt_n_total_calls'])
        save_results(tau_opt.x[0], eval_path, c)
        
#logger.info("DONE STEP 4")
print("DONE STEP 4")


# In[ ]:


logger = get_logger()

#logger.info("STEP 5: Get best TAU results")
print("STEP 5: Get best TAU results")

s = get_setup()

# Get evaluations performed in order from last evaluation to first
evaluation_dirs = sorted(os.listdir(TAU_EVAL_PATH), reverse=True)

# Keep track of parsed datasets
parsed_datasets = []

# Add 'const' variants to optimized models
optimized_models = deepcopy(s['models'])
optimized_models += [m + ' const' for m in optimized_models if 'MC' in m]

# Store all results in a dict
results = {}

# Iterate over all evaluation dirs
for eval_dir in evaluation_dirs:
    
    # Get all dataset-specific subdirs in evaluation dir
    eval_path = os.path.join(TAU_EVAL_PATH, eval_dir)
    dataset_dirs = get_directories_in_path(eval_path)
    
    # Iterate over dataset-specific subdirs
    for dataset_name in dataset_dirs:
        
        # Make sure we have not added a later evaluation of this dataset to results
        if not dataset_name in parsed_datasets:
            
            dataset_eval_path = os.path.join(eval_path, dataset_name)
            results[dataset_name] = {}
            
            # Iterate over optimized models
            for model in optimized_models:
                
                # Get optimized tau
                opt_result_path = os.path.join(dataset_eval_path, model + '_opt.txt')
                tau = np.loadtxt(opt_result_path, dtype=float).tolist()
                
                results[dataset_name][model] = tau
            
            parsed_datasets.append(dataset_name)

dump_yaml(results, os.getcwd(), 'tau_results.yml')
#logger.info("DONE STEP 5")
print("DONE STEP 5")


# In[ ]:


logger = get_logger()

#logger.info("STEP 6: Test set evaluation")
print("STEP 6: Test set evaluation")

RANDOM_SEED_NP_FIRST_RUN = 1
RANDOM_SEED_TF_FIRST_RUN = 1

# Read in config
s = get_setup()

# Read in grid search results
g = get_grid_search_results()

# Read in tau optimization results
t = get_tau_results()

def evaluate_dataset(c, X_train, y_train, X_test, y_test, tf_seed):
    
    dataset = Dataset(X_train, 
                      y_train, 
                      X_test, 
                      y_test, 
                      s['discard_leftovers'],
                      normalize_X=s['normalize_X'], 
                      normalize_y=s['normalize_y'])
    
    # Initialize results dict
    results = {k: [] for k in ['model', 'epoch', 'PLL', 'CRPS', 'RMSE']}
    
    # Get and initialize model
    with tf.Graph().as_default() as g:

        with g.device('/cpu:0'):
            
            # Set random generator seed for reproducible models
            tf.set_random_seed(tf_seed)
            
            # Note: Tau must be set to base model (MCBN or MCDO) tau for get_mc_moments 
            # to be able to find var (overridden for const in metrics calc)
            if c['base_model_name'] in ['BN', 'MCBN']:
                model = ModelBN(s['n_hidden'],
                                K=c['k'],
                                nonlinearity=s['nonlinearity'],
                                bn=True,
                                do=False,
                                tau=c['taus'][c['base_model_name']],
                                dataset=dataset,
                                in_dim=c['in_dim'],
                                out_dim=c['out_dim'])
            elif c['base_model_name'] in ['DO', 'MCDO']:
                keep_prob = 1 - c['dropout']
                model = ModelDO(s['n_hidden'], 
                                K=c['k'], 
                                nonlinearity=s['nonlinearity'], 
                                bn=False, 
                                do=True,
                                tau=c['taus'][c['base_model_name']], 
                                dataset=dataset, 
                                in_dim=c['in_dim'], 
                                out_dim=c['out_dim'],
                                first_layer_do=True)

            model.initialize(l2_lambda=c['lambda'], learning_rate=s['learning_rate'])

        # Start session (regular session is default session in with statement)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=False,
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1)) as sess:
            
            sess.run(tf.global_variables_initializer())
            
            # Train model n_epochs iterations and get test results
            last_epoch = 1
            while dataset.curr_epoch <= c['n_epochs']:
                
                new_epoch = last_epoch < dataset.curr_epoch
                last_epoch = dataset.curr_epoch
                
                # Each eval_interval:th epoch, save all metrics
                if new_epoch and (dataset.curr_epoch % s['test_eval_interval'] == 0
                                  or dataset.curr_epoch == c['n_epochs']):
                    
                    # BN BASED MODELS
                    if c['base_model_name'] == 'MCBN':
                        # MCBN
                        samples = model.get_mc_samples(s['mc_samples'], dataset.X_test, c['batch_size'])
                        mean, var = model.get_mc_moments(samples)
                        
                        results['model'] += ['MCBN']
                        results['epoch'] += [dataset.curr_epoch]
                        results['PLL']   += [pll(samples, dataset.y_test, s['mc_samples'], c['taus']['MCBN'])]
                        results['RMSE']  += [rmse(mean, dataset.y_test)]
                        results['CRPS']  += [crps(dataset.y_test, mean, var)]

                        # MCBN const
                        results['model'] += ['MCBN const']
                        results['epoch'] += [dataset.curr_epoch]
                        results['PLL']   += [pll(np.array([mean]), dataset.y_test, 1, c['taus']['MCBN const'])]
                        results['RMSE']  += [rmse(mean, dataset.y_test)]
                        results['CRPS']  += [crps(dataset.y_test, mean, c['taus']['MCBN const']**(-1))]
                        
                        # At final epoch, save prediction mean, var and true y
                        if dataset.curr_epoch == c['n_epochs']:
                            final_predictions = {
                                'yHat': mean.ravel(),
                                'MCBN var': var.ravel(),
                                'MCBN const var': [c['taus']['MCBN const']**(-1)] * len(mean),
                                'y': dataset.y_test.ravel()
                            }
                            optimum_predictions = {
                                'MCBN PLL_opt': [pll_maximum(mean, dataset.y_test)],
                                'MCBN CRPS_opt': [crps_minimum(mean, dataset.y_test)]
                            }
                    
                    elif c['base_model_name'] == 'BN':
                        model.update_layer_statistics(dataset.X_train)
                        samples = model.predict(dataset.X_test)
                        
                        results['model'] += ['BN']
                        results['epoch'] += [dataset.curr_epoch]
                        results['PLL']   += [pll(np.array([samples]), dataset.y_test, 1, model.tau)]
                        results['RMSE']  += [rmse(samples, dataset.y_test)]
                        results['CRPS']  += [crps(dataset.y_test, samples, model.tau**(-1))]
                    
                    # DO BASED MODELS
                    elif c['base_model_name'] == 'MCDO':
                        # MCDO
                        samples = model.get_mc_samples(s['mc_samples'], dataset.X_test, keep_prob)
                        mean, var = model.get_mc_moments(samples)

                        results['model'] += ['MCDO']
                        results['epoch'] += [dataset.curr_epoch]
                        results['PLL']   += [pll(samples, dataset.y_test, s['mc_samples'], c['taus']['MCDO'])]
                        results['RMSE']  += [rmse(mean, dataset.y_test)]
                        results['CRPS']  += [crps(dataset.y_test, mean, var)]

                        # MCDO const
                        results['model'] += ['MCDO const']
                        results['epoch'] += [dataset.curr_epoch]
                        results['PLL']   += [pll(np.array([mean]), dataset.y_test, 1, c['taus']['MCDO const'])]
                        results['RMSE']  += [rmse(mean, dataset.y_test)]
                        results['CRPS']  += [crps(dataset.y_test, mean, c['taus']['MCDO const']**(-1))]
                        
                        # At final epoch, save prediction mean, var and true y
                        if dataset.curr_epoch == c['n_epochs']:
                            final_predictions = {
                                'yHat': mean.ravel(),
                                'MCDO var': var.ravel(),
                                'MCDO const var': [c['taus']['MCDO const']**(-1)] * len(mean),
                                'y': dataset.y_test.ravel()
                            }
                            optimum_predictions = {
                                'MCDO PLL_opt': [pll_maximum(mean, dataset.y_test)],
                                'MCDO CRPS_opt': [crps_minimum(mean, dataset.y_test)]
                            }
                        
                    elif c['base_model_name'] == 'DO':
                        samples = model.predict(dataset.X_test, 1)
                                                
                        results['model'] += ['DO']
                        results['epoch'] += [dataset.curr_epoch]
                        results['PLL']   += [pll(np.array([samples]), dataset.y_test, 1, model.tau)]
                        results['RMSE']  += [rmse(samples, dataset.y_test)]
                        results['CRPS']  += [crps(dataset.y_test, samples, model.tau**(-1))]
                
                
                batch = dataset.next_batch(c['batch_size'])
                if c['base_model_name'] in ['BN', 'MCBN']:
                    model.run_train_step(batch)
                elif c['base_model_name'] in ['DO', 'MCDO']:
                    model.run_train_step(batch, keep_prob)
    
    if 'MC' in c['base_model_name']:
        return DF(results), DF(final_predictions), DF(optimum_predictions)
    return DF(results), None, None

def save_dataset_results(results_df, final_predictions_df, bm_opt_df, eval_path, dataset_name, base_model_name):
    # Get dataset dir
    dataset_path = os.path.join(eval_path, dataset_name)
    make_path_if_missing(dataset_path)
    
    # Group by model and save
    for model_name, model_df in results_df.groupby('model'):
        model_df.reset_index(drop=True).to_csv(os.path.join(dataset_path, model_name + '.csv'))
        
    # Save final predictions dataframe
    final_predictions_df.to_csv(os.path.join(dataset_path, base_model_name + ' final_predictions.csv'))
    
    # Save optimum predictions dataframe
    bm_opt_df.to_csv(os.path.join(dataset_path, base_model_name + ' optimum_predictions.csv'))
    
    
# Create parent evaluation dir
eval_path = get_new_dir_in_parent_path(TEST_EVAL_PATH)

# Save used setup, grid search and tau results
dump_yaml(s, eval_path, 'eval_setup.yml')
dump_yaml(g, eval_path, 'eval_grid_search_results.yml')
dump_yaml(t, eval_path, 'tau_results.yml')

all_results = None

# Iterate over datasets to be evaluated
for dataset_name in t.keys():
    
    #logger.info("Dataset: " + dataset_name)
    print("Dataset: " + dataset_name)
    
    # Load dataset into memory
    X_train, y_train, X_test, y_test = dl.load_uci_data_test(dataset_name)
    feature_indices, target_indices = dl.load_uci_info(dataset_name)
    
    # Iterate over base optimization models (BN or DO)
    train_and_evaluation_models = get_train_and_evaluation_models(s['models'])
    
    #for bn_or_do_model, evaluation_models in train_and_evaluation_models.iteritems():
    for bn_or_do_model, evaluation_models in train_and_evaluation_models.items():
                
        # Get grid search parameters
        bm_results_df = None
        bm_final_pred_df = None
        bm_opt_df = None
        for base_model_name in evaluation_models:
            
            # Run multiple times based on n_testruns
            for run_count in range(s['n_testruns']):
            
                # Set random generator seed for reproducible batch order
                # Common for all base models for a certain dataset
                np_seed = RANDOM_SEED_NP_FIRST_RUN + run_count
                tf_seed = RANDOM_SEED_TF_FIRST_RUN + run_count
                
                np.random.seed(np_seed)
                opt_dict = g[dataset_name][base_model_name]


                #logger.info("Model: {}, run: {} of {}".format(base_model_name, run_count+1, s['n_testruns']))
                print("Model: {}, run: {} of {}".format(base_model_name, run_count+1, s['n_testruns']))

                # Get dataset configuration
                c = {'base_model_name': base_model_name,
                     'in_dim': len(feature_indices),
                     'out_dim': len(target_indices),
                     'k': s['k_specific'].get(dataset_name) or s['k'],
                     'lambda': opt_dict['lambda'],
                     'batch_size': opt_dict['batch_size'],
                     'dropout': opt_dict.get('dropout'), # Can be None
                     'taus': t[dataset_name],
                     'n_epochs': g[dataset_name][base_model_name]['cv_epoch']
                    }

                df, df_fp, df_opt = evaluate_dataset(c, X_train, y_train, X_test, y_test, tf_seed)
                
                df['run_count'] = run_count+1
                #logger.info(df)
                print(df)
                
                bm_results_df = df if bm_results_df is None else bm_results_df.append(df, ignore_index=True)
                
                if df_fp is not None:
                    df_fp['run_count'] = run_count+1
                    bm_final_pred_df = df_fp if bm_final_pred_df is None else bm_final_pred_df.append(df_fp, ignore_index=True)
                
                if df_opt is not None:
                    df_opt['run_count'] = run_count+1
                    #logger.info(df_opt)
                    print(df_opt)
                    bm_opt_df = df_opt if bm_opt_df is None else bm_opt_df.append(df_opt, ignore_index=True)
                
        save_dataset_results(bm_results_df, bm_final_pred_df, bm_opt_df, eval_path, dataset_name, 'MC'+bn_or_do_model)

        bm_results_df['dataset'] = dataset_name
        all_results = bm_results_df if all_results is None else all_results.append(bm_results_df, ignore_index=True)

    # Save all results
    all_results.to_csv(os.path.join(eval_path, 'results.csv'))
    
#logger.info("DONE STEP 6")
print("DONE STEP 6")


# In[ ]:


s = get_setup()
dataset_names = s['datasets']
all_models = s['models'] + ['{} const'.format(m) for m in s['models'] if 'MC' in m]

split_count = {dn: 0 for dn in dataset_names}
all_results_df = None

for dirname in sorted(os.listdir(TEST_EVAL_PATH)):
    dataset_split_eval_path = os.path.join(TEST_EVAL_PATH, dirname)
    dataset_name = get_directories_in_path(dataset_split_eval_path)[0]
    
    split_count[dataset_name] += 1
    results_dir_path = os.path.join(dataset_split_eval_path, dataset_name)
    
    for m in all_models:
        results_path = os.path.join(results_dir_path, '{}.csv'.format(m))
        #results_df = DF.from_csv(results_path)
        results_df = pd.read_csv(results_path)
        
        final_result_df = results_df.groupby('run_count').tail(1).reset_index(drop=True)
        final_result_df['split_seed'] = s['split_seed']
        final_result_df['dataset'] = dataset_name
        
        if 'MC' in m:
            base_model = m.replace(' const', '')
            optimal_results_path = os.path.join(results_dir_path, '{} optimum_predictions.csv'.format(base_model))
            #optimal_results_df = DF.from_csv(optimal_results_path)
            optimal_results_df = pd.read_csv(optimal_results_path)
            
            # CRPS
            final_result_df['CRPS_opt'] = optimal_results_df['{} CRPS_opt'.format(base_model)].values
            
            # PLL
            final_result_df['PLL_opt'] = optimal_results_df['{} PLL_opt'.format(base_model)].values
        else:
            final_result_df['CRPS_opt'] = None
            final_result_df['PLL_opt'] = None
        
        if all_results_df is None:
            all_results_df = final_result_df
        else:
            all_results_df = all_results_df.append(final_result_df, ignore_index=True)
        
col_order = ['dataset','model','split_seed','run_count','CRPS','PLL','RMSE','CRPS_opt','PLL_opt']
all_results_df = all_results_df[col_order]
all_results_df['model'].replace({'MCBN const': 'CUBN', 'MCDO const': 'CUDO'}, inplace=True)
all_results_df.to_csv('final_results.csv')
