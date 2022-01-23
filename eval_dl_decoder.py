"""
Evaluating deep learning based BMI decoders implemented with TensorFlow
"""

# import packages
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, transform_data
from bmi.utils import seed_tensorflow, count_params
from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
from sklearn.metrics import mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer

def main(args):
    run_start = timer.time()
    
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]   
    # select the x-y velocity components
    y = y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

    print("Hyperparameter configuration setting")
    if args.config_filepath:
        # open JSON hyperparameter configuration file
        print(f"Using hyperparameter configuration from a file: {args.config_filepath}")
        with open(args.config_filepath, 'r') as f:
            config = json.load(f)
        
    else:
        # define model configuration
        config = {'timesteps'    : args.timesteps,
                  'n_layers'     : args.n_layers,
                  'units'        : args.units,
                  'batch_size'   : args.batch_size,
                  'learning_rate': args.learning_rate,
                  'dropout'      : args.dropout,
                  'optimizer'    : args.optimizer,
                  'epochs'       : args.epochs}
    config['input_dim'] = X.shape[-1]
    config['output_dim'] = y.shape[-1]
    config['window_size'] = args.window_size
    config['loss'] = args.loss
    config['metric'] = args.metric
    print(f"Hyperparameter configuration: {config}")

    # set seed for reproducibility
    seed_tensorflow(args.seed)

    rmse_test_folds = []
    cc_test_folds = []

    tscv = TimeSeriesSplitCustom(n_splits=args.n_folds, test_size=int(args.test_size*len(y)), min_train_size=int(args.min_train_size*len(y)))
    for train_idx, test_idx in tscv.split(X, y):
        # specify training set
        X_train = X[train_idx,:]            
        y_train = y[train_idx,:]
    
        # specify test set
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
    
        # standardize input data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if (args.decoder == 'qrnn') or (args.decoder == 'lstm'):
            # transform data into sequence data
            X_train, y_train = transform_data(X_train, y_train, timesteps=config['timesteps'])
            X_test, y_test = transform_data(X_test, y_test, timesteps=config['timesteps'])

        # Create and compile model
        print("Compiling and training a model")
        if args.decoder == 'qrnn':
            model = QRNNDecoder(config)
        elif args.decoder == 'lstm':
            model = LSTMDecoder(config)
        elif args.decoder == 'mlp':
            model = MLPDecoder(config)
        total_count, _, _ = count_params(model)
        # fit model
        train_start = timer.time()
        history = model.fit(X_train, y_train, validation_data=None, epochs=config['epochs'], verbose=args.verbose, callbacks=None)
        train_end = timer.time()
        train_time = (train_end - train_start) / 60
        print(f"Training the model took {train_time:.2f} minutes")

        # predict using the trained model
        y_test_pred = model.predict(X_test, batch_size=config['batch_size'], verbose=args.verbose)

        # evaluate performance
        print("Evaluating the model performance")
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
        cc_test = pearson_corrcoef(y_test, y_test_pred)

        rmse_test_folds.append(rmse_test)
        cc_test_folds.append(cc_test)

    for i in range(args.n_folds):
        print(f"Fold-{i+1} | RMSE test = {rmse_test_folds[i]:.2f}, CC test = {cc_test_folds[i]:.2f}")

    print (f"Storing results into file: {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['y_test'] = y_test
        f['y_test_pred'] = y_test_pred
        f['rmse_test_folds'] = np.asarray(rmse_test_folds)
        f['cc_test_folds'] = np.asarray(cc_test_folds)

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print (f"Whole processes took {run_time:.2f} minutes")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
    parser.add_argument('--seed',             type=float, default=42,      help='Seed for reproducibility')
    parser.add_argument('--feature',          type=str,   default='mua',   help='Type of spiking activity (sua or mua)')
    parser.add_argument('--decoder',          type=str,   default='qrnn',  help='Deep learning based decoding algorithm')
    parser.add_argument('--n_folds',          type=int,   default=5,       help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,     help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,     help='Testing data size')
    parser.add_argument('--config_filepath',  type=str,   default='',      help='JSON hyperparameter configuration file')
    parser.add_argument('--timesteps',        type=int,   default=5,       help='Number of timesteps')
    parser.add_argument('--n_layers',         type=int,   default=1,       help='Number of layers')
    parser.add_argument('--units',            type=int,   default=600,     help='Number of units (hidden state size)')
    parser.add_argument('--window_size',      type=int,   default=2,       help='Window size')
    parser.add_argument('--dropout',          type=float, default=0.1,     help='Dropout rate')
    parser.add_argument('--optimizer',        type=str,   default='Adam',  help='Optimizer')
    parser.add_argument('--epochs',           type=int,   default=50,      help='Number of epochs')
    parser.add_argument('--batch_size',       type=int,   default=32,      help='Batch size')
    parser.add_argument('--learning_rate',    type=float, default=0.001,   help='Learning rate')
    parser.add_argument('--loss',             type=str,   default='mse',   help='Loss function')
    parser.add_argument('--metric',           type=str,   default='mse',   help='Predictive performance metric')
    parser.add_argument('--verbose',          type=int,   default=0,       help='Wether or not to print the output')
    
    args = parser.parse_args()
    main(args)
    