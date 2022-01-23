"""
Optimize hyperparameter of Wiener filter
"""

# import packages
import argparse
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, transform_data
from bmi.decoders import WienerDecoder
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

   # define model configuration
    config = {'reg_type'    : args.reg_type,
              'reg_alpha'   : args.reg_alpha,
              'timesteps'   : args.timesteps}    
    print(f"Hyperparameter configuration: {config}")

    rmse_valid_folds = []
    cc_valid_folds = []

    tscv = TimeSeriesSplitCustom(n_splits=args.n_folds, test_size=int(args.test_size*len(y)), min_train_size=int(args.min_train_size*len(y)))
    for train_idx, _ in tscv.split(X, y):
        # specify training set
        X_train = X[train_idx,:]            
        y_train = y[train_idx,:]

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.test_size, shuffle=False)
    
        # standardize input data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

        # transform data into sequence data
        X_train, y_train = transform_data(X_train, y_train, timesteps=args.timesteps)
        X_valid, y_valid = transform_data(X_valid, y_valid, timesteps=args.timesteps)

        # reshape data
        X_train = X_train.reshape(X_train.shape[0], (X_train.shape[1]*X_train.shape[2]), order='F')
        X_valid = X_valid.reshape(X_valid.shape[0], (X_valid.shape[1]*X_valid.shape[2]), order='F')

        # instantiate model
        model = WienerDecoder(args.reg_type, args.reg_alpha)
        # fit model
        train_start = timer.time()
        model.fit(X_train, y_train) # train model
        train_end = timer.time()
        train_time = (train_end - train_start) / 60
        #print(f"Training the model took {train_time:.2f} minutes")

        # predict using the trained model
        y_valid_pred = model.predict(X_valid)

        # evaluate performance
        rmse_valid = mean_squared_error(y_valid, y_valid_pred, squared=False)
        cc_valid = pearson_corrcoef(y_valid, y_valid_pred)

        rmse_valid_folds.append(rmse_valid)
        cc_valid_folds.append(cc_valid)

    for i in range(args.n_folds):
        print(f"Fold-{i+1} | RMSE valid = {rmse_valid_folds[i]:.2f}, CC valid = {cc_valid_folds[i]:.2f}")

    print (f"Storing results into file: {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['rmse_valid_folds'] = np.asarray(rmse_valid_folds)
        f['cc_valid_folds'] = np.asarray(cc_valid_folds)
        
    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print (f"Whole processes took {run_time:.2f} minutes")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
    parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
    parser.add_argument('--timesteps',        type=int,   default=4,      help='Number of timesteps')
    parser.add_argument('--reg_type',         type=str,   default='',     help='Regularization type')
    parser.add_argument('--reg_alpha',        type=float, default=0,      help='Regularization constant')
    parser.add_argument('--n_folds',          type=int,   default=5,      help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')
    
    args = parser.parse_args()
    main(args)
