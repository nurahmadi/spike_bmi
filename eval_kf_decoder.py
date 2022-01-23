"""
Evaluating spike-based BMI decoding using Kalman filter (KF)
"""

# import packages
import argparse
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom
from bmi.decoders import KalmanDecoder
from sklearn.metrics import mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer

def main(args):
    run_start = timer.time()
    
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]   

    # define model configuration
    config = {'reg_type'    : args.reg_type,
              'reg_alpha'   : args.reg_alpha}
    print(f"Hyperparameter configuration: {config}")    

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

        # instantiate model
        model = KalmanDecoder(args.reg_type, args.reg_alpha)
        # fit model
        train_start = timer.time()
        model.fit(X_train, y_train) # train model
        train_end = timer.time()
        train_time = (train_end - train_start) / 60
        #print(f"Training the model took {train_time:.2f} minutes")

        # predict using the trained model
        y_test_pred = model.predict(X_test, y_test[:1,:])
        # select the x-y velocity components
        y_test = y_test[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)
        y_test_pred = y_test_pred[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

        # evaluate performance
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
    # arguments
    parser.add_argument('--input_filepath',   type=str,   help='File path to the dataset')
    parser.add_argument('--output_filepath',  type=str,   help='File path to the stored result')
    parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
    parser.add_argument('--reg_type',         type=str,   default='',     help='Regularization type')
    parser.add_argument('--reg_alpha',        type=float, default=0,      help='Regularization constant')
    parser.add_argument('--n_folds',          type=int,   default=5,      help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')
    
    args = parser.parse_args()
    main(args)
