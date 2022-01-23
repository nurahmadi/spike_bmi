"""
Optimize hyperparameters for deep learning based BMI decoders using Optuna
"""

# import packages
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import h5py
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, transform_data
from bmi.utils import seed_tensorflow
from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
import time as timer

def main(args):
    run_start = timer.time()
    
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]   
    # select the x-y velocity components
    y = y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)
    
    def objective(trial):
        # define hyperparameter space
        if args.decoder == 'qrnn':
            max_timesteps = 5
            max_layers = 1
            max_units = 600
        elif args.decoder == 'lstm':
            max_timesteps = 5
            max_layers = 1
            max_units = 250
        elif args.decoder == 'mlp':
            max_timesteps = 1
            max_layers = 3
            max_units = 400
        config = {"input_dim": X.shape[-1],
                  "output_dim": y.shape[-1],
                  "timesteps": trial.suggest_int("timesteps", 1, max_timesteps),
                  "n_layers": trial.suggest_int("n_layers", 1, max_layers),
                  "units": trial.suggest_int("units", 50, max_units, step=50),
                  "window_size": 2,
                  "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96]),
                  "epochs": 100,
                  "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                  "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
                  "optimizer": trial.suggest_categorical("optimizer", ['Adam', 'RMSProp']),
                  "loss": 'mse',
                  "metric": 'mse'}
        print(f"Hyperparameter configuration: {config}")

        # set seed for reproducibility
        seed_tensorflow(args.seed)

        rmse_valid_folds = []
        best_epochs = []
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

            if (args.decoder == 'qrnn') or (args.decoder == 'lstm'):
                # transform data into sequence data
                X_train, y_train = transform_data(X_train, y_train, timesteps=config["timesteps"])
                X_valid, y_valid = transform_data(X_valid, y_valid, timesteps=config["timesteps"])

            # early stopping callback
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=args.verbose, mode='min', restore_best_weights=True)
            # pruning trial callback
            pruning = TFKerasPruningCallback(trial, 'val_loss')
            # define callbacks
            callbacks = [earlystop, pruning]

            # Create and compile model
            print("Compiling and training a model")
            if args.decoder == 'qrnn':
                model = QRNNDecoder(config)
            elif args.decoder == 'lstm':
                model = LSTMDecoder(config)
            elif args.decoder == 'mlp':
                model = MLPDecoder(config)
            # fit model
            train_start = timer.time()
            history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=config['epochs'], verbose=args.verbose, callbacks=callbacks)
            train_end = timer.time()
            train_time = (train_end - train_start) / 60
    
            if earlystop.stopped_epoch != 0:
                stop_epoch = earlystop.stopped_epoch + 1
            else:
                stop_epoch = config['epochs']
            best_epoch = np.argmin(history.history['val_loss']) + 1
            best_epochs.append(best_epoch)
            print(f"Training stopped at epoch {stop_epoch} with the best epoch at {best_epoch}")
            print(f"Training the model took {train_time:.2f} minutes")

            # predict using the trained model
            y_valid_pred = model.predict(X_valid, batch_size=config['batch_size'], verbose=args.verbose)

            # evaluate performance
            print("Evaluating the model performance")
            rmse_valid = mean_squared_error(y_valid, y_valid_pred, squared=False)
            rmse_valid_folds.append(rmse_valid)

        epochs = int(np.asarray(best_epochs).mean())
        objective.epochs = epochs
        rmse_valid_mean = np.asarray(rmse_valid_folds).mean()
        return rmse_valid_mean

    # create and optimize study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print(f"Storing study trial into a file: {args.output_filepath}")
    with open(args.output_filepath, 'wb') as f:
        pickle.dump(study, f)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best hyperparameters:")
    best_params = study.best_trial.params
    best_params["epochs"] = objective.epochs
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    param_filepath = f"{args.output_filepath.split('.')[0]}.json"
    print(f"Storing best params into a file: {param_filepath}")
    with open(param_filepath, 'w') as f:
        json.dump(best_params, f)

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print (f"Whole processes took {run_time:.2f} minutes")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
    parser.add_argument('--seed',             type=float, default=42,     help='Seed for reproducibility')
    parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
    parser.add_argument('--decoder',          type=str,   default='qrnn', help='Deep learning based decoding algorithm')
    parser.add_argument('--n_folds',          type=int,   default=5,      help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')
    parser.add_argument('--verbose',          type=int,   default=0,      help='Wether or not to print the output')
    parser.add_argument('--n_trials',         type=int,   default=2,      help='Number of trials for optimization')
    parser.add_argument('--timeout',          type=int,   default=300,    help='Stop study after the given number of seconds')
    parser.add_argument('--n_startup_trials', type=int,   default=1,      help='Number of trials for which pruning is disabled')
    
    args = parser.parse_args()
    main(args)
