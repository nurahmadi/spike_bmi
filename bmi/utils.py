"""
List of utility functions
"""

# import packages
import os
import random
import numpy as np
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon
import tensorflow as tf
from optuna.trial import TrialState

def flatten_list(lst):
    """
    Flatten a list of list
    """
    return [item for sublist in lst for item in sublist]

def seed_tensorflow(seed=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def stat_test(x, y, alternative='two-sided'):
    """
    Calculate paired statistical test using the t-test if the data has normal distrubution, otherwise using the Wilcoxon signed-rank test.
    
    Parameters
    ----------
    x, y : ndarray
        Two sets of measurement data, which must have the same shape. 
    alternative : str, {'two-sided', 'less', 'greater'}
        Defines the alternative hypothesis. 
    
    Returns
    ----------
    stat : float
        Statistic.
    pval : float
        The p-value for the test depending on alternative.
    norm : int
        Normality. 1 for normal, 0 for otherwise.
    note : str
        Note to represent statistical significance level.
    """
    alpha = 0.05
    _, p = shapiro(y - x)
    if p > alpha:
        norm = 1
        stat, pval = ttest_rel(x, y, alternative=alternative)
    else:
        norm = 0
        stat, pval = wilcoxon(x, y, alternative=alternative)
        
    if pval < 0.001:
        note = '***'
    elif pval < 0.01:
        note = '**'
    elif pval < 0.05:
        note = '*'
    else:
        note = ''
    
    return stat, pval, norm, note

def conf_interval(x, mode='ci', dist='t', ci=0.95):
    """
    Calculate confidence interval or standard error (of the mean) interval.

    Parameters
    ----------
    x : ndarray
        Input data.
    mode : str, default 'ci'
        If mode = 'ci', calculate confidence interval; otherwise, calculate standard error (of the mean) interval.
    ci : float
        Confidence level, ci = 0.95 for 95% confidence interval.
    dist : str, {'t', 'normal'}
        Distribution type which is either t-distribution or normal distribution.

    Returns
    -------
    mean : ndarray
        Mean value.
    lower : ndarray
        Lower bound value.
    upper : ndarray
        Upper bound value.
    """
    mean = np.mean(x, axis=0)
    sem = stats.sem(x, axis=0)
    if mode=='ci':
        if dist == 't':
            lower, upper = stats.t.interval(ci, x.shape[0]-1, loc=mean, scale=sem)
        elif dist == 'normal':
            lower, upper = stats.norm.interval(ci, loc=mean, scale=sem) 
    else: # mode=='sem':
        lower = mean - sem
        upper = mean + sem        
    return mean, lower, upper

def list2str(lst, sep=','):
    """
    Convert list into string
    """
    return sep.join(str(item) for item in lst)

def str2list(s, sep=',', dtype=str):
    """
    Convert list into string
    """
    return list(map(dtype, s.split(sep)))

def count_params(model):
    """
    Count trainable, non-trainable, and total parameters of a model.
    """
    trainable_count = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count
    print(f"Total params: {total_count}")
    print(f"Trainable params: {trainable_count}")
    print(f"Non-trainable params: {non_trainable_count}")
    return total_count, trainable_count, non_trainable_count

def show_result(study):
    """
    Show the hyperparameter optimization results.
    """
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))