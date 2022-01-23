"""
Process the raw data to get spike and kinematic data
"""

# import packages
import argparse
import h5py
import numpy as np
from bmi.utils import flatten_list

def main(args):

    num_chan = 96
    print (f"Reading raw data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_pos = f['cursor_pos'][()].T   # transpose to be N x 2 dimension, where N = number of samples
        target_pos = f['target_pos'][()].T # transpose to be N x 2 dimension, where N = number of samples
        task_time = f['t'][()].squeeze()   # time associated with the task, squeeze the format data into 1D
        spikes = f['spikes'][()].T # transpose to be shape: number of channels x number of units
        num_unit = spikes.shape[1]
        print(f"Number of channels: {num_chan}, number of units: {num_unit}")
        all_spikes = [] # list all spikes from num_chan x num_unit
        for i in range(num_chan):
            chan_spikes = []
            for j in range(num_unit):
                if (f[spikes[i,j]].ndim == 2):
                    tmp_spikes = f[spikes[i,j]][()].squeeze(axis=0) # dimension: num of spikes, remove first dimension axis=0
                else:
                    tmp_spikes = np.empty(0)
                chan_spikes.append(tmp_spikes)
            all_spikes.append(chan_spikes)

    sua_trains = []
    len_sua_trains = []    
    for i in range(num_chan):
        for j in range(num_unit):
            if (j > 0) & (all_spikes[i][j].shape[0] > 0): # first unit (j=0) is unsorted unit
                #print(f"Include channel-{i}, unit-{j}, shape: {all_spikes[i][j].shape}")
                sua_train = all_spikes[i][j]
                sua_idx = np.where((sua_train >= task_time[0]) & (sua_train <= task_time[-1]))[0]
                sua_train = sua_train[sua_idx]
                sua_trains.append(sua_train)
                len_sua_trains.append(len(sua_train))      
    num_sua = len(sua_trains)  

    # Computing MUA or threshold crossings    
    mua_trains = []   
    len_mua_trains = []   
    for i in range(num_chan):
        chan_mua = []
        for j in range(num_unit):
            if (all_spikes[i][j].shape[0] > 0):
                mua_train = all_spikes[i][j]
                mua_idx = np.where((mua_train >= task_time[0]) & (mua_train <= task_time[-1]))[0]
                mua_train = mua_train[mua_idx]
                chan_mua.append(mua_train)
        chan_mua = flatten_list(chan_mua)
        chan_mua.sort()
        mua_trains.append(np.asarray(chan_mua))
        len_mua_trains.append(len(chan_mua))
    num_mua = len(mua_trains) 

    print(f"[Before filtering] Number of SUA: {len(sua_trains)}, Number of MUA: {len(mua_trains)}")

    # minimum spike rate for unit to be included
    min_spikerate = 0.5
    task_duration = task_time[-1] - task_time[0]
    min_numspike = int(np.round(min_spikerate * task_duration))

    len_sua_trains = np.asarray(len_sua_trains)
    len_mua_trains = np.asarray(len_mua_trains)

    sua_valid_idx = np.where(len_sua_trains > min_numspike)[0]
    mua_valid_idx = np.where(len_mua_trains > min_numspike)[0]

    sua_trains_valid = []
    for idx in sua_valid_idx:
        sua_trains_valid.append(sua_trains[idx])

    mua_trains_valid = []
    for idx in mua_valid_idx:
        mua_trains_valid.append(mua_trains[idx])

    print(f"[After filtering] Number of SUA: {len(sua_trains_valid)}, Number of MUA: {len(mua_trains_valid)}")

    # calculate velocity and acceleration   
    dt_task = np.diff(task_time).mean() # sampling period (0.004 sec)
    task_vel = np.diff(task_pos, axis=0) / dt_task # in mm/s
    task_acc = np.diff(task_vel, axis=0) / dt_task # in mm/s^2
    task_vel = np.concatenate((task_vel, task_vel[-1:,:]), axis=0) # padding with the last element
    task_acc = np.concatenate((task_acc, task_acc[-2:,:]), axis=0) # padding with the last 2 elements
    # concatenate position, velocity, and acceleration
    task_data = np.concatenate((task_pos, task_vel, task_acc), axis=1) 

    with h5py.File(args.output_filepath, 'w') as f:
        f['task_time'] = task_time
        f['task_data'] = task_data
        f['target_pos'] = target_pos
        dt = h5py.special_dtype(vlen=np.dtype('f8'))
        f.create_dataset('sua_trains', data=np.asarray(sua_trains_valid, dtype=dt))
        f.create_dataset('mua_trains', data=np.asarray(mua_trains_valid, dtype=dt))
    
    print(f"Finished processing and storing spike and kinematic data into file: {args.output_filepath}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',   type=str,   help='Path to the raw data')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the spike and kinematic data')
    
    args = parser.parse_args()
    main(args)