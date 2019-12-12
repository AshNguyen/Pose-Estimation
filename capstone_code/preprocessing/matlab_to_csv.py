import os
import scipy.io
import pandas as pd

import glob
files = glob.glob('./labels/*.mat')

def extract_matlab_data(file):
    data = scipy.io.loadmat(file)
    xs = data['x']
    ys = data['y']
    visibility = data['visibility']
    w, h = data['dimensions'][0][1], data['dimensions'][0][0]
    action = data['action'][0]
    pose_type = data['pose'][0]
    # bbox = data['bbox']
    file_id = file[-8:-4]
    kp_names = ["head", "left_shoulder", "right_shoulder", "left_elbow", 
                "right_elbow", "left_wrist", "right_wrist", "left_hip", 
                "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    col_names = ['file_id', 'frame']
    col_names = col_names + [name+'_x' for name in kp_names] + [name+'_y' for name in kp_names] + ['visi_'+name for name in kp_names]
    col_names = col_names + ['w', 'h', 'action', 'pose']
    df = pd.DataFrame(columns = col_names)
    for i in range(xs.shape[0]):
        extracted = [file_id, i] + list(xs[i]) + list(ys[i]) + list(visibility[i]) + [w, h, action, pose_type]
        df = df.append(pd.DataFrame([extracted], columns=col_names))
    return df


kp_names = ["head", "left_shoulder", "right_shoulder", "left_elbow", 
            "right_elbow", "left_wrist", "right_wrist", "left_hip", 
            "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
col_names = ['file_id', 'frame']
col_names = col_names + [name+'_x' for name in kp_names] + [name+'_y' for name in kp_names] + ['visi_'+name for name in kp_names]
col_names = col_names + ['w', 'h', 'action', 'pose']

df = pd.DataFrame(columns = col_names)
for file in files:
    extracted = extract_matlab_data(file)
    df = df.append(extracted)


df.to_csv('./processed_penn_data.csv')