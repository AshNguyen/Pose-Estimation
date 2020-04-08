import os
import scipy.io
import pandas as pd
import numpy as np

def augment_position(xs, ys, n_aug):
    max_x = np.max(np.array(xs).reshape(-1))
    min_x = np.min(np.array(xs).reshape(-1))
    max_y = np.max(np.array(ys).reshape(-1))
    min_y = np.min(np.array(ys).reshape(-1))
    x_aug = [-i*(min_x - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_x)/n_aug for i in range(0, n_aug+1)]
    y_aug = [-i*(min_y - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_y)/n_aug for i in range(0, n_aug+1)]
    augmented_x = []
    augmented_y = []
    for move_x in x_aug: 
        for move_y in y_aug:
            augmented_x.append(np.array(xs)+move_x*np.ones(shape=(xs.shape[0], xs.shape[1])))
            augmented_y.append(np.array(ys)+move_y*np.ones(shape=(ys.shape[0], ys.shape[1])))
    return np.array(augmented_x), np.array(augmented_y)

def augment_size(xs, ys):
    augmented_x, augmented_y = [], []
    for shrink in np.linspace(0.1, 1.0, 10):
        augmented_x.append(xs*shrink)
        augmented_y.append(ys*shrink)
    return np.array(augmented_x), np.array(augmented_y)

def serialize_data_clf(nframe, data, n_aug, skip_frame=1, save=False, save_path=None):
    import joblib
    try:
        train_input = np.load(save_path+"clf_input_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame), allow_pickle=True)
        train_label = np.load(save_path+"clf_label_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame), allow_pickle=True)
        encoder = joblib.load(save_path+"clf_encoder_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
        print('Data exists, loading...')
        return train_input, train_label, encoder
    except:
        print('Creating data...')
        train_input = -np.ones(shape=(1, nframe, 13*3))
        train_label = []
        for ID in data['file_id'].unique():
            df = data[data['file_id'] == ID].sort_values(['frame']).reset_index(drop=True)
            if np.floor(df.shape[0]/skip_frame) < nframe:
                continue

            df = df[::skip_frame]
            df.iloc[:, 2:15] = df.iloc[:, 2:15]/df['w'].iloc[0]
            df.iloc[:, 15:28] = df.iloc[:, 15:28]/df['h'].iloc[0]
            if n_aug > 0:
                xs, ys = augment_position(df.iloc[:, 2:15], df.iloc[:, 15:28], n_aug)

                _class = df['action']
                X_train = []
                y_train = []
                for n in range(xs.shape[0]):
                    for i in range(nframe, df.shape[0]):
                        curr = np.append(xs[n,i-nframe:i,:], ys[n,i-nframe:i,:], axis=1)
                        curr = np.append(curr, df.iloc[i-nframe:i, 28:41], axis=1)
                        X_train.append(curr)
                        y_train.append(_class[i])
            else: 
                _class = df['action']
                X_train = []
                y_train = []
                for i in range(nframe, df.shape[0]):
                    X_train.append(np.array(df.iloc[i-nframe:i, 2:41]))
                    y_train.append(_class[i])
            X_train = np.array(X_train)

            train_input = np.append(train_input, X_train, axis=0)
            train_label = train_label + y_train
        train_input = train_input[1:, :, :]
        train_label = np.array(train_label)

        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder()
        train_label = encoder.fit_transform(train_label.reshape(-1,1)).toarray()
        
        if save:
            train_input.dump(save_path+"clf_input_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
            train_label.dump(save_path+"clf_label_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
            joblib.dump(encoder, save_path+"clf_encoder_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
        
        return train_input, train_label, encoder


def serialize_data_clf_val(train_encoder, nframe, data, n_aug, skip_frame=1, save=False, save_path=None):
    import joblib
    try:
        train_input = np.load(save_path+"clf_input_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame), allow_pickle=True)
        train_label = np.load(save_path+"clf_label_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame), allow_pickle=True)
        encoder = joblib.load(save_path+"clf_encoder_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
        print('Data exists, loading...')
        return train_input, train_label, encoder
    except:
        print('Creating data...')
        train_input = -np.ones(shape=(1, nframe, 13*3))
        train_label = []
        for ID in data['file_id'].unique():
            df = data[data['file_id'] == ID].sort_values(['frame']).reset_index(drop=True)
            if np.floor(df.shape[0]/skip_frame) < nframe:
                continue

            df = df[::skip_frame]
            df.iloc[:, 2:15] = df.iloc[:, 2:15]/df['w'].iloc[0]
            df.iloc[:, 15:28] = df.iloc[:, 15:28]/df['h'].iloc[0]
            if n_aug > 0:
                xs, ys = augment_position(df.iloc[:, 2:15], df.iloc[:, 15:28], n_aug)

                _class = df['action']
                X_train = []
                y_train = []
                for n in range(xs.shape[0]):
                    for i in range(nframe, df.shape[0]):
                        curr = np.append(xs[n,i-nframe:i,:], ys[n,i-nframe:i,:], axis=1)
                        curr = np.append(curr, df.iloc[i-nframe:i, 28:41], axis=1)
                        X_train.append(curr)
                        y_train.append(_class[i])
            else: 
                _class = df['action']
                X_train = []
                y_train = []
                for i in range(nframe, df.shape[0]):
                    X_train.append(np.array(df.iloc[i-nframe:i, 2:41]))
                    y_train.append(_class[i])
            X_train = np.array(X_train)

            train_input = np.append(train_input, X_train, axis=0)
            train_label = train_label + y_train
        train_input = train_input[1:, :, :]
        train_label = np.array(train_label)

        encoder = train_encoder
        train_label = encoder.transform(train_label.reshape(-1,1)).toarray()
        
        if save:
            train_input.dump(save_path+"clf_input_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
            train_label.dump(save_path+"clf_label_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
            joblib.dump(encoder, save_path+"clf_encoder_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
        
        return train_input, train_label, encoder


def serialize_data_next_frame(nframe, data, n_aug, skip_frame=1, save=False, save_path=None):
    try:
        train_input = np.load(save_path+"nf_input_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame), allow_pickle=True)
        train_label = np.load(save_path+"nf_label_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame), allow_pickle=True)
        print('Data exists, loading...')
        return train_input, train_label
    except:
        print('Creating data...')
        train_input = -np.ones(shape=(1, nframe, 13*3))
        train_label = -np.ones(shape=(1, 13*3))
        for ID in data['file_id'].unique():
            df = data[data['file_id'] == ID].sort_values(['frame']).reset_index(drop=True)
            if np.floor(df.shape[0]/skip_frame) < nframe:
                continue

            df = df[::skip_frame]
            df.iloc[:, 2:15] = df.iloc[:, 2:15]/df['w'].iloc[0]
            df.iloc[:, 15:28] = df.iloc[:, 15:28]/df['h'].iloc[0]
            if n_aug > 0:
                xs, ys = augment_position(df.iloc[:, 2:15], df.iloc[:, 15:28], n_aug)

                _class = df['action']
                X_train = []
                y_train = []
                for n in range(xs.shape[0]):
                    for i in range(nframe, df.shape[0]):
                        curr = np.append(xs[n,i-nframe:i+1,:], ys[n,i-nframe:i+1,:], axis=1)
                        curr = np.append(curr, df.iloc[i-nframe:i+1, 28:41], axis=1)
                        X_train.append(curr[:-1])
                        y_train.append(curr[-1])
            else: 
                X_train = []
                y_train = []
                for i in range(nframe, df.shape[0]):
                    X_train.append(np.array(df.iloc[i-nframe:i, 2:41]))
                    y_train.append(np.array(df.iloc[i, 2:41]))
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            train_input = np.append(train_input, X_train, axis=0)
            train_label = np.append(train_label, y_train, axis=0)
        train_input = train_input[1:, :, :]
        train_label = train_label[1:,:]

        if save:
            train_input.dump(save_path+"nf_input_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
            train_label.dump(save_path+"nf_label_{}-nframe_{}-augmented_{}-skip".format(nframe, n_aug, skip_frame))
            
        return train_input, train_label


def serialize_data_encoding(data, skip_frame=1, save=False, save_path=None):
    try:
        train_input = np.load(save_path+"en_input", allow_pickle=True)
        train_label = np.load(save_path+"en_label", allow_pickle=True)
        print('Data exists, loading...')
        return train_input, train_label
    except:
        print('Creating data...')
        train_input = -np.ones(shape=(1, 13*3))
        train_label = -np.ones(shape=(1, 13*3))
        for ID in data['file_id'].unique():
            df = data[data['file_id'] == ID].sort_values(['frame']).reset_index(drop=True)

            df = df[::skip_frame]
            df.iloc[:, 2:15] = df.iloc[:, 2:15]/df['w'].iloc[0]
            df.iloc[:, 15:28] = df.iloc[:, 15:28]/df['h'].iloc[0]
      
            train_input = np.append(train_input, np.array(df.iloc[:, 2:41]), axis=0)
            train_label = np.append(train_label, np.array(df.iloc[:, 2:41]), axis=0)
        train_input = train_input[1:,:]
        train_label = train_label[1:,:]

        if save:
            train_input.dump(save_path+"en_input")
            train_label.dump(save_path+"en_label")
            
        return train_input, train_label