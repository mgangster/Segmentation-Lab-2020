

import os
import glob
import numpy as np
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
from tensorflow.contrib.keras.python import keras
from scipy import misc

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def show(im, x=5, y=5):
    plt.figure(figsize=(x,y))
    plt.imshow(im)
    plt.show()
    
def show_images(maybe_ims, x=4, y=4):
    if isinstance(maybe_ims, (list, tuple)):
        border = np.ones((maybe_ims[0].shape[0], 10, 3))
        border = border.astype(np.uint8)
        new_im = maybe_ims[0]
        for i in maybe_ims[1:]:
            new_im = np.concatenate((new_im, border, i), axis=1)
        show(new_im, len(maybe_ims)*x, y)
    else:
        show(maybe_ims)

# helpers for loading a few images from the grading data 
def get_im_files(path, subset_name):
    return sorted(glob.glob(os.path.join(path, subset_name, 'images', '*.jpeg')))
                  
def get_mask_files(path, subset_name):
    return sorted(glob.glob(os.path.join(path, subset_name, 'masks', '*.png')))

def get_pred_files(subset_name):
    return sorted(glob.glob(os.path.join('..','data', 'runs', subset_name, '*.png')))

def get_im_file_sample(pred_run_name, subset_name, grading_data_dir_name=None,  n_file_names=10):
    if grading_data_dir_name:
        path = os.path.join('..', 'data', grading_data_dir_name)
    else:
        path = os.path.join('..', 'data')
    
    ims = np.array(get_im_files(path, subset_name)) 
    masks = np.array(get_mask_files(path, subset_name))  
    
    shuffed_inds = np.random.permutation(np.arange(masks.shape[0]))

    ims_subset = ims[shuffed_inds[:n_file_names]]
    masks_subset = masks[shuffed_inds[:n_file_names]]
    return list(zip(ims_subset, masks_subset))
    
def load_images(file_tuple):
    im = misc.imread(file_tuple[0])
    mask = misc.imread(file_tuple[1])
    if len(file_tuple) == 2:
        return im, mask
    else:
        pred = misc.imread(file_tuple[2])
        if pred.shape[0] != im.shape[0]:
            mask = misc.imresize(mask, pred.shape)
            im = misc.imresize(im, pred.shape)
        return im, mask, pred


def plot_keras_model(model, fig_name):
    base_path = os.path.join('..', 'data', 'figures')
    make_dir_if_not_exist(base_path)
    keras.utils.vis_utils.plot_model(model, os.path.join(base_path, fig_name))
    keras.utils.vis_utils.plot_model(model, os.path.join(base_path, fig_name +'_with_shapes'), show_shapes=True)


def train_val_curve(train_loss, val_loss=None):
    train_line = plt.plot(train_loss, label='train_loss')
    train_patch = mpatches.Patch(color='blue',label='train_loss')
    handles = [train_patch]
    if val_loss:
        val_line = plt.plot(val_loss, label='val_loss')
        val_patch = mpatches.Patch(color='orange',label='val_loss') 
        handles.append(val_patch)
        
    plt.legend(handles=handles, loc=2)
    plt.title('training curves') 
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()

# https://github.com/fchollet/keras/blob/master/keras/callbacks.py
class LoggerPlotter(keras.callbacks.Callback):
    """Callback that accumulates epoch averages of metrics.
    and plots train and validation curves on end of epoch
    """
    def __init__(self):
        self.hist_dict = {'loss':[], 'val_loss':[]}
        
    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size
        

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen
            
            self.hist_dict['loss'].append(logs['loss'])
            if 'val_loss' in self.params['metrics']:
                self.hist_dict['val_loss'].append(logs['val_loss'])
                train_val_curve(self.hist_dict['loss'], self.hist_dict['val_loss'])
            else:
                train_val_curve(self.hist_dict['loss'])
