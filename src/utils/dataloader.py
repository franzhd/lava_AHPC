from typing import Any
import torch
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import os
import numpy as np

class WisdmEncodedDatasetParser():
    def __init__(self, file_name):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        
        self.train_dataset = (x_train, y_train)
        self.val_dataset = (x_val,y_val)
        self.test_dataset = (x_test,y_test)

        
        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None):
        if subset:
            N = self.train_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None):
        if subset:
            N = self.val_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])
        return self.val_dataset

    def get_test_set(self, subset=None):
        if subset:
            N = self.test_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])
        return self.test_dataset
    
    def load_wisdm2_data(self,file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['x_train'], data['x_val'], data['x_test'], data['y_train'], data['y_val'], data['y_test'])


class WisdmDatasetParser():
    def __init__(self, file_name, norm="std", class_sublset = None, subset_list = None):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        self.class_sublset = class_sublset
        self.norm = norm
        self.mean = np.mean(x_train, axis=(0,1))
        self.std = np.std(x_train, axis=(0,1))
        print(self.mean.shape)
        print(self.std.shape)
        if self.norm == "std":
            x_train = x_train - self.mean
            x_train = x_train/self.std

            x_val = x_val - self.mean
            x_val = x_val/self.std

            x_test = x_test - self.mean
            x_test = x_test/self.std
        
        elif self.norm == "custom":
            x_train = x_train/self.std
            x_val = x_val/self.std
            x_test = x_test/self.std
        
        elif self.norm == None:
            pass
        
        print(f'ytrain shape {y_train.shape}')
        print(f'yval shape {y_val.shape}')
        print(f'ytest shape {y_test.shape}')
        
        x_train = np.transpose(x_train,axes=(0,2,1))
        x_val = np.transpose(x_val,axes=(0,2,1))
        x_test = np.transpose(x_test,axes=(0,2,1))
        
        if self.class_sublset is not None:
            if self.class_sublset == '7BC':
                selected_classes =  [1,6,7,8,13,14,17]
            elif self.class_sublset == '7WC':
                selected_classes = [11,12,13,14,15,16,17]
            elif self.class_sublset == 'subset_2':
                selected_classes = [6, 7, 8, 9, 10, 11, 12]
            elif self.class_sublset == 'custom':
                selected_classes = subset_list
            
            x_train, y_train = filter_dataset(x_train, y_train, selected_classes)
            x_val, y_val = filter_dataset(x_val, y_val, selected_classes)
            x_test, y_test = filter_dataset(x_test, y_test, selected_classes)

        if len(y_test.shape) > 1:
            self.train_dataset = (x_train, np.argmax(y_train, axis=-1))
            self.val_dataset = (x_val, np.argmax(y_val, axis=-1))
            self.test_dataset = (x_test, np.argmax(y_test, axis=-1))
        else:
            self.train_dataset = (x_train, y_train)
            self.val_dataset = (x_val,y_val)
            self.test_dataset = (x_test,y_test)
        
        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None):
        if subset:
            N = self.train_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None):
        if subset:
            N = self.val_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])
        return self.val_dataset

    def get_test_set(self, subset=None):
        if subset:
            N = self.test_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:subset]
            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])
        return self.test_dataset
    
    def de_std(self, data):
        if self.norm == "norm":
            data= data * self.std
            data= data + self.mean
        if self.norm == "custom":
            data= data * self.std

    def do_std(self, data):
        data= data - self.mean
        data= data / self.std
        
    @staticmethod
    def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])

def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])


def filter_dataset(x_train, y_train, selected_classes):
    # Create a mapping dictionary for the selected classes
    class_mapping = {original: new for new, original in enumerate(selected_classes)}
    
    # Convert selected_classes to a set for faster look-up
    selected_set = set(selected_classes)
    
    # Get the indices of the selected classes in y_train
    original_class_indices = np.argmax(y_train, axis=1)
    mask = np.isin(original_class_indices, selected_classes)
    
    # Filter the data and labels using the mask
    filtered_x = x_train[mask]
    filtered_y = y_train[mask]
    
    # Map the original class indices to new indices
    new_class_indices = np.vectorize(class_mapping.get)(original_class_indices[mask])
    
    # Create the new one-hot encoded labels
    new_one_hot_y = np.zeros((filtered_y.shape[0], len(selected_classes)))
    new_one_hot_y[np.arange(filtered_y.shape[0]), new_class_indices] = 1
    
    return filtered_x, new_one_hot_y

class WisdmDataset(Dataset):
     
    def __init__(self, data, transform=None, augument=None):
        xs, y = data
        self.x = []
        self.y = y 
        self.augument = augument
        self.transform = transform 
        if self.transform is not None:
            print("transforming array ....")
            for x in xs:
                tmp = self.transform(x)
                self.x.append(tmp)
            print(f' lenght of transformed array {len(self.x)}')
            self.x = np.array(self.x)

        else:
            self.x = xs

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augument:
            x = self.augument(x)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.y.size


# class To_spike(object): #discretization
#     def __init__(self, num_levels):
#         self.num_levels = num_levels
#         self.thresholds = np.linspace(-1, 1, num_levels)

#     def __call__(self, sample):
        
#         digitized = np.digitize(sample, self.thresholds) - 1
#         out = np.eye(self.num_levels)[digitized]
#         return out.reshape(sample.shape[0]*self.num_levels, sample.shape[-1])


# class SendOnDelta(object):
#     def __init__(self, tresholds): #treshold needs to be a value (0,1)
#         # sourcery skip: raise-specific-error
        
#         for treshold in tresholds:
#             if(treshold<0 or treshold>1):
#                 raise Exception("Thresholds out of range permitted") 
        
#         self.tresholds = tresholds
#     def __call__(self, sample):
#         # sourcery skip: raise-specific-error
#         if len(self.tresholds) == 1:
#             self.tresholds = self.tresholds * sample.shape[0]
#         if(len(self.tresholds)!=sample.shape[0]):
#             raise Exception(f"tresholds len {len(self.tresholds)} not compatible with {sample.shape[0]}")
        
#         new_sample = np.zeros((sample.shape[0], 2, sample.shape[1]))
#         for i in range(sample.shape[0]):
#             t = 0
#             t_ref = 0
#             while t<sample.shape[1]:
#                 if sample[i,t] - sample[i,t_ref] >= self.tresholds[i]:
#                     new_sample[i, 0, t] = 1
#                     t_ref = t
#                 elif sample[i,t] - sample[i,t_ref] <= - self.tresholds[i]:
#                     new_sample[i, 1, t] = 1
#                     t_ref = t
#                 t += 1
#         return new_sample

# class LIF(object):
#     def __init__(self, dt, V_th, V_reset, tau, g_L, V_init, E_L, tref):
        
#         self.V_th = V_th
#         self.V_reset = V_reset
#         self.g_L = g_L
#         self.V_init = V_init
#         self.E_L = E_L

#         self.tau = tau
#         self.dt = dt
#         self.tref = tref
    
#     def __call__(self,sample):
        
#         dim = sample.shape[0]
#         if len(self.V_th) == 1:
#             self.V_th = self.V_th * dim

#         if len(self.V_reset) == 1:
#             self.V_reset = self.V_reset * dim  

#         if len(self.g_L) == 1:
#            self. g_L = self.g_L * dim

#         if len(self.V_init) == 1:
#             self.V_init = self.V_init * dim

#         if len(self.E_L) == 1:
#            self.E_L = self.E_L * dim

#         if(len(self.V_th) != dim):
#             raise Exception(f"parameters len {self.dim} not compatible with {sample.shape[0]}")
        
#         new_sample = np.zeros((sample.shape[0], sample.shape[1]))

#         for i in range(dim):
            
#             v = np.zeros((sample.shape[1]))
#             v[0] = self.V_init[i]
#             tr = 0.

#             for t in range(sample.shape[1]-1):
                
#                 if tr > 0:
#                     v[t] = self.V_reset[i]
#                     tr = tr - 1
                
#                 elif v[t] >= self.V_th[i]:
#                     new_sample[i][t] = 1
#                     v[t] = self.V_reset[i]
#                     tr = self.tref / self.dt
                
#                 dv = (-(v[t] - self.E_L[i]) + sample[i][t] / self.g_L[i]) * (self.dt / self.tau)
#                 v[t + 1] = v[t] + dv

#         return new_sampleget_validation_set