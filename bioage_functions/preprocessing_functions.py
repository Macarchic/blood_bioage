import numpy as np
import pandas as pd
import copy
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.metrics import make_scorer, root_mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import scale
from chemotools.baseline import CubicSplineCorrection
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.model_selection import LeaveOneOut
from multiprocessing.pool import ThreadPool  # Замість ProcessPoolExecutor
from itertools import repeat
from sklearn.model_selection import train_test_split
import time
import threading
import aiofiles
import asyncio
import psutil



def arpls(X, lam=1e6, ratio=0.05, niter=5):
    """
    Asymmetric Recursive Polynomial Least Squares (ARPLS) for baseline correction.

    Parameters:
    y (array): The input signal (spectrum).
    lam (float): The smoothing parameter.
    ratio (float): The ratio for asymmetry.
    niter (int): The number of iterations.

    Returns:
    array: The corrected baseline.
    """
    corrected_absorptions = []
    absorptions = np.asarray(X)
    if absorptions.ndim == 1:
        absorptions = absorptions.reshape(1,-1)
        isonedim = True
    else:
        isonedim = False
    
    for object in absorptions:
        L = len(object)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L-2, L))
        H = lam * D.T @ D
        w = np.ones(L)
        for i in range(niter):
            W = sparse.diags(w, 0)
            Z = W + H
            z = splinalg.spsolve(Z, w * object)
            d = object - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            w = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        corrected_absorptions.append(object-np.asarray(z))
    corrected_absorptions = np.asarray(corrected_absorptions)
    if isonedim:
        corrected_absorptions = corrected_absorptions.reshape(-1)
    return corrected_absorptions


def read_dataset(path):
    BioAge_dataset = pd.read_csv(path)
    y = np.asarray(BioAge_dataset.iloc[-1,1:].tolist())
    wl = np.asarray(BioAge_dataset['wavelength'][BioAge_dataset['wavelength'].notna()].tolist())
    set = BioAge_dataset.drop('wavelength',axis=1)
    set = set.drop(BioAge_dataset.index[-1])
    X = set.to_numpy().T
    return wl,X,y,BioAge_dataset


def stratify_dataset(X,y,bins=5,random=42,test_size=0.1):
    np.random.seed(random)
    # Используем KBinsDiscretizer для создания бинов
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform', random_state=random)
    y_binned = est.fit_transform(y.reshape(-1, 1)).astype(int).reshape(-1)
    
    # Разделение данных с учетом стратификации
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y_binned, random_state=random)
    
    return X_train,y_train,X_test,y_test

def zscore(X,threshold):
    """
    Calculates the Z-scores for the input data and identifies outliers based on a specified threshold.

    Parameters:
    -----------
    X : array-like
        The input feature matrix (2D array), where each row represents a sample and each column represents a feature.
    
    threshold : float
        The Z-score threshold for identifying outliers. Data points with a Z-score greater than this threshold (in absolute value) will be considered outliers.

    Returns:
    --------
    outliers : tuple of arrays
        A tuple of arrays containing the indices of the samples (rows) and features (columns) where the Z-scores exceed the specified threshold. 
        This can be used to identify and potentially remove outliers in the dataset.

    Notes:
    ------
    - The function first calculates the Z-scores for each element in the feature matrix `X`.
    - The Z-score is calculated using the median and standard deviation along each feature (column).
    - The function then identifies the indices of elements where the absolute Z-score exceeds the specified threshold.

    Example:
    --------
    outlier_indices = zscore(X, threshold=2)
    
    # To access the rows and columns of the outliers:
    outlier_rows, outlier_columns = outlier_indices
    """
    zscore = (X - np.median(X,axis=0))/np.std(X,axis=0)
    return np.where(np.abs(zscore) > threshold)

def delete_outliers(X_input,y_input,threshold=2,summary = False):
    """
    Identifies and removes outlier samples from the dataset based on a Z-score threshold.

    Parameters:
    -----------
    X_input : array-like
        The input feature matrix (2D array), where each row represents a sample and each column represents a feature.
    
    y_input : array-like
        The target values associated with each sample in the input feature matrix.
    
    threshold : float, optional, default=2
        The Z-score threshold for identifying outliers. Samples with a Z-score above this threshold will be considered outliers.
    
    summary : bool, optional, default=False
        If True, a summary of the outlier detection and removal process will be printed, including the number of outlier points per sample and the indices of deleted   samples.
    
    Returns:
    --------
    new_X : ndarray
        The feature matrix after removing the identified outlier samples.
    
    new_y : ndarray
        The target values after removing the corresponding outlier samples.
    
    Notes:
    ------
    - The function first calculates the Z-scores for each sample in the feature matrix `X_input`.
    - Samples that have more than 75% of their points identified as outliers (based on the Z-score) are considered outlier samples and are removed.
    - If no outliers are detected, a message is printed to inform the user.
    - If `summary` is set to True, the function prints a detailed summary of the outlier removal process, including the number of outlier points for each sample and the total number of outlier samples removed.

    Example:
    --------
    X_cleaned, y_cleaned = delete_outliers(X, y, threshold=2, summary=True)
    
    """
    X = np.asarray(X_input)
    y = np.asarray(y_input)
    zscore_indices = zscore(X,threshold=threshold)
    count = []
    delete = []
    set_indices = set(zscore_indices[0])
    threshold = 0.75*X.shape[1]
    
    for idx in set(zscore_indices[0]):
        count.append(sum(zscore_indices[0]==idx))
            
    for idx,num in zip(set_indices,count):
        if num > threshold:
            delete.append(idx)
    new_X = np.delete(X,delete,axis=0)
    new_y = np.delete(y,delete,axis=0)

    if delete == []:
        print('There are no outliers!')
    elif summary == True:
        for idx,num in zip(set_indices,count):
            print(f'sample [{idx}]\t\033[42m{num}\033[0m\toutlier points')
        print(f'There were deleted {len(delete)} outlier samples: X{delete}')
    return new_X,new_y

def delete_outliers(X_input,y_input,threshold=3,summary = False):
    
    X = np.asarray(X_input)
    y = np.asarray(y_input)
    zscore_indices = np.where(np.abs(zscore(X)) > threshold)
    count = []
    delete = []
    set_indices = set(zscore_indices[0])
    threshold = 0.75*X.shape[1]
    
    for idx in set(zscore_indices[0]):
        count.append(sum(zscore_indices[0]==idx))
            
    for idx,num in zip(set_indices,count):
        if num > threshold:
            delete.append(idx)
    new_X = np.delete(X,delete,axis=0)
    new_y = np.delete(y,delete,axis=0)

    if delete == []:
        print('There are no outliers!')
    elif summary == True:
        for idx,num in zip(set_indices,count):
            print(f'sample [{idx}]\t\033[42m{num}\033[0m\toutlier points')
        print(f'There were deleted {len(delete)} outlier samples: X{delete}')
    return new_X,new_y

def kolya_read(path):
    data = pd.read_csv(path)
    wl = np.array(data.columns[:-2],dtype=float)
    Age = np.array(data.loc[:,'Age'].tolist())
    HbA1c = np.array(data.loc[:,'HbA1c'].tolist())
    data = data.drop(['Age','HbA1c'],axis=1)
    X = data.to_numpy()
    return wl,X,Age,HbA1c

def zscore(X):
    zscore = (X - np.median(X,axis=0))/np.std(X,axis=0)
    return zscore

def snv(X):
    data = np.asarray(X)
    for idx in range(data.shape[0]):
        data[idx] = (data[idx]-data[idx].mean())/data[idx].std()
    return data

def vector_normalization(X_input):
    X = np.asarray(X_input)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    new_X = X / norm
    return new_X

def clsp_baseline(X_input,indices=[3050,7100]):
    X = np.asarray(X_input)
    cspl = CubicSplineCorrection(indices=indices)
    new_X = np.asarray(cspl.fit_transform(X))
    return new_X
    
def msc(input_data):
    """
    Perform Multiplicative Scatter Correction
    """
    # Mean center the data
    mean_spectrum = np.mean(input_data, axis=0)
    
    # Initialize the corrected data matrix
    corrected_data = np.zeros_like(input_data)
    
    for i in range(input_data.shape[0]):
        # Fit a linear model to the reference spectrum
        y = input_data[i, :]
        X = mean_spectrum.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        
        # Apply the correction
        corrected_data[i, :] = (y - model.intercept_) / model.coef_
        
    return corrected_data




def preprocess1(wl_input,X_input,y_input):
    wl = np.asarray(wl_input)
    X = np.asarray(X_input)
    new_y = np.asarray(y_input)

    # cspl = CubicSplineCorrection(indices=[3050, 7100])
    # X = cspl.fit_transform(X)
    
    crop = (wl>800)&(wl<1800)
    X=X[:,crop]
    new_wl=wl[crop]
    
    X = savgol_filter(X,window_length=43,polyorder=3)
    
    # X,new_y = delete_outliers(X,y,summary=False)
    
    X = snv(X)
    new_X = zscore(X)
    
    return new_wl[1:-1],new_X[:,1:-1],new_y
