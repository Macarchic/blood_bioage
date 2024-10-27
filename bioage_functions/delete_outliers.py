import numpy as np
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




