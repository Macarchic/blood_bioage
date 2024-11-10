import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.covariance import MinCovDet

from sklearn.ensemble import IsolationForest

from scipy.spatial.distance import pdist, squareform

from scipy.signal import savgol_filter

from sklearn.manifold import TSNE

def euclidean_deletion(X,components):
    pca = PCA()
    T = pca.fit_transform(StandardScaler().fit_transform(X))
    euclidean = np.zeros(X.shape[0])

    for i in range(components):
        euclidean += (T[:,i] - np.mean(T[:,:components]))**2/np.var(T[:,:components])

    threshold = np.percentile(euclidean, 95)
    outliers = np.where(euclidean > threshold)[0]
    return outliers


def mahalanobis_deletion(X,components): 
    pca = PCA()
    T = pca.fit_transform(StandardScaler().fit_transform(X))
    robust_cov = MinCovDet().fit(T[:,:components])
    m = robust_cov.mahalanobis(T[:,:components])
    threshold = np.percentile(m, 95)
    outliers = np.where(m > threshold)[0]
    return outliers

def isolation_forest_deletion(X):
    isolation_forest = IsolationForest(contamination='auto')
    isolation_forest.fit(X)
    anomaly_labels = isolation_forest.predict(X)
    outliers = np.where(anomaly_labels == -1)[0]
    return outliers
    
def manhatan_deletion(X,components):
    pca = PCA()
    T = pca.fit_transform(StandardScaler().fit_transform(X))
    manhattan_distances = pdist(T[:, :components], metric='cityblock') 
    manhattan_matrix = squareform(manhattan_distances)

    m_manhattan = np.max(manhattan_matrix, axis=1)
    threshold_manhattan = np.percentile(m_manhattan, 95)
    outliers = np.where(m_manhattan > threshold_manhattan)[0]
    return outliers

def zscore_deletion(X):
    median_X = np.median(X, axis=0)
    mad_X = np.median(np.abs(X - median_X), axis=0) * 1.4826

    # Calculate robust Z-scores
    z_scores = (X - median_X) / mad_X
    outlier_indices = np.where(np.abs(z_scores) > 2)

    # Count outliers in each sample
    outlier_counts = np.bincount(outlier_indices[0], minlength=X.shape[0])
    outliers = np.where(outlier_counts > 0.5 * X.shape[1])[0]
    return outliers

def find_component(X):
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
    components = np.argmax(cumulative_variance >= 95) + 1 
    return components

def snv(X):
    data = np.asarray(X)
    for idx in range(data.shape[0]):
        data[idx] = (data[idx]-data[idx].mean())/data[idx].std()
    return data

def preprocess_data(X, window_length=43,polyorder=3):
    sav_X = pd.DataFrame(savgol_filter(X,window_length=window_length,polyorder=polyorder))
    X = pd.DataFrame(snv(sav_X))
    return X

def plot_tsne(X,outliers=None,preprocessed=False):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Set plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))

    # Plot normal points
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color='royalblue', alpha=0.6, s=50, label='Normal Points', edgecolor='w')

    # Highlight outliers in red and label them
    if outliers is not None:
        plt.scatter(X_tsne[outliers, 0], X_tsne[outliers, 1], color='red', s=100, label='Outliers', edgecolor='black')
        for outlier in outliers:
            plt.text(X_tsne[outlier, 0], X_tsne[outlier, 1], str(outlier), color='black', fontsize=6, ha='center', va='center')
    if preprocessed:
        X_title = 'preprocessed data'
    else:
        X_title = 'raw data'
    plot_title = f't-SNE Visualization for {X_title}'
    plt.title(plot_title, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(frameon=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()