from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import prince
import pandas as pd
from sklearn.cluster import KMeans

def PCA(mat, p):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    pca = prince.PCA(n_components=20)
    pca= pca.fit(pd.DataFrame(mat))
    reduction_dim = pca.transform(pd.DataFrame(mat))
    reduction_dim = reduction_dim.to_numpy()
    
    return reduction_dim

def stat_model(n_expr, mat, k, labels):
  means_nmi_score=0.
  varience_nmi_score=0.
  means_ari_score=0.
  varience_ari_score=0.

  results_nmi_score=[]
  results_ari_score=[]
  for i in range(n_expr):
    pred = clust(mat, k)
    results_nmi_score = normalized_mutual_info_score(pred,labels)
    results_ari_score = adjusted_rand_score(pred,labels)
  means_nmi_score = np.mean(results_nmi_score)
  means_ari_score = np.mean(results_ari_score)
  varience_nmi_score = np.std(results_nmi_score)
  varience_ari_score = np.std(results_ari_score)

  return means_nmi_score, varience_nmi_score, means_ari_score, varience_ari_score

def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        red_mat = PCA(mat, p)
        
    elif method=='AFC':
        red_mat = mat[:,:p]
        
    elif method=='UMAP':
        red_mat = mat[:,:p]
        
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")
    
    return red_mat


def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    result = model.fit(mat)

    return result.labels_

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'AFC', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')

