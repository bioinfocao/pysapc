"""
Test Sparse Affinity Propagation (SAP)
Compare its result with SKlearn Affinity Propagation (AP) Clustering result
Test SAP with sparse affinity/similarity matrix
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
import os
from scipy.sparse import coo_matrix,csr_matrix,lil_matrix
from datetime import datetime
from sklearn.cluster import AffinityPropagation
#from SparseAPCluster import SAP
from pysapc import SAP
from pysapc import sparseAP_cy # cython for calculation speed optimization

##################################################################################

def loadMatrix(data_file, dataCutoff=None):
    """
    Load similarity data file
    if dataCutoff is not None, all value (affinity/similarity) below this will be discarded
    """
    #print('{0}, loading data'.format(datetime.now()))
    simi=pd.DataFrame.from_csv(data_file,sep='\t',index_col=None)
    samples=sorted(list(set(simi.row) | set(simi.col)))
    samplesInd={el:ind for ind,el in enumerate(samples)}
    row,col,data=simi.row.map(lambda x:samplesInd[x]),simi.col.map(lambda x:samplesInd[x]),simi.data
    if dataCutoff is not None:
        row_new,col_new,data_new=[],[],[]
        for r,c,d in zip(row,col,data):
            if d>dataCutoff:
                row_new.append(r)
                col_new.append(c)
                data_new.append(d)
        simi_mat=coo_matrix((data_new,(row_new,col_new)), shape=(len(samplesInd),len(samplesInd)))
    else:
        simi_mat=coo_matrix((data,(row,col)), shape=(len(samplesInd),len(samplesInd)))
    return simi_mat

def clusterSimilarityWithSklearnAPC(data_file,damping=0.9,max_iter=200,convergence_iter=15,preference='min'):
    """
    Compare Sparse Affinity Propagation (SAP) result with SKlearn Affinity Propagation (AP) Clustering result.
    Please note that convergence condition for Sklearn AP is "no change in the number of estimated clusters",
    for SAP the condition is "no change in the cluster assignment". 
    So SAP may take more iterations and the there will be slightly difference in final cluster assignment (exemplars for each sample).
    """
    # loading data
    simi_mat=loadMatrix(data_file)
    simi_mat_dense=simi_mat.todense()

    # get preference
    if preference=='min':
        preference=np.min(simi_mat_dense)
    elif preference=='median':
        preference=np.median(simi_mat_dense)
    
    print('{0}, start SKlearn Affinity Propagation'.format(datetime.now()))
    af=AffinityPropagation(damping=damping, preference=preference, affinity='precomputed',verbose=True)
    af.fit(simi_mat_dense)
    cluster_centers_indices,labels = af.cluster_centers_indices_,af.labels_
    sk_exemplars=np.asarray([cluster_centers_indices[i] for i in labels])
    print('{0}, start Fast Sparse Affinity Propagation Cluster'.format(datetime.now()))
    sap=SAP(preference=preference,convergence_iter=convergence_iter,max_iter=max_iter,damping=damping,verboseIter=100)
    sap_exemplars=sap.fit_predict(simi_mat_dense)
    
    # Caculate similarity between sk_exemplars and sap_exemplars
    exemplars_similarity=sparseAP_cy.arrSamePercent(np.array(sk_exemplars), np.array(sap_exemplars))
    
    return exemplars_similarity


def clusterSimilarityWithDenseMatrix(data_file,cutoff,damping=0.9,max_iter=200,convergence_iter=15,preference='min'):
    """
    Test SAP with sparse affinity/similarity matrix
    """
    # loading data
    simi_mat_dense=loadMatrix(data_file,dataCutoff=None).todense()
    simi_mat_sparse=loadMatrix(data_file,dataCutoff=cutoff)

    # get preference
    if preference=='min':
        preference=np.min(simi_mat_dense)
    elif preference=='median':
        preference=np.median(simi_mat_dense)

    print('{0}, start Sparse Affinity Propagation with dense matrix'.format(datetime.now()))
    sap_dense=SAP(preference=preference,convergence_iter=convergence_iter,max_iter=max_iter,damping=damping,verboseIter=100)
    sap_dense_exemplars=sap_dense.fit_predict(simi_mat_dense)
    
    print('{0}, start Sparse Affinity Propagation with sparse matrix'.format(datetime.now()))
    sap_sparse=SAP(preference=preference,convergence_iter=convergence_iter,max_iter=max_iter,damping=damping,verboseIter=100)
    sap_sparse_exemplars=sap_sparse.fit_predict(simi_mat_sparse)    
    
    # Caculate similarity between sap_dense_exemplars and sap_sparse_exemplars
    exemplars_similarity=sparseAP_cy.arrSamePercent(np.array(sap_dense_exemplars), np.array(sap_sparse_exemplars))
    
    return exemplars_similarity

def testDense():
    """
    Test dense similarity matrix, Compare FSAPC result with SKlearn Affinity Propagation (AP) Clustering result
    """
    dense_similarity_matrix_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),'FaceClusteringSimilarities.txt')
    exemplars_similarity=clusterSimilarityWithSklearnAPC(data_file=dense_similarity_matrix_file,damping=0.9,max_iter=200,convergence_iter=15,preference='min')
    print("Exemplar label similarity between sklearn.cluster.AffinityPropagation and SAP is: {0}".format(exemplars_similarity))
    
def testSparse():
    """
    test sparse similarity matrix
    """
    dense_similarity_matrix_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),'FaceClusteringSimilarities.txt')
    #cutoff=-23.5 # 10% of all data left with this cutoff -> 0.78 similarity
    #cutoff=-27.7 # 20% of all data left with this cutoff -> 1.0 similarity
    for cutoff in [-23.5, -27.7]:
        exemplars_similarity=clusterSimilarityWithDenseMatrix(data_file=dense_similarity_matrix_file,cutoff=cutoff,damping=0.9,max_iter=500,convergence_iter=15,preference='min')
        print("Exemplar label similarity between dense and sparse (data above: {0}) SAP is: {1}".format(cutoff,exemplars_similarity))    
    
