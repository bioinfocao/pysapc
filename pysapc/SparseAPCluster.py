"""
Sparse Affinity Propagation (SAP)
Designed for large data set using scipy sparse matrix(affinity/similarity matrix)
Speed optimized with cython
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import numpy as np
from datetime import datetime
from scipy.sparse import coo_matrix,csr_matrix,lil_matrix
import sparseAP_cy # cython for calculation speed optimization
import sparseMatrixPrepare

#########################################################################

def matixToRowColDataArr(X):
    """
    Convert sparse affinity/similarity matrix to numpy array format (row_array,col_array,data_array)
    So cython update function can work efficiently on it.
    """
    # convert to coo format (from lil,csr,csc)
    if isinstance(X, coo_matrix):
        X_coo=X
    elif (isinstance(X, csr_matrix)) or (isinstance(X, lil_matrix)):
        X_coo=X.tocoo()
    else: # others like numpy matrix could be convert to coo matrix
        X_coo=coo_matrix(X)
    # Upcast matrix to a floating point format (if necessary)
    X_coo=X_coo.asfptype() 
    # get row_array,col_array,data_array in their correct data type (for cython to work)
    row_array,col_array,data_array=X_coo.row.astype(np.int),X_coo.col.astype(np.int),X_coo.data
    
    return row_array,col_array,data_array

def updateR_cython(S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array, row_indptr, rowBased_row_array, rowBased_col_array, damping):
    """
    Update Responsibilities Matrix (R)
    """
    as_data=A_rowbased_data_array+S_rowBased_data_array
    as_max_data_arr=sparseAP_cy.updateR_maxRow(as_data,row_indptr)
    r_new_data_arr=S_rowBased_data_array-as_max_data_arr
    r_row_data=(r_new_data_arr*(1.0-damping)) + (R_rowbased_data_array*damping)
    return r_row_data

def updateA_cython(A_rowbased_data_array, R_rowbased_data_array, col_indptr, row_to_col_ind_arr,col_to_row_ind_arr, kk_col_index, damping):
    """
    Update Availabilities Matrix (A)
    """
    A_colbased_data_array=sparseAP_cy.npArrRearrange_float(A_rowbased_data_array,row_to_col_ind_arr)
    R_colbased_data_array=sparseAP_cy.npArrRearrange_float(R_rowbased_data_array,row_to_col_ind_arr)
    r_col_data=np.copy(R_colbased_data_array)
    r_col_data[r_col_data<0]=0
    r_col_data[kk_col_index]=R_colbased_data_array[kk_col_index]
    a_col_data_new=sparseAP_cy.updateA_col(r_col_data,col_indptr,kk_col_index)
    a_col_data=(a_col_data_new*(1.0-damping)) + (A_colbased_data_array*damping)
    A_rowbased_data_array=sparseAP_cy.npArrRearrange_float(a_col_data,col_to_row_ind_arr)
    return A_rowbased_data_array

def updateR_cython_para(S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array, row_indptr, rowBased_row_array, rowBased_col_array, damping):
    """
    Update Responsibilities Matrix (R), with cython multiprocessing.
    """
    as_data=A_rowbased_data_array+S_rowBased_data_array
    as_max_data_arr=sparseAP_cy.updateR_maxRow_para(as_data,row_indptr)
    r_new_data_arr=S_rowBased_data_array-as_max_data_arr
    r_row_data=(r_new_data_arr*(1.0-damping)) + (R_rowbased_data_array*damping)
    return r_row_data


def updateA_cython_para(A_rowbased_data_array, R_rowbased_data_array, col_indptr, row_to_col_ind_arr,col_to_row_ind_arr, kk_col_index, damping):
    """
    Update Availabilities Matrix (A), with cython multiprocessing.
    """
    A_colbased_data_array=sparseAP_cy.npArrRearrange_float_para(A_rowbased_data_array,row_to_col_ind_arr)
    R_colbased_data_array=sparseAP_cy.npArrRearrange_float_para(R_rowbased_data_array,row_to_col_ind_arr)
    r_col_data=np.copy(R_colbased_data_array)
    r_col_data[r_col_data<0]=0
    r_col_data[kk_col_index]=R_colbased_data_array[kk_col_index]
    a_col_data_new=sparseAP_cy.updateA_col_para(r_col_data,col_indptr,kk_col_index)
    a_col_data=(a_col_data_new*(1.0-damping)) + (A_colbased_data_array*damping)
    A_rowbased_data_array=sparseAP_cy.npArrRearrange_float_para(a_col_data,col_to_row_ind_arr)
    return A_rowbased_data_array

def getPreferenceList(preference,nSamplesOri,data_array):
    """
    Input preference should be a numeric scalar, or a string of 'min' / 'median', or a list/np 1D array(length of samples).
    Return preference list(same length as samples)
    """
    # numeric value
    if isinstance(preference, float) or isinstance(preference, int) or isinstance(preference, long):
        preference_list=[float(preference)]*nSamplesOri
    # str/unicode min/mean
    elif isinstance(preference, basestring):
        if str(preference)=='min':
            preference=data_array.min()
        elif str(preference)=='median':
            preference=np.median(data_array)
        else: #other string
            raise ValueError("Preference should be a numeric scalar, or a string of 'min' / 'median',\
            or a list/np 1D array(length of samples).\n Your input preference is: {0})".format(str(prefernce)))
        preference_list=[preference]*nSamplesOri
    # list or numpy array
    elif (isinstance(preference, list) or isinstance(preference, np.ndarray)) and len(preference)==nSamplesOri: 
        preference_list=preference
    else:
        raise ValueError("Preference should be a numeric scalar, or a str of 'min' / 'median',\
        or a list/np 1D array(length of samples).\n Your input preference is: {0})".format(str(prefernce)))
    return preference_list


def sparseAffinityPropagation(row_array,col_array,data_array,\
        preference='min',convergence_iter=15,convergence_percentage=0.999999,max_iter=200,damping=0.9,verboseIter=100, parallel=True):
    """
    Sparse Affinity Propagation (SAP) clustering function
    This function can be called directly if row_array,col_array,data_array available.
    If called directly, there should be no duplicate datapoints(means (row_array[i],col_array[i]) should be unique for i in range(0,len(row_array)))

    Parameters
    ----------------------
    X: coo_matrix,csr_matrix,lil_matrix, precomputed sparse affinity/similarity matrix
        (affinity/similarity could be cosine, pearson, euclidean distance, or others).
        Please note that affinity/similarity matrix doesn't need to be symmetric, s(A,B) can be different from s(B,A).
        In fact it could be that s(A,B) exist and s(B,A) not exist in the sparse affinity/similarity matrix
    
    preference: a numeric scalar(float), or a str of 'min'/'median', or a list/numpy 1D array(length of samples)
        the preference of a datapoint K, p(K), which will set to the affinity/similarity matrix s(K,K), is the 
        priori suitability of datapoint K to serve as an exemplar (cluster center), Higher values of preference will lead to more exemplars (cluster centers).
        A good initial choice is minimum('min') or median('median') of the full dense affinity/similarity matrix.
        Plsease note that minimum('min') or median('median') of sparse affinity/similarity matrix is not recommended. 
    
    convergence_iter: int, optional, default: 15. Number of iterations with no change or change less than 1.0-convergence_percentage 
        in exemplars (cluster centers) label of datapoint.
    
    convergence_percentage: float, optional, default: 0.999999, 
        This parameter is used to define convergence condition. 
        If set as 0.999999, then one or less out of 1 million datapoints does not change their exemplars (cluster centers) will be considered as convergence.
        This parameter is added because pySAPC is designed to deal with large data.
    
    max_iter: int, optional, default: 2000
        Maximum number of iterations. Try to increase max_iter if FSAPC is not convergence yet at max_iter.
    
    damping: float, optional, default: 0.9.
        Damping factor should between 0.5 and 1.
    
    verboseIter: int/None, default: 100
        The level of verbose. if set to 0 or None, do not print verbose output; 
        If set to 1, print the status for each interation.
        If set to 100, for each 100 iterations, print current status
        
    parallel: boolean, default: True
        Turn on cython multiprocessing or not. It is recommended to set it True for speed up.
        
    Returns
    ----------------------
    The exemplars (cluster centers) for each datapoint. Exemplars are index(row index of matrix) of cluster centers for each datapoint.
    """
    if (verboseIter is not None) and (verboseIter >0): print('{0}, Starting Sparse Affinity Propagation'.format(datetime.now()))
    
    # Convert to numpy array if not
    if not isinstance(row_array,np.ndarray): row_array=np.asarray(row_array)
    if not isinstance(col_array,np.ndarray): col_array=np.asarray(col_array)
    if not isinstance(data_array,np.ndarray): data_array=np.asarray(data_array)
    # Make sure rowindex/colindex are int, data is float
    row_array,col_array,data_array=row_array.astype(np.int),col_array.astype(np.int),data_array.astype(np.float)
    
    # Get parameters (nSamplesOri, preference_list, damping)
    nSamplesOri=max((row_array.max(),col_array.max()))+1
    preference_list=getPreferenceList(preference,nSamplesOri,data_array)
    if damping < 0.5 or damping >= 1:raise ValueError('damping must be >= 0.5 and < 1')

    # set diag of affinity/similarity matrix to preference_list
    row_array,col_array,data_array=sparseAP_cy.setDiag(row_array,col_array,data_array,np.asarray(preference_list))
    
    # reOrder by rowbased
    sortedLeftOriInd = np.lexsort((col_array,row_array)).astype(np.int)
    if parallel:
        rowBased_row_array=sparseAP_cy.npArrRearrange_int_para(row_array.astype(np.int),sortedLeftOriInd)
        rowBased_col_array=sparseAP_cy.npArrRearrange_int_para(col_array.astype(np.int),sortedLeftOriInd)
        S_rowBased_data_array=sparseAP_cy.npArrRearrange_float_para(data_array,sortedLeftOriInd)
    else:
        rowBased_row_array=sparseAP_cy.npArrRearrange_int(row_array.astype(np.int),sortedLeftOriInd)
        rowBased_col_array=sparseAP_cy.npArrRearrange_int(col_array.astype(np.int),sortedLeftOriInd)
        S_rowBased_data_array=sparseAP_cy.npArrRearrange_float(data_array,sortedLeftOriInd)        

    # For the FSAPC to work, specifically in computation of R and A matrix, each row/column of Affinity/similarity matrix should have at least two datapoints.
    # Samples do not meet this condition are removed from computation (so their exemplars are themself) or copy a minimal value of corresponding column/row
    rowBased_row_array,rowBased_col_array,S_rowBased_data_array,rowLeftOriDict,singleSampleInds,nSamples=\
            sparseMatrixPrepare.rmSingleSamples(rowBased_row_array,rowBased_col_array,S_rowBased_data_array,nSamplesOri)
    
    # Initialize matrix A, R; Remove degeneracies in data;
    # Get col_indptr,row_indptr,row_to_col_ind_arr,col_to_row_ind_arr,kk_col_index
    S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array,col_indptr,row_indptr,row_to_col_ind_arr,col_to_row_ind_arr,kk_col_index=\
        sparseMatrixPrepare.preCompute(rowBased_row_array,rowBased_col_array,S_rowBased_data_array)
    
    # Iterate update R,A matrix until meet convergence condition or reach max iteration
    # In FSAPC, the convergence condition is when there is more than convergence_iter iteration in rows that have exact clustering result or 
    # have similar clustering result that similarity is great than convergence_percentage if convergence_percentage is set other than None
    # (default convergence_percentage is 0.999999, that is one datapoint in 1 million datapoints have different clustering.)
    # This condition is added to FSAPC is because FSAPC is designed to deal with large data set.
    lastLabels,labels=np.empty((0), dtype=np.int),np.empty((0), dtype=np.int)
    convergeCount=0
    for it in range(1,max_iter+1):
        lastLabels=labels
        if parallel:
            R_rowbased_data_array=updateR_cython_para(\
                S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array, row_indptr, rowBased_row_array, rowBased_col_array, damping)
            A_rowbased_data_array=updateA_cython_para(\
                A_rowbased_data_array, R_rowbased_data_array, col_indptr, row_to_col_ind_arr,col_to_row_ind_arr, kk_col_index, damping)
        else:
            R_rowbased_data_array=updateR_cython(\
                S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array, row_indptr, rowBased_row_array, rowBased_col_array, damping)
            A_rowbased_data_array=updateA_cython(\
                A_rowbased_data_array, R_rowbased_data_array, col_indptr, row_to_col_ind_arr,col_to_row_ind_arr, kk_col_index, damping)            
        AR_rowBased_data_array=A_rowbased_data_array+R_rowbased_data_array
        labels=sparseAP_cy.rowMaxARIndex(AR_rowBased_data_array,row_indptr).astype(np.int)
        
        # check convergence
        if convergence_percentage is None:
            if np.array_equal(lastLabels, labels) and len(labels)!=0:
                convergeCount+=1
            else:
                convergeCount=0
        else: 
            if sparseAP_cy.arrSamePercent(lastLabels, labels)>=convergence_percentage and len(labels)!=0:
                convergeCount+=1
            else:
                convergeCount=0
        if convergeCount==convergence_iter and it<max_iter:
            if (verboseIter is not None) and (verboseIter > 0):
                print('{0}, Converged after {1} iterations.'.format(datetime.now(),it))
            break
        elif it==max_iter:
            if (verboseIter is not None) and (verboseIter > 0):
                print('{0}, Max iterations:{1} reached. labels doesnot change for last {2} iterations.'.format(datetime.now(),it,convergeCount))
        else:
            if (verboseIter is not None) and (verboseIter > 0) and ((it-1)%verboseIter==0):
                print('{0}, {1} of {2} iterations, labels doesnot change for last {3} iterations.'.format(datetime.now(),it,max_iter,convergeCount))
            
    # Converting labels back to original sample index
    sampleLables=np.asarray(rowBased_col_array[labels])
    if singleSampleInds is None or len(singleSampleInds)==0:
        finalLabels=sampleLables
    else:
        finalLabels=[rowLeftOriDict[el] for el in sampleLables]
        for ind in sorted(singleSampleInds): # sorted singleSampleInds, insert samples that removed in rmSingleSamples()
            finalLabels.insert(ind,ind)
        finalLabels=np.asarray(finalLabels)
    return finalLabels


class SAP():
    """
    Sparse Affinity Propagation (SAP) for large data (sparse affinity/similarity matrix)
    
    To test installation, in python shell, run:
    from pysapc import tests
    tests.testDense()
    tests.testSparse()
    
    Quick Start:
    Use pysapc to cluster sparse similarity matrix (scipy sparse matrix):
    from pysapc import SAP
    sap=SAP(preference,convergence_iter=convergence_iter,max_iter=max_iter,damping=damping,verboseIter=100)
    sap_exemplars=sap.fit_predict(X) # X should be a scipy sparse similarity matrix

    
    Parameters
    ----------------------
    X: coo_matrix,csr_matrix,lil_matrix, precomputed sparse affinity/similarity matrix
        (affinity/similarity could be cosine, pearson, euclidean distance, or others).
        Please note that affinity/similarity matrix doesn't need to be symmetric, s(A,B) can be different from s(B,A).
        In fact it could be that s(A,B) exist and s(B,A) not exist in the sparse affinity/similarity matrix
    
    preference: a numeric scalar(float), or a str of 'min'/'median', or a list/numpy 1D array(length of samples)
        the preference of a datapoint K, p(K), which will set to the affinity/similarity matrix s(K,K), is the 
        priori suitability of datapoint K to serve as an exemplar (cluster center), Higher values of preference will lead to more exemplars (cluster centers).
        A good initial choice is minimum('min') or median('median') of the full dense affinity/similarity matrix.
        Plsease note that minimum('min') or median('median') of sparse affinity/similarity matrix is not recommended. 
    
    convergence_iter: int, optional, default: 15. Number of iterations with no change or change less than 1.0-convergence_percentage 
        in exemplars (cluster centers) label of datapoint.
    
    convergence_percentage: float, optional, default: 0.999999, 
        This parameter is used to define convergence condition. 
        If set as 0.999999, then one or less out of 1 million datapoints does not change their exemplars (cluster centers) will be considered as convergence.
        This parameter is added because pySAPC is designed to deal with large data.
    
    max_iter: int, optional, default: 2000
        Maximum number of iterations. Try to increase max_iter if FSAPC is not convergence yet at max_iter.
    
    damping: float, optional, default: 0.9.
        Damping factor should between 0.5 and 1.
    
    verboseIter: int/None, default: 100
        The level of verbose. if set to 0 or None, do not print verbose output; 
        If set to 1, print the status for each interation.
        If set to 100, for each 100 iterations, print current status
        
    parallel: boolean, default: True
        Turn on cython multiprocessing or not. It is recommended to set it True for speed up.

    Attributes
    ----------------
    exemplars_: the cluster centers for each datapoint
        The index(row index of matrix) of examplers(cluster centers) for each datapoint 

    Notes
    ---------------
    To prepare sparse matrix, either use a single cutoff for all samples (for example keep top 20 percent of full matrix) 
    or use different cutoff values for each samples so that each samples have K nearest neighbors. 
    Users are recommended to try several sparse matrix and compare their clustering result to determine 
    when the clustering result reach plateau (when including more data do not change clustering result significantly)
    
    References
    ----------------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages Between Data Points", Science Feb. 2007
    """
    
    def __init__(self, preference=None, convergence_iter=15, convergence_percentage=0.999999,\
             max_iter=2000, damping=0.9, verboseIter=100, parallel=True):
        self.preference=preference
        self.convergence_iter=convergence_iter
        self.convergence_percentage=convergence_percentage
        self.max_iter=max_iter
        self.damping=damping
        self.verboseIter=verboseIter
        self.parallel=parallel
        self.exemplars_=[]
        
    def denseToSparseAbvCutoff(self, denseMatrix, cutoff):
        """
        Remove datas in denseMatrix that is below cutoff, Convert the remaining datas into sparse matrix.
        Parameters:
        ----------------------
        denseMatrix: dense numpy matrix
        
        cutoff: int or float
        
        Returns
        ----------------------
        Scipy csr_matrix
        
        """
        maskArray=denseMatrix>=cutoff
        sparseMatrix=csr_matrix( (np.asarray(denseMatrix[maskArray]).reshape(-1),np.nonzero(maskArray)),\
                    shape=denseMatrix.shape)
        return sparseMatrix
    
    def denseToSparseTopPercentage(self, denseMatrix, percentage=10.0):
        """
        Keep top percentage (such as 10%) of data points, remove all others. Convert into sparse matrix.
        Parameters:
        ----------------------
        denseMatrix: dense numpy matrix
        
        percentage: float, default is 10.0
            percentage of top data points to keep. default is 10.0% that is for 10000 data points keep top 1000.
        
        Returns
        ----------------------
        Scipy csr_matrix
        
        """
        rowN,colN=denseMatrix.shape
        totalN=rowN*colN
        topN=min(int(totalN*(percentage/100.0)), totalN)
        arr=np.array(denseMatrix.flatten())[0]
        cutoff=arr[arr.argsort()[-(topN)]]
        sparseMatrix=self.denseToSparseAbvCutoff(denseMatrix,cutoff)
        return sparseMatrix  
    
    def fit(self, X, preference=None):
        """
        Apply Sparse Affinity Propagation (SAP) to precomputed sparse affinity/similarity matrix X
        
        Parameters
        ----------------------
        X: coo_matrix,csr_matrix,lil_matrix, precomputed sparse affinity/similarity matrix
           (affinity/similarity could be cosine, pearson, euclidean distance, or others).
           Please note that affinity/similarity matrix doesn't need to be symmetric, s(A,B) can be different from s(B,A).
           In fact it could be that s(A,B) exist and s(B,A) not exist in the sparse affinity/similarity matrix
           
        preference: a numeric scalar(float), or a str of 'min'/'median', or a list/numpy 1D array(length of samples)
            the preference of a datapoint K, p(K), which will set to the affinity/similarity matrix s(K,K), is the 
            priori suitability of datapoint K to serve as an exemplar (cluster center), Higher values of preference will lead to more exemplars (cluster centers).
            A good initial choice is minimum('min') or median('median') of the full dense affinity/similarity matrix.
            Plsease note that minimum('min') or median('median') of sparse affinity/similarity matrix,
            which is top of the full dense affinity/similarity matrix, is not a good choice.  
        
        Notes
        ----------------------
        After fitting, the clustering result (exemplars/ cluster centers) could be accessed by exemplars_ Attribute
        Or use fit_predict function, which will return a list of exemplars (row index of affinity/similarity matrix)
        """
        if (self.preference is None) and (preference is None):
            raise ValueError("Preference should be a numeric scalar, or a string of 'min' / 'median',\
            or a list/np 1D array(length of samples).\n Your input preference is: {0})".format(str(prefernce)))
        if preference is not None:
            preference_input=preference
        else:
            preference_input=self.preference
        row_array,col_array,data_array=matixToRowColDataArr(X)
        self.exemplars_=sparseAffinityPropagation(row_array,col_array,data_array,\
                            preference=preference_input,convergence_iter=self.convergence_iter,\
                            convergence_percentage=self.convergence_percentage,\
                            max_iter=self.max_iter,damping=self.damping,verboseIter=self.verboseIter,parallel=self.parallel)
        return self
    
    def fit_predict(self, X, preference=None):
        """
        Apply Sparse Affinity Propagation (SAP) to precomputed sparse affinity/similarity matrix X
        
        Parameters
        ----------------------
        X: coo_matrix,csr_matrix,lil_matrix, precomputed sparse affinity/similarity matrix
           (affinity/similarity could be cosine, pearson, euclidean distance, or others).
           Please note that affinity/similarity matrix doesn't need to be symmetric, s(A,B) can be different from s(B,A).
           In fact it could be that s(A,B) exist and s(B,A) not exist in the sparse affinity/similarity matrix

        preference: a numeric scalar(float), or a str of 'min'/'median', or a list/numpy 1D array(length of samples)
            the preference of a datapoint K, p(K), which will set to the affinity/similarity matrix s(K,K), is the 
            priori suitability of datapoint K to serve as an exemplar (cluster center), Higher values of preference will lead to more exemplars (cluster centers).
            A good initial choice is minimum('min') or median('median') of the full dense affinity/similarity matrix.
            Plsease note that minimum('min') or median('median') of sparse affinity/similarity matrix,
            which is top of the full dense affinity/similarity matrix, is not a good choice.       
        
        Returns
        ----------------------
        The exemplars (cluster centers) for each datapoint. Exemplars are index(row index of matrix) of cluster centers for each datapoint.
        """
        if (self.preference is None) and (preference is None):
            raise ValueError("Preference should be a numeric scalar, or a string of 'min' / 'median',\
            or a list/np 1D array(length of samples).\n Your input preference is: {0})".format(str(prefernce)))
        if preference is not None:
            preference_input=preference
        else:
            preference_input=self.preference
        row_array,col_array,data_array=matixToRowColDataArr(X)
        self.exemplars_=sparseAffinityPropagation(row_array,col_array,data_array,\
                            preference=self.preference,convergence_iter=self.convergence_iter,\
                            convergence_percentage=self.convergence_percentage,\
                            max_iter=self.max_iter,damping=self.damping,verboseIter=self.verboseIter,parallel=self.parallel)
        return self.exemplars_
