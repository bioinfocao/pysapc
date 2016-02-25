"""
Prepare Sparse Matrix for Sparse Affinity Propagation Clustering (SAP)
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import sparseAP_cy # cython for calculation

############################################################################################
# 

def copySym(rowBased_row_array,rowBased_col_array,rowBased_data_array,singleRowInds):
    """
    For single col items or single row items, copy sym minimal value
    For example if for sample 'A', the only datapoint of [s(A,A),s(A,B),s(A,C)...] is s(A,B),
    then we copy the minimal value of [s(A,A),s(C,A),s(D,A)...] (except s(B,A), because if we copy s(B,A), for 'A' we still only have one data point)
    """
    copy_row_array,copy_col_array,copy_data_array=sparseAP_cy.copySingleRows(rowBased_row_array,rowBased_col_array,rowBased_data_array,singleRowInds)
    #if symCopy=='all':
        #rowBased_row_array=np.concatenate((rowBased_row_array,copy_col_array))
        #rowBased_col_array=np.concatenate((rowBased_col_array,copy_row_array))
        #rowBased_data_array=np.concatenate((rowBased_data_array,copy_data_array))
    #else:# symCopy=='min' or others will be treated as 'min'
    df = pd.DataFrame(zip(copy_row_array,copy_col_array,copy_data_array), columns=['row', 'col', 'data'])
    copy_row_list,copy_col_list,copy_data_list=[],[],[]
    for ind in singleRowInds:
        copyData=df[(df.col==ind) & (df.row!=ind)].sort(['data']).copy()
        copyData_min=copyData[0:1]
        copy_row_list+=list(copyData_min.col)
        copy_col_list+=list(copyData_min.row)
        copy_data_list+=list(copyData_min.data)
    rowBased_row_array=np.concatenate((rowBased_row_array,copy_row_list))
    rowBased_col_array=np.concatenate((rowBased_col_array,copy_col_list))
    rowBased_data_array=np.concatenate((rowBased_data_array,copy_data_list))
    return rowBased_row_array,rowBased_col_array,rowBased_data_array

def rmSingleSamples(rowBased_row_array,rowBased_col_array,rowBased_data_array,nSamplesOri):
    """
    Affinity/similarity matrix does not need be symmetric, that is s(A,B) does not need be same as s(B,A).
    Also since Affinity/similarity matrix is sparse, it could be that s(A,B) exist but s(B,A) does not exist in the sparse matrix.
    For the FSAPC to work, specifically in computation of R and A matrix, each row/column of Affinity/similarity matrix should have at least two datapoints.
    So in FSAPC, we first remove samples that do not have affinity/similarity with other samples, that is samples that only have affinity/similarity with itself
    And we remove samples only have one symmetric datapoint, for example for sample 'B' only s(B,C) exist and for sample 'C' only s(C,B) exist
    In these two cases, these samples are removed from FSAPC computation and their examplers are set to themself.
    For samples that only have one data (affinity/similarity) with others, For example if for sample 'A', the only datapoint of [s(A,A),s(A,B),s(A,C)...] is s(A,B),
    and there exist at least one value in [s(A,A),s(C,A),s(D,A)...] (except s(B,A), because if we copy s(B,A), for 'A' we still only have one data point)
    then we copy the minimal value of [s(A,A),s(C,A),s(D,A)...] 
    nSamplesOri is the number of samples of orignail input data
    """ 
    # find rows and cols that only have one datapoint
    singleRowInds=set(sparseAP_cy.singleItems(rowBased_row_array))
    singleColInds=set(sparseAP_cy.singleItems(rowBased_col_array))
    # samples that have one datapoint in row and col are samples only have affinity/similarity with itself
    singleSampleInds=singleRowInds & singleColInds
    
    # in case every col/row have more than one datapoint, just return original data
    if len(singleRowInds)==0 and len(singleColInds)==0:
        return rowBased_row_array,rowBased_col_array,rowBased_data_array,None,None,nSamplesOri
    
    # remove samples that only have affinity/similarity with itself
    # or only have one symmetric datapoint, for example for sample 'B' only s(B,C) exist and for sample 'C' only s(C,B) exist
    # in these two cases, these samples are removed from FSAPC computation and their examplers are set to themself.
    if len(singleSampleInds)>0:
        # row indexs that left after remove single samples
        rowLeft=sorted(list(set(range(nSamplesOri))-singleSampleInds)) 
        # map of original row index to current row index(after remove rows/cols that only have single item)
        rowOriLeftDict={ori:left for left,ori in enumerate(rowLeft)} 
        rowLeftOriDict={left:ori for ori,left in rowOriLeftDict.items()}
        rowBased_row_array,rowBased_col_array,rowBased_data_array=sparseAP_cy.removeSingleSamples(rowBased_row_array,rowBased_col_array,rowBased_data_array,singleSampleInds)
    else: # no samples are removed
        rowLeftOriDict=None
    #if len(singleSampleInds)>0:
        #rowBased_row_array,rowBased_col_array,rowBased_data_array=sparseAP_cy.removeSingleSamples(rowBased_row_array,rowBased_col_array,rowBased_data_array,singleSampleInds)
    
    # for samples that need copy a minimal value to have at least two datapoints in row/column
    # for samples that row have single data point, copy minimal value of this sample's column
    singleRowInds=singleRowInds-singleSampleInds
    if len(singleRowInds)>0:
        rowBased_row_array,rowBased_col_array,rowBased_data_array=copySym(rowBased_row_array.astype(np.int),rowBased_col_array.astype(np.int),rowBased_data_array,singleRowInds)
    # for samples that col have single data point, copy minimal value of this sample's row
    singleColInds=singleColInds-singleSampleInds
    if len(singleColInds)>0:
        rowBased_col_array,rowBased_row_array,rowBased_data_array=copySym(rowBased_col_array.astype(np.int),rowBased_row_array.astype(np.int),rowBased_data_array,singleColInds)
    
    # change row, col index if there is any sample removed
    if len(singleSampleInds)>0:
        changeIndV=np.vectorize(lambda x:rowOriLeftDict[x])
        rowBased_row_array=changeIndV(rowBased_row_array)
        rowBased_col_array=changeIndV(rowBased_col_array) 
    
    #rearrange based on new row index and new col index, print ('{0}, sort by row,col'.format(datetime.now()))
    sortedLeftOriInd = np.lexsort((rowBased_col_array,rowBased_row_array)).astype(np.int)
    rowBased_row_array=sparseAP_cy.npArrRearrange_int_para(rowBased_row_array.astype(np.int),sortedLeftOriInd)
    rowBased_col_array=sparseAP_cy.npArrRearrange_int_para(rowBased_col_array.astype(np.int),sortedLeftOriInd)
    rowBased_data_array=sparseAP_cy.npArrRearrange_float_para(rowBased_data_array,sortedLeftOriInd)
    
    return rowBased_row_array,rowBased_col_array,rowBased_data_array,rowLeftOriDict,singleSampleInds,nSamplesOri-len(singleSampleInds)

def preCompute(rowBased_row_array,rowBased_col_array,S_rowBased_data_array):
    """
    format affinity/similarity matrix
    """
    
    # Get parameters
    data_len=len(S_rowBased_data_array)
    row_indptr=sparseAP_cy.getIndptr(rowBased_row_array)
    if row_indptr[-1]!=data_len: row_indptr=np.concatenate((row_indptr,np.array([data_len])))
    row_to_col_ind_arr=np.lexsort((rowBased_row_array,rowBased_col_array))
    colBased_row_array=sparseAP_cy.npArrRearrange_int_para(rowBased_row_array,row_to_col_ind_arr)
    colBased_col_array=sparseAP_cy.npArrRearrange_int_para(rowBased_col_array,row_to_col_ind_arr)
    col_to_row_ind_arr=np.lexsort((colBased_col_array,colBased_row_array))
    col_indptr=sparseAP_cy.getIndptr(colBased_col_array)
    if col_indptr[-1]!=data_len: col_indptr=np.concatenate((col_indptr,np.array([data_len])))
    kk_col_index=sparseAP_cy.getKKIndex(colBased_row_array,colBased_col_array)
    
    #Initialize matrix A, R
    A_rowbased_data_array=np.array([0.0]*data_len)
    R_rowbased_data_array=np.array([0.0]*data_len)
    
    #Add random samll value to remove degeneracies
    random_state=np.random.RandomState(0)
    S_rowBased_data_array+=1e-12*random_state.randn(data_len)*(np.amax(S_rowBased_data_array)-np.amin(S_rowBased_data_array))
    
    #Convert row_to_col_ind_arr/col_to_row_ind_arr data type to np.int datatype so it is compatible with cython code
    row_to_col_ind_arr=row_to_col_ind_arr.astype(np.int)
    col_to_row_ind_arr=col_to_row_ind_arr.astype(np.int)
    
    return S_rowBased_data_array, A_rowbased_data_array, R_rowbased_data_array,col_indptr,row_indptr,row_to_col_ind_arr,col_to_row_ind_arr,kk_col_index


