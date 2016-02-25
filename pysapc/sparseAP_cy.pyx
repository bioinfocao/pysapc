"""
cython helper for Sparse Affinity Propagation Clustering (SAP)
Use C code to speed up computation for SAP
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange

#DTYPEfloat = np.float
DTYPEint = np.int
#ctypedef np.float_t DTYPEfloat_t
ctypedef np.int_t DTYPEint_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef updateR_maxRow(double[::1] arr,DTYPEint_t[::1] sp):
    """
    Given a numpy array(arr) and split index(sp[0,...,len(arr)]),
    Return a array that have max value of each split, except the position that have max value will have second max value
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t s=sp.shape[0]
    cdef double m,sm,val
    cdef DTYPEint_t i,startInd,endInd,mi,j,k
    cdef double[::1] max_row=np.empty((n), dtype=np.float64)
    for i in range(1,s):
        startInd=sp[i-1]
        endInd=sp[i]
        if arr[startInd]>=arr[startInd+1]:
            m=arr[startInd]
            mi=startInd
            sm=arr[startInd+1]
        else:
            m=arr[startInd+1]
            mi=startInd+1
            sm=arr[startInd]
        for j in range(startInd+2,endInd):
            val=arr[j]
            if val>sm:
                if val>m:
                    sm=m
                    m=val
                    mi=j
                else:
                    sm=val
        for k in range(startInd,endInd):
            max_row[k]=m
        max_row[mi]=sm
    return np.asarray(max_row)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef updateR_maxRow_para(double[::1] arr,DTYPEint_t[::1] sp):
    """
    Given a numpy array(arr) and split index(sp[0,...,len(arr)]),
    Return a array that have max value of each split, except the position that have max value will have second max value
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t s=sp.shape[0]
    cdef double m,sm,val
    cdef DTYPEint_t i,startInd,endInd,mi,j,k
    cdef double[::1] max_row=np.empty((n), dtype=np.float64)
    with nogil, parallel():
        for i in prange(1,s):
            startInd=sp[i-1]
            endInd=sp[i]
            if arr[startInd]>=arr[startInd+1]:
                m=arr[startInd]
                mi=startInd
                sm=arr[startInd+1]
            else:
                m=arr[startInd+1]
                mi=startInd+1
                sm=arr[startInd]
            for j in range(startInd+2,endInd):
                val=arr[j]
                if val>sm:
                    if val>m:
                        sm=m
                        m=val
                        mi=j
                    else:
                        sm=val
            for k in range(startInd,endInd):
                max_row[k]=m
            max_row[mi]=sm
    return np.asarray(max_row)

cdef inline double arr_slice_sum(double[::1] arr, DTYPEint_t start, DTYPEint_t end) nogil:
    cdef:
        DTYPEint_t i
        double s=arr[start]
    for i in range(start+1,end):
        s += arr[i]
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef updateA_col(double[::1] r_col_data, DTYPEint_t[::1] sp, DTYPEint_t[::1] kk_index_col):
    """
    Given a numpy array(r_col_data), split index(sp[0,...,len(arr)]), position of column based self index
    """
    cdef DTYPEint_t n=r_col_data.shape[0]
    cdef DTYPEint_t s=sp.shape[0]
    cdef double[::1] a_col_data=np.empty((n), dtype=np.float64)
    cdef DTYPEint_t i,j,k,kk_ind, startInd,endInd
    cdef double aVal,col_sum
    for i in range(1,s):
        startInd=sp[i-1]
        endInd=sp[i]
        col_sum=r_col_data[startInd]
        for j in range(startInd+1,endInd):
            col_sum+=r_col_data[j]
        for k in range(startInd,endInd):
            aVal=col_sum-r_col_data[k]
            if aVal>0.0:
                a_col_data[k]=0.0
            else:
                a_col_data[k]=aVal
        kk_ind=kk_index_col[i-1]
        a_col_data[kk_ind]=col_sum-r_col_data[kk_ind]
    return np.asarray(a_col_data)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef updateA_col_para(double[::1] r_col_data, DTYPEint_t[::1] sp, DTYPEint_t[::1] kk_index_col):
    """
    Given a numpy array(r_col_data), split index(sp[0,...,len(arr)]), position of column based self index
    """
    cdef DTYPEint_t n=r_col_data.shape[0]
    cdef DTYPEint_t s=sp.shape[0]
    cdef double[::1] a_col_data=np.empty((n), dtype=np.float64)
    cdef double[::1] col_sum_arr=np.empty((s), dtype=np.float64)
    cdef DTYPEint_t i,j,k,kk_ind, startInd,endInd
    cdef double aVal
    with nogil, parallel():
        for i in prange(1,s):
            startInd=sp[i-1]
            endInd=sp[i]
            col_sum_arr[i-1]=arr_slice_sum(r_col_data,startInd,endInd)
            for k in range(startInd,endInd):
                aVal=col_sum_arr[i-1]-r_col_data[k]
                if aVal>0.0:
                    a_col_data[k]=0.0
                else:
                    a_col_data[k]=aVal
            kk_ind=kk_index_col[i-1]
            a_col_data[kk_ind]=col_sum_arr[i-1]-r_col_data[kk_ind]
    return np.asarray(a_col_data)

@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef rowMaxARIndex(double[::1] AR_arr,DTYPEint_t[::1] sp):
    """
    Given a numpy array(AR_arr) and split array(sp), find index of max value in each block
    """
    cdef DTYPEint_t n=AR_arr.shape[0]
    cdef DTYPEint_t s=sp.shape[0]
    cdef DTYPEint_t i,startInd,endInd,mi,j
    cdef double m,val
    cdef DTYPEint_t[::1] max_ar_index=np.empty((s-1), dtype=DTYPEint)
    for i in range(1,s):
        startInd=sp[i-1]
        endInd=sp[i]
        m=AR_arr[startInd]
        mi=startInd
        for j in range(startInd+1,endInd):
            val=AR_arr[j]
            if val>m:
                m=val
                mi=j
        max_ar_index[i-1]=mi
    return np.asarray(max_ar_index)
    
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef npArrRearrange_float(double[::1] arr,DTYPEint_t[::1] ind):
    """
    Rearrange a numpy array(arr) based on index array(ind)
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef double[::1] new_arr=np.empty((n), dtype=np.float64)
    for i in range(n):
        new_arr[i]=arr[ind[i]]
    return np.asarray(new_arr)
    
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef npArrRearrange_float_para(double[::1] arr,DTYPEint_t[::1] ind):
    """
    Parallel version, Rearrange a numpy array(arr) based on index array(ind)
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef double[::1] new_arr=np.empty((n), dtype=np.float64)
    with nogil, parallel():
        for i in prange(n):
            new_arr[i]=arr[ind[i]]
    return np.asarray(new_arr)
    
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef npArrRearrange_int(DTYPEint_t[::1] arr,DTYPEint_t[::1] ind):
    """
    Rearrange a numpy array(arr) based on index array(ind)
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef DTYPEint_t[::1] new_arr=np.empty((n), dtype=DTYPEint)
    for i in range(n):
        new_arr[i]=arr[ind[i]]
    return np.asarray(new_arr)
    
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef npArrRearrange_int_para(DTYPEint_t[::1] arr,DTYPEint_t[::1] ind):
    """
    Parallel version, Rearrange a numpy array(arr) based on index array(ind)
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef DTYPEint_t[::1] new_arr=np.empty((n), dtype=DTYPEint)
    with nogil, parallel():
        for i in prange(n):
            new_arr[i]=arr[ind[i]]
    return np.asarray(new_arr)
    
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef npArrRmNeg(double[::1] arr):
    """
    Given a numpy array(arr), set item in array less than 0 to 0
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef double[::1] new_arr=np.empty((n), dtype=np.float64)
    for i in range(n):
        if arr[i]>0.0:
            new_arr[i]=arr[i]
        else:
            new_arr[i]=0.0
    return np.asarray(new_arr)
    
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef npArrRmNeg_para(double[::1] arr):
    """
    Given a numpy array(arr), set item in array less than 0 to 0
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef double[::1] new_arr=np.empty((n), dtype=np.float64)
    with nogil, parallel():
        for i in prange(n):
            if arr[i]>0.0:
                new_arr[i]=arr[i]
            else:
                new_arr[i]=0.0
    return np.asarray(new_arr)

@cython.boundscheck(False)    
@cython.wraparound(False)     
cpdef getKKIndex(DTYPEint_t[::1] row,DTYPEint_t[::1] col):
    """
    Given two int numpy array(row,col), find index that row[i]==col[i]
    """
    cdef DTYPEint_t n=row.shape[0]
    cdef DTYPEint_t i
    cdef DTYPEint_t k=0
    for i in range(n):
        if row[i]==col[i]:
            k+=1
    cdef DTYPEint_t[::1] kkInds=np.empty((k), dtype=DTYPEint)
    k=0
    cdef DTYPEint_t j
    for j in range(n):
        if row[j]==col[j]:
            kkInds[k]=j
            k+=1
    return np.asarray(kkInds)
    
@cython.boundscheck(False)
@cython.wraparound(False)     
cpdef getIndptr(DTYPEint_t[::1] arr):
    """
    Given an sorted int numpy array(arr), find indptr
    """
    cdef DTYPEint_t n=arr.shape[0]
    cdef DTYPEint_t i
    cdef DTYPEint_t k=0
    for i in range(1,n):
        if arr[i]>arr[i-1]:
            k+=1
    cdef DTYPEint_t[::1] indptr=np.empty((k+1), dtype=DTYPEint)
    cdef DTYPEint_t j
    indptr[0]=0
    k=1
    for j in range(1,n):
        if arr[j]>arr[j-1]:
            indptr[k]=j
            k+=1
    return np.asarray(indptr)

@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
cpdef arrSamePercent(DTYPEint_t[::1] ref, DTYPEint_t[::1] target):
    """
    Given two int numpy array, calculate how much percent are they same
    """
    cdef DTYPEint_t n=ref.shape[0]
    cdef DTYPEint_t m=target.shape[0]
    if n==0 or m==0: return 0.0
    cdef DTYPEint_t k=0
    cdef DTYPEint_t i
    for i in range(n):
        if target[i]==ref[i]:
            k+=1
    return float(k)/n
    
@cython.boundscheck(False)
@cython.wraparound(False)     
cpdef singleItems(DTYPEint_t[::1] arr):
    """
    Given an int numpy array(arr), find items that only present one time in the array
    """
    cdef DTYPEint_t[::1] arr_sorted=np.sort(arr)
    cdef DTYPEint_t[::1] arr_diff=np.diff(arr_sorted)
    cdef DTYPEint_t n=arr_diff.shape[0]
    cdef DTYPEint_t i
    cdef list singles=[]
    for i in range(n-1):
        if arr_diff[i]>0:
            if arr_diff[i+1]>0:
                singles.append(arr_sorted[i+1])
    if arr_diff[n-1]>0: singles.append(arr_sorted[n])
    return singles

@cython.boundscheck(False)
@cython.wraparound(False)     
cpdef removeSingleSamples(DTYPEint_t[::1] row_array,DTYPEint_t[::1] col_array,double[::1] data_array, set singleInd_set):
    """
    Given int numpy array row_array, col_array and float array data_array, remove items in three array that row==col==singleInd
    """
    cdef DTYPEint_t n=row_array.shape[0]
    cdef DTYPEint_t singles_len=len(singleInd_set)
    cdef DTYPEint_t m=n-singles_len
    cdef DTYPEint_t[::1] row_array_left=np.empty((m), dtype=DTYPEint)
    cdef DTYPEint_t[::1] col_array_left=np.empty((m), dtype=DTYPEint)
    cdef double[::1] data_array_left=np.empty((m), dtype=np.float64)
    cdef DTYPEint_t k=0
    cdef DTYPEint_t i
    for i in range(n):
        ind=row_array[i]
        if (ind!=col_array[i]) or (ind not in singleInd_set):
            row_array_left[k]=ind
            col_array_left[k]=col_array[i]
            data_array_left[k]=data_array[i]
            k+=1
    return np.asarray(row_array_left),np.asarray(col_array_left),np.asarray(data_array_left)
    
@cython.boundscheck(False)
@cython.wraparound(False)     
cpdef copySingleRows(DTYPEint_t[::1] row_array,DTYPEint_t[::1] col_array,double[::1] data_array, set singleRow_set):
    """
    Given int numpy array row_array, col_array and float array data_array, copy row,col,data that col=singleRow!=row
    """
    cdef DTYPEint_t n=row_array.shape[0]
    cdef list copy_row=[]
    cdef list copy_col=[]
    cdef list copy_data=[]
    cdef DTYPEint_t i
    for i in range(n):
        colInd=col_array[i]
        if colInd in singleRow_set and row_array[i] != colInd:
            copy_row.append(row_array[i])
            copy_col.append(col_array[i])
            copy_data.append(data_array[i])
    return np.asarray(copy_row),np.asarray(copy_col),np.asarray(copy_data)
    
@cython.boundscheck(False)
@cython.wraparound(False)     
cpdef setDiag(DTYPEint_t[::1] row_array,DTYPEint_t[::1] col_array,double[::1] data_array, double[::1] preference_array):
    """
    Given int numpy array row_array, col_array and float array data_array of a sparse matrix, set diagnoal to preference_array
    """
    cdef DTYPEint_t n=len(row_array)
    cdef list diagIndInArr=[]
    cdef DTYPEint_t i
    for i in range(n):
        if row_array[i]==col_array[i]:
            diagIndInArr.append(row_array[i])
            data_array[i]=preference_array[row_array[i]]
    
    cdef set diagIndInArr_set=set(diagIndInArr)       
    cdef DTYPEint_t m=len(preference_array)-len(diagIndInArr_set)
    cdef DTYPEint_t[::1] diagNotInArr_row=np.empty((m), dtype=DTYPEint)
    cdef DTYPEint_t[::1] diagNotInArr_col=np.empty((m), dtype=DTYPEint)
    cdef double[::1] diagNotInArr_data=np.empty((m), dtype=np.float64)
    cdef DTYPEint_t j
    cdef DTYPEint_t k=0
    for j in range(len(preference_array)):
        if j not in diagIndInArr_set:
            diagNotInArr_row[k]=j
            diagNotInArr_col[k]=j
            diagNotInArr_data[k]=preference_array[j]
            k+=1
    row_array=np.concatenate((row_array,diagNotInArr_row))
    col_array=np.concatenate((col_array,diagNotInArr_col))
    data_array=np.concatenate((data_array,diagNotInArr_data))
    return np.asarray(row_array),np.asarray(col_array),np.asarray(data_array)
            
