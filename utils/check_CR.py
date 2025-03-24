import numpy as np
import time
from combtools import cnt_nzeros
from itertools import combinations

def binary_repr(array):
    return sum(1<<i for i,val in enumerate(array) if val)

def check_CR(Mat:np.array)->bool:
    h,w = Mat.shape
    if h>w:
        Mat = Mat.T
    binary_reprs = [binary_repr(i) for i in Mat]   
    for row_i,row_j in combinations(binary_reprs,2):
        if cnt_nzeros(row_i&row_j) > 1:
            return False
    return True

if __name__ == "__main__":
    assert binary_repr([0,1,1]) == 6
    assert binary_repr([1,0,0]) == 1
    a = binary_repr([0,0,1,1])
    b = binary_repr([1,0,1,0])
    assert cnt_nzeros(a) == 2
    assert cnt_nzeros(b) == 2
    assert cnt_nzeros(a&b) == 1
    matrix = np.array([[0,0,1],[0,1,0],[1,0,0]])
    assert check_CR(matrix)
    matrix = np.array([[0,1,1],[1,1,1]])
    assert not check_CR(matrix)
    matrix = np.array([[0,1,1],[1,1,1]]).T
    assert not check_CR(matrix)
    start=time.time()
    assert check_CR(np.identity(3000))
    print(f'Use {time.time()-start} second')

