from collections.abc import Iterable
from itertools import combinations,product,permutations
from random import choice,shuffle,seed
import numpy as np
from numpy.linalg import matrix_rank as rank
import time

from functools import lru_cache
from typing import Set, List, Tuple

@lru_cache(maxsize=256)
def _factorial(n:int)->int:
    return 1 if n <= 1 else n*_factorial(n-1)

_factorial_cache = [1,1,2,6]
def _factorial(n:int)->int:
    while len(_factorial_cache) <= n:
        _factorial_cache.append(len(_factorial_cache)*_factorial_cache[-1])
    return _factorial_cache[n]


def factorial(n:int,k:int=None)->int:
    if k is not None:
        return _factorial(n)//_factorial(n-k)
    else:
        return _factorial(n)

@lru_cache(maxsize=512)
def binomial(n:int,*k:List[int])->int:
    res = _factorial(n)
    for i in k:
        res //= _factorial(i)
    if n: res //= factorial(n-sum(k))
    return res

def cnt_nzeros(n:int):
    cnt = 0
    while n:
        cnt += 1
        n ^= n&(-n)
    return cnt

def to_nary(n,q):
    if n == 0: return ''
    elif n<q: return str(n)
    return to_nary(n//q,q)+str(n%q)

def cnt_binomial_mod(n,q): # count the number of nonzeros in Cn,k for all k, mod q
    if n < q: return n+1
    else:
        return (1+(n%q))*cnt_binomial_mod(n//q,q)

def combination_with_order(iterable:Iterable,r:int)->Iterable:
    '''
    Generate all combinations of r elements from iterable with order.
    '''
    for comb in combinations(iterable,r):
        for comb_with_order in permutations(comb):
            yield comb_with_order



if __name__ == '__main__':
    assert factorial(5,3) == 60
    assert binomial(5,3) == 10
    assert binomial(10,2,3,4) == 45*56*5
    for n in range(100):
        cnt = 0
        for k in range(n+1):
            if binomial(n,k)%2:
                cnt += 1
        assert cnt == 1<<cnt_nzeros(n)
    q = 5
    d = 60
    for n in range(d):
        cnt = 0
        #print(f'{to_nary(n,q):>5s}'+' '*(d-n),end='')
        for k in range(n+1):
            c = binomial(n,k)%q
            #print(f'{binomial(n,k)%q:^2d}',end='')
            if c: cnt += 1
        assert cnt == cnt_binomial_mod(n,q)
        #print()
        #print(cnt,to_nary(cnt,q))
    for i in combination_with_order(range(3),2):
        print(i)
