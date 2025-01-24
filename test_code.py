from joblib import Parallel, delayed
import ray
import time
import math
import numpy as np

""" def pond_uniforme(k):
    return 1

def func(i, j, n, pond):
    d = {}
    d['premier'] = i
    d['second'] = j
    d['puissance'] = n
    if(j<i):
        d['sum'] = 0
    else:
        sum = 0
        for k in range(i,j+1):
            sum += pond(k)*k**n
        d['sum'] = sum
    return d

n_jobs=8
n=2
start_time = time.process_time()
res = Parallel(n_jobs=n_jobs)(delayed(func)(i, j,n,pond_uniforme) for i in range(1000) for j in range(i,1000))
end_time = time.process_time()

print(res)
print("len: ", len(res))
print("temps: ", end_time-start_time) """

# Start Ray.


@ray.remote
def f(i,j,n,data):
    dico={}
    dico['i'] = i; dico['j'] = j
    if(i>j):
        dico['sum'] = 0
    else:
        sum=0
        for k in range(i,j+1):
            sum += k**n
        dico['sum'] = sum*data[10000]
    return dico

ray.init()
# Start 4 tasks in parallel.
data = np.ones(1000000)
n=2
data_shared = ray.put(data)
result_ids = [f.remote(i,j,n,data_shared) for i in range(100) for j in range(i,100)]
    
results = ray.get(result_ids)
print(results)
