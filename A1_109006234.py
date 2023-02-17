import scipy.linalg.blas as blas
import scipy.linalg.lapack as lapack
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time   

result = [] # evaluate Ax=b in double precision
sizes = [10, 100, 1000, 2000, 3000, 4000, 5000]
for n in sizes:
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    start = time.time()
    lu, piv, x, info = lapack.dgesv(a, b) #lu, piv and info variables are not used
    end = time.time()
    runtime = end-start
    del lu, piv, info #delete unused variable to save memory

    ax = np.dot(a, x)
    if(np.allclose(ax, b)==False): #check correctness
        print('Errors are discovered at n=', n)
    result.append(runtime)
    
#drawing plots
plt.plot(sizes, result)
plt.title("Execution Time for Solving Ax=b")
plt.ylabel("Execution time (seconds)")
plt.xlabel("Matrix size")
plt.show