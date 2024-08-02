import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE: ", device)

def schulz_inverse_stable(A, damping_factor=0.001, max_iterations=20, tol=1e-6):
    n = A.shape[0]
    I = torch.eye(n, device=A.device)
    A = A + damping_factor * I
    X = torch.eye(n, device=A.device) * 0.0005

    for i in range(max_iterations):
        X = X @ (2 * I - A @ X)
    
    return X

#d = [512, 1024, 2048, 4096]
d = [8192]
#Ns = [200, 800, 6400, 12800]
Ns = [12800]
damping_factor = 0.01


exact_time = []
schulz_time = []

for _ in range(3):

    for dim in d:
        print("Dimension: ", dim)
        time_schulz = 0
        time_inv = 0
        
        tmp = torch.randn((dim, 1)).to(device)
        A = tmp @ tmp.T
        
        I = torch.eye(dim, device=A.device)
        A = A + damping_factor * I


        ## compute the condition number
        # cond_nums.append(torch.linalg.cond(A, p='fro').item())
        # print("Condition Number: ", cond_nums)

        st_time = time.time()
        A_inv = torch.inverse(A)
        time_inv = time.time() - st_time
        print("Inverse Time: ", time_inv)
        exact_time.append(time_inv)
        # A_inv = np.linalg.inv(A.cpu().numpy()) # true inverse

        st_time = time.time()
        A_inv_schulz = schulz_inverse_stable(A, damping_factor=0, max_iterations=20)
        time_schulz = time.time() - st_time
        print("Schulz Time: ", time_schulz)
        schulz_time.append(time_schulz)
    

## take the average of the times
print("Exact Time: ", np.mean(exact_time))
print("Schulz Time: ", np.mean(schulz_time))