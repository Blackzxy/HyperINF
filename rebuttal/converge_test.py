import matplotlib.pyplot as plt
import torch
import numpy as np 



def schulz_inverse_stable(A, damping_factor=0.001, max_iterations=20, tol=1e-6):
    n = A.shape[0]
    I = torch.eye(n, device=A.device)
    A = A + damping_factor * I
    X = torch.eye(n, device=A.device) * 0.00005

    for i in range(max_iterations):
        X = X @ (2 * I - A @ X)
    
    return X


def neumann_series(A, max_iterations=20):
    n = A.shape[0]
    I = torch.eye(n, device=A.device)

    ## residual
    B = I - A

    ## initialize the residual
    A_inv = I
    term = I

    for i in range(max_iterations):
        term = B @ term
        A_inv += term
    
    return A_inv


d = [512, 1024, 2048, 4096]
Ns = [200, 800, 6400, 12800]
damping_factor = 0.01
device = 'cuda'
cnt=0
#ax, fig = plt.subplots(5, 4, figsize=(20, 25))
#ax, fig = plt.subplots(3, 4, figsize=(20, 15))
ax, fig = plt.subplots(2, 4, figsize=(20, 10))
        

for idx, N in enumerate(Ns):
    cond_nums = []
    cnt=0
    err_datainf = []
    for dim in d:
        print(cnt)
        err_schulz = []
        err_lissa = []
        err_neuman = []
        err_sor = []

        
        
        A = torch.zeros(size=(dim, dim)).to(device)
        datainf_approx = torch.zeros(size=(dim, dim)).to(device)
        I = torch.eye(dim, device=A.device)

        for _ in range(N):
            ## normal distribution
            tmp = torch.randn((dim, 1)).to(device)

            ## normal distribution in mean: 0, std: 5
            #tmp = torch.normal(mean=0, std=5, size=(dim, 1)).to(device)

            ## uniform distribution
            # tmp = torch.rand((dim, 1)).to(device)
            A += tmp @ tmp.T

            datainf_approx += I - tmp @ tmp.T / (damping_factor + tmp.T @ tmp)

            del tmp
        


        ## the maxtrix we want to invert
        A = A / N + damping_factor * I

        ## B matrix for neumann
        B = I - A

        datainf_approx = datainf_approx / (N * damping_factor)

        ## compute the condition number
        # cond_nums.append(torch.linalg.cond(A, p='fro').item())
        # print("Condition Number: ", cond_nums)


        A_inv = torch.inverse(A)
        

        vl = torch.randn((dim, 1), device=A.device)
        #vl = torch.normal(mean=0, std=5, size=(dim, 1)).to(device)
        #vl = torch.rand((dim, 1)).to(device)

        ## normalize the vl
        #vl = vl / torch.norm(vl, p=2)
        A_inv_vl = A_inv @ vl
        
        ## datainf err
        err_datainf.append(torch.norm(A_inv_vl - datainf_approx @ vl, p='fro').item())
        print("DataInf: ", err_datainf[-1])
        
        ## Schulz iteration
        X = torch.eye(dim, device=A.device) * 0.00005
        # X = torch.rand((dim, dim), device=A.device) * 0.00005
        # X = torch.randn((dim, dim), device=A.device) * 0.00005
        for i in range(40):
            X = X @ (2 * I - A @ X)
            err = torch.norm(A_inv_vl - X @ vl, p='fro').item()
            print("Schulz: ", err)
            err_schulz.append(err)
            del err

        ## lissa iteration
        A_norm = A / (torch.norm(A, p='fro')+1)
        r_l = vl
        for _ in range(40):
            r_l = vl + (I - A_norm) @ r_l
            #print(torch.tensor(A_inv_vl).shape, r_l.shape)
            err = torch.norm(A_inv_vl - r_l / (torch.norm(A, p='fro')+1), p='fro').item()
            print("LiSSA: ", err)
            err_lissa.append(err)
            del err
        
        # ## neumann series
        # neuman_inv = I
        # term = I
        # for _ in range(40):
        #     term = B @ term
        #     neuman_inv += term
        #     err = torch.norm(A_inv_vl - neuman_inv @ vl, p='fro').item()
        #     print("Neumann: ", err)
        #     err_neuman.append(err)
        #     del err
        

        # ## successive over relaxtion
        # n = A.shape[0]
        # D = torch.diag(torch.diag(A)).to(device)
        # L = torch.tril(A, diagonal=-1).to(device)
        # U = torch.triu(A, diagonal=1).to(device)
        # omega = 0.5

        # # precompute the D^-1
        # #D_inv = torch.inverse(D)
        # D_omega_L_inv = torch.inverse(D - omega * L)

        # # init sor_inv
        # sor_inv = torch.eye(n, device=A.device)
        # for k in range(40):
        #     #sor_inv = (1 - omega) * sor_inv + omega * D_inv @ (I - (L + U) @ sor_inv)
        #     sor_inv = D_omega_L_inv @ ((1 - omega) * D @ sor_inv + omega * U @ sor_inv) + omega * D_omega_L_inv @ I
        #     err = torch.norm(A_inv_vl - sor_inv @ vl, p='fro').item()
        #     print("SOR: ", err)
        #     err_sor.append(err)
        #     # print("SOR: ", min(err, 1e16))
        #     # err_sor.append(min(err, 1e16))
        #     del err




        

        
        cnt+=1
        if cnt==1:
            fig[0][idx].plot(err_schulz, marker='o', markersize=7, linestyle='-', color='r', label=f'dim={dim}')
            fig[1][idx].plot(err_lissa, marker='o', markersize=7, linestyle='-', color='r', label=f'dim={dim}')
            # fig[3][idx].plot(err_neuman, marker='o', markersize=7, linestyle='-', color='r', label=f'dim={dim}')
            # fig[4][idx].plot(err_sor, marker='o', markersize=7, linestyle='-', color='r', label=f'dim={dim}')
        elif cnt==2:
            fig[0][idx].plot(err_schulz, marker='o', markersize=7, linestyle='-', color='g', label=f'dim={dim}')
            fig[1][idx].plot(err_lissa, marker='o', markersize=7, linestyle='-', color='g', label=f'dim={dim}')
            # fig[3][idx].plot(err_neuman, marker='o', markersize=7, linestyle='-', color='g', label=f'dim={dim}')
            # fig[4][idx].plot(err_sor, marker='o', markersize=7, linestyle='-', color='g', label=f'dim={dim}')
            
        elif cnt==3:
            fig[0][idx].plot(err_schulz, marker='o', markersize=7, linestyle='-', color='b', label=f'dim={dim}')
            fig[1][idx].plot(err_lissa, marker='o', markersize=7, linestyle='-', color='b', label=f'dim={dim}')
            # fig[3][idx].plot(err_neuman, marker='o', markersize=7, linestyle='-', color='b', label=f'dim={dim}')
            # fig[4][idx].plot(err_sor, marker='o', markersize=7, linestyle='-', color='b', label=f'dim={dim}')
           
        elif cnt==4:
            fig[0][idx].plot(err_schulz, marker='o', markersize=7, linestyle='-', color='c', label=f'dim={dim}')
            fig[1][idx].plot(err_lissa, marker='o', markersize=7, linestyle='-', color='c', label=f'dim={dim}')
        #     fig[3][idx].plot(err_neuman, marker='o', markersize=7, linestyle='-', color='c', label=f'dim={dim}')
        #     fig[4][idx].plot(err_sor, marker='o', markersize=7, linestyle='-', color='c', label=f'dim={dim}')
            
        elif cnt==5:
            fig[0][idx].plot(err_schulz, marker='o', markersize=7, linestyle='-', color='m', label=f'dim={dim}')
            fig[1][idx].plot(err_lissa, marker='o', markersize=7, linestyle='-', color='m', label=f'dim={dim}')
        #     fig[3][idx].plot(err_neuman, marker='o', markersize=7, linestyle='-', color='m', label=f'dim={dim}')
        #     fig[4][idx].plot(err_sor, marker='o', markersize=7, linestyle='-', color='m', label=f'dim={dim}')
          
        
        
        ## set title
        #fig[idx].set_title(f'N={N}', fontsize=25)
        fig[0][idx].set_title(f'HyperINF N={N}', fontsize=25)
        fig[1][idx].set_title(f'LiSSA N={N}', fontsize=25)
        # fig[3][idx].set_title(f'Neumann N={N}', fontsize=25)
        # fig[4][idx].set_title(f'SOR N={N}', fontsize=25)
        ## scale the y-axis
        # fig[1][idx].set_yscale('log')
        # fig[3][idx].set_yscale('log')
        # fig[4][idx].set_yscale('log')
        #fig[idx].set_xlabel('Iteration', fontsize=25)
        fig[0][idx].set_xlabel('Iteration', fontsize=25)
        fig[1][idx].set_xlabel('Iteration', fontsize=25)
        # fig[3][idx].set_xlabel('Series Length', fontsize=25)
        # fig[4][idx].set_xlabel('Iteration', fontsize=25)
        ## set x-axis value's fontsize
        #fig[idx].tick_params(axis='x', labelsize=15)
        fig[0][idx].tick_params(axis='x', labelsize=15)
        fig[1][idx].tick_params(axis='x', labelsize=15)
        # fig[3][idx].tick_params(axis='x', labelsize=15)
        # fig[4][idx].tick_params(axis='x', labelsize=15)
        ## set y-axis label
        #fig[idx].set_ylabel('Error', fontsize=25)
        if idx == 0:
            fig[0][idx].set_ylabel('Error', fontsize=25)
            fig[1][idx].set_ylabel('Log of Error', fontsize=25)
            # fig[3][idx].set_ylabel('Log of Error', fontsize=25)
            # fig[4][idx].set_ylabel('Log of Error', fontsize=25)
        ## set y-axis value's fontsize
        #fig[idx].tick_params(axis='y', labelsize=15)
        fig[0][idx].tick_params(axis='y', labelsize=15)
        fig[1][idx].tick_params(axis='y', labelsize=15)
        # fig[3][idx].tick_params(axis='y', labelsize=15)
        # fig[4][idx].tick_params(axis='y', labelsize=15)
        ## set legend
        #fig[idx].legend(fontsize=20)
        if idx == 0:
            fig[0][idx].legend(fontsize=20)
            fig[1][idx].legend(fontsize=20)
            # fig[3][idx].legend(fontsize=20)
            # fig[4][idx].legend(fontsize=20)
        ## set grid
        #fig[idx].grid(True, linestyle='--', alpha=0.6)
        fig[0][idx].grid(True, linestyle='--', alpha=0.6)
        fig[1][idx].grid(True, linestyle='--', alpha=0.6)
        # fig[3][idx].grid(True, linestyle='--', alpha=0.6)
        # fig[4][idx].grid(True, linestyle='--', alpha=0.6)
        # ## make zoom-in plot
        # axins = zoomed_inset_axes(fig[idx], 6, loc=1)
        # axins.plot(err_n, marker='o', markersize=7, linestyle='-', color='r')
        # x1, x2, y1, y2 = 0, 2, 0, 0.1
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        # plt.xticks(visible=False)
        # plt.yticks(visible=False)
        # mark_inset(fig[idx], axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        del A, A_inv, A_inv_vl, vl, r_l, err_schulz, err_lissa

    # # print(cond_nums)
    # # fig[2][idx].plot(d, cond_nums, marker='o', markersize=7, linestyle='-', color='r')
    # fig[2][idx].plot(d, err_datainf, marker='o', markersize=7, linestyle='-', color='r')
    # # ## set title
    # # fig[2][idx].set_title(f'N={N}', fontsize=25)
    # fig[2][idx].set_title(f'DataInf N={N}', fontsize=25)
    # # ## set x-axis label
    # # fig[2][idx].set_xlabel('Dimension', fontsize=25)
    # fig[2][idx].set_xlabel('Dimension d', fontsize=25)
    # # ## set x-axis value's fontsize
    # # fig[2][idx].tick_params(axis='x', labelsize=15)
    # fig[2][idx].tick_params(axis='x', labelsize=15)
    # # ## set y-axis label
    # # fig[2][idx].set_ylabel('Condition Number', fontsize=25)
    # if idx == 0:
    #     fig[2][idx].set_ylabel('Error', fontsize=25)
    # # ## set y-axis value's fontsize
    # # fig[2][idx].tick_params(axis='y', labelsize=15)
    # fig[2][idx].tick_params(axis='y', labelsize=15)
    # # ## set grid
    # # fig[2][idx].grid(True, linestyle='--', alpha=0.6)
    # fig[2][idx].grid(True, linestyle='--', alpha=0.6)


plt.tight_layout()
plt.savefig('hvp_converge_correct_lissa.pdf')
