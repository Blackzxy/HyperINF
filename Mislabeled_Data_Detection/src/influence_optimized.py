from time import time
from tqdm import tqdm
import sys
from collections import defaultdict
import pandas as pd
import pickle, os
import torch
import numpy as np

class IFEngine(object):
    def __init__(self):
        self.time_dict=defaultdict(list)
        self.hvp_dict=defaultdict(list)
        self.IF_dict=defaultdict(list)

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict, noise_index=None):
        self.tr_grad_dict = tr_grad_dict
        self.tr_grad_dict_flat = {}
        # for key in self.tr_grad_dict:
        #     self.tr_grad_dict_flat[key] = {}
        #     for weight_name in self.tr_grad_dict[key]:
        #         self.tr_grad_dict_flat[key][weight_name] = self.tr_grad_dict[key][weight_name].reshape(1, -1)

        self.val_grad_dict = val_grad_dict
        # self.val_grad_dict_flat = {}
        # for key in self.val_grad_dict:
        #     self.val_grad_dict_flat[key] = {}
        #     for weight_name in self.val_grad_dict[key]:
        #         self.val_grad_dict_flat[key][weight_name] = self.val_grad_dict[key][weight_name].reshape(1, -1)
        
        self.noise_index = noise_index

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        self.compute_val_grad_avg()

    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        self.val_grad_avg_dict={}
        self.val_grad_avg_dict_flat={}
        for weight_name in self.val_grad_dict[0]:
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.val_grad_dict[0][weight_name].shape).to(self.val_grad_dict[0][weight_name].device)
            #self.val_grad_avg_dict_flat[weight_name]=torch.zeros(self.val_grad_dict_flat[0][weight_name].shape).to(self.val_grad_dict_flat[0][weight_name].device)
            for val_id in self.val_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.val_grad_dict[val_id][weight_name] / self.n_val
                #self.val_grad_avg_dict_flat[weight_name] += self.val_grad_dict_flat[val_id][weight_name] / self.n_val

    def estimate_lambda(self, lambda_const_param=10):
        '''
        Compute the damping_factor. Adapt from DataINF (https://github.com/ykwon0407/DataInf).
        lambda_const = 0.1 x (n * d_l)^(-1) \sum_{i=1}^{n} ||grad_i^l||_2^2
        '''
        self.lambda_const = {}
        for _, (weight_name, gv) in enumerate(tqdm(self.val_grad_avg_dict.items())):
            # lambda_const computation = 0.1 x (n * d_l)^(-1) \sum_{i=1}^{n} ||grad_i^l||_2^2
            S=torch.zeros(self.n_train).to(gv.device)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const_l = torch.mean(S) / lambda_const_param # layer-wise lambda
            self.lambda_const[weight_name] = lambda_const_l


    def compute_hvps(self, lambda_const_param=10, compute_accurate=True):
        self.estimate_lambda(lambda_const_param=lambda_const_param)
        #self.compute_hvp_iterative_full_rank(lambda_const_param=lambda_const_param)
        self.compute_hvp_iterative(lambda_const_param=lambda_const_param)
        self.compute_hvp_identity()
        self.compute_hvp_datainf(lambda_const_param=lambda_const_param)
        self.compute_hvp_LiSSA(lambda_const_param=lambda_const_param)
        if compute_accurate:
            self.compute_hvp_accurate(lambda_const_param=lambda_const_param)
        for method_name in self.time_dict:
            print(f"{method_name}: {self.time_dict[method_name]}")


    def compute_hvp_identity(self):
        start_time = time()
        self.hvp_dict['identity'] = self.val_grad_avg_dict.copy()
        self.time_dict['identity'] = time()-start_time
        print("Time taken for Hessian-free: ", self.time_dict['identity'])

    def compute_hvp_iterative(self, lambda_const_param=10, n_iteration=30):

        def schulz_inverse_stable(A, damping_factor=0, max_iterations=20, tol=1e-6):
            n = A.shape[0]
            #I = np.eye(n)
            I = torch.eye(n, device=A.device)
            A_damped = A + damping_factor * I  # Apply damping

            #X = np.eye(n) * 0.00005  # Initial estimate of inverse matrix
            X = torch.eye(n, device=A.device) * 0.00005  # Initial estimate of inverse matrix

            for _ in range(max_iterations):
                #X = X.dot(2 * I - A_damped.dot(X))
                X = X @ (2 * I - A_damped @ X)

                # # Check for convergence
                # if np.linalg.norm(I - A.dot(X)) < tol:
                #     break

            return X
        
        start_time = time()
        hvp_iterative_dict={}
        # device = self.val_grad_avg_dict[list(self.val_grad_avg_dict.keys())[0]].device

        for _, weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            # iterative hvp computation
            # G_l: same shape of self.tr_grad_dict[0][weight_name].T @ self.tr_grad_dict[0][weight_name]
            tmp_grad = self.tr_grad_dict[0][weight_name].to(self.val_grad_avg_dict[weight_name].device)
            # print(tmp_grad.shape)
            # sys.exit(0)
            G_l = torch.zeros((tmp_grad.T @ tmp_grad).shape).to(self.val_grad_avg_dict[weight_name].device)
            del tmp_grad

            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device)
                G_l += tmp_grad.T @ tmp_grad / self.n_train
                del tmp_grad


            # G_l = torch.zeros((self.tr_grad_dict[0][weight_name].T @ self.tr_grad_dict[0][weight_name]).shape).to(self.val_grad_avg_dict[weight_name].device)
            # for tr_id in self.tr_grad_dict:
            #     tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device) # (grad_i^l)^T
            #     G_l += tmp_grad.T @ tmp_grad / self.n_train
            
            G_l = G_l + self.lambda_const[weight_name] * torch.eye(G_l.shape[0], device=G_l.device)
           # G_l = G_l.cpu().detach().numpy()
            G_l_inv = schulz_inverse_stable(G_l, damping_factor=0.001, max_iterations=n_iteration, tol=1e-6)

            hvp_iterative_dict[weight_name] = torch.tensor(self.val_grad_avg_dict[weight_name] @ G_l_inv)
            #print(hvp_iterative_dict[weight_name])
        self.hvp_dict['Low-Rank_HyperINF'] = hvp_iterative_dict
        self.time_dict['Low-Rank_HyperINF'] = time()-start_time
        print("Time taken for Low-Rank HyperINF: ", self.time_dict['Low-Rank_HyperINF'])

    def compute_hvp_iterative_full_rank(self, lambda_const_param=10, n_iteration=30):

        def schulz_inverse_stable(A, damping_factor=0, max_iterations=20, tol=1e-6):
            n = A.shape[0]
            #I = np.eye(n)
            I = torch.eye(n, device=A.device)
            A_damped = A + damping_factor * I  # Apply damping

            #X = np.eye(n) * 0.00005  # Initial estimate of inverse matrix
            X = torch.eye(n, device=A.device) * 0.00005  # Initial estimate of inverse matrix

            for _ in range(max_iterations):
                #X = X.dot(2 * I - A_damped.dot(X))
                X = X @ (2 * I - A_damped @ X)

                # # Check for convergence
                # if np.linalg.norm(I - A.dot(X)) < tol:
                #     break

            return X
        
        start_time = time()
        hvp_iterative_dict={}
        # device = self.val_grad_avg_dict[list(self.val_grad_avg_dict.keys())[0]].device


        for _, weight_name in enumerate(tqdm(self.val_grad_avg_dict_flat)):
  
            # iterative hvp computation
            # G_l: same shape of self.tr_grad_dict[0][weight_name].T @ self.tr_grad_dict[0][weight_name]
            G_l = torch.zeros((self.tr_grad_dict_flat[0][weight_name].T @ self.tr_grad_dict_flat[0][weight_name]).shape).to(self.val_grad_avg_dict_flat[weight_name].device)
            for tr_id in self.tr_grad_dict_flat:
                tmp_grad = self.tr_grad_dict_flat[tr_id][weight_name].to(self.val_grad_avg_dict_flat[weight_name].device) # (grad_i^l)^T
                G_l += tmp_grad.T @ tmp_grad / self.n_train
                del tmp_grad
            
            G_l = G_l + self.lambda_const[weight_name] * torch.eye(G_l.shape[0], device=G_l.device)
           # G_l = G_l.cpu().detach().numpy()
            G_l_inv = schulz_inverse_stable(G_l, damping_factor=0.001, max_iterations=n_iteration, tol=1e-6)

            hvp_iterative_dict[weight_name] = torch.tensor(self.val_grad_avg_dict_flat[weight_name] @ G_l_inv)
            #print(hvp_iterative_dict[weight_name])
        self.hvp_dict['Full-Rank_HyperINF'] = hvp_iterative_dict
        self.time_dict['Full-Rank_HyperINF'] = time()-start_time
        print("Time taken for HyperINF-FullRank: ", self.time_dict['Full-Rank_HyperINF'])
               



    def compute_hvp_datainf(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict={}
       

        for _ , weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            lambda_const = self.lambda_const[weight_name]
            # hvp computation
            hvp=torch.zeros(self.val_grad_avg_dict[weight_name].shape).to(self.val_grad_avg_dict[weight_name].device)
            for tr_id in self.tr_grad_dict: # i
                tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device) # grad_i^l
                # L_(l,i) / (lambda + ||grad_i^l||_2^2) in Eqn. (5) 
                C_tmp = torch.sum(self.val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2)).to(self.val_grad_avg_dict[weight_name].device)
                # (v_l^T - C_tmp * (grad_i^l)^T ) / (n * lambda) in Eqn. (5)
                hvp += (self.val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                del tmp_grad

            hvp_proposed_dict[weight_name] = hvp 
        self.hvp_dict['Datainf'] = hvp_proposed_dict
        self.time_dict['Datainf'] = time()-start_time
        print("Time taken for Datainf: ", self.time_dict['Datainf'])

    def compute_hvp_accurate(self, lambda_const_param=10):
        start_time = time()
        hvp_accurate_dict={}
        for _ , weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation (eigenvalue decomposition)
            # AAt_matrix = torch.zeros(torch.outer(self.tr_grad_dict[0][weight_name].reshape(-1), 
            #                                      self.tr_grad_dict[0][weight_name].reshape(-1)).shape)
            # for tr_id in self.tr_grad_dict:
                
            #     tmp_mat = torch.outer(self.tr_grad_dict[tr_id][weight_name].reshape(-1), 
            #                           self.tr_grad_dict[tr_id][weight_name].reshape(-1))
            #     AAt_matrix += tmp_mat

            AAt_matrix = torch.zeros((self.tr_grad_dict[0][weight_name].T @ self.tr_grad_dict[0][weight_name]).shape)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                AAt_matrix += tmp_grad.T @ tmp_grad / self.n_train
            
            AAt_matrix = AAt_matrix + lambda_const * torch.eye(AAt_matrix.shape[0])
            AAt_matrix = AAt_matrix.cpu().detach().numpy()
            AAt_matrix_inv = np.linalg.inv(AAt_matrix)
            hvp_accurate_dict[weight_name] = torch.tensor(self.val_grad_avg_dict[weight_name].cpu().detach().numpy() @ AAt_matrix_inv)
                
            # L, V = torch.linalg.eig(AAt_matrix)
            # L, V = L.float(), V.float()
            # hvp = self.val_grad_avg_dict[weight_name].reshape(-1) @ V
            # hvp = (hvp / (lambda_const + L/ self.n_train)) @ V.T

            # hvp_accurate_dict[weight_name] = hvp.reshape(len(self.tr_grad_dict[0][weight_name]), -1)
            # del tmp_mat, AAt_matrix, V # to save memory
        self.hvp_dict['accurate'] = hvp_accurate_dict
        self.time_dict['accurate'] = time()-start_time 
        print("Time taken for Accurate: ", self.time_dict['accurate'])

    def compute_hvp_LiSSA(self, lambda_const_param=10, n_iteration=10, alpha_const=1.):
        start_time = time()
        hvp_LiSSA_dict={}
       

        for _, weight_name in enumerate(tqdm(self.val_grad_avg_dict)):
            lambda_const = self.lambda_const[weight_name]
            # hvp computation
            running_hvp=self.val_grad_avg_dict[weight_name]
            for _ in range(n_iteration):
                hvp_tmp=torch.zeros(self.val_grad_avg_dict[weight_name].shape).to(self.val_grad_avg_dict[weight_name].device)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name].to(self.val_grad_avg_dict[weight_name].device)
                    hvp_tmp += (torch.sum(tmp_grad*running_hvp)*tmp_grad - lambda_const*running_hvp) / self.n_train
                    #del tmp_grad
                running_hvp = self.val_grad_avg_dict[weight_name] + running_hvp - alpha_const*hvp_tmp
            hvp_LiSSA_dict[weight_name] = running_hvp 


        self.hvp_dict['LiSSA'] = hvp_LiSSA_dict
        self.time_dict['LiSSA'] = time()-start_time 
        print("Time taken for LiSSA: ", self.time_dict['LiSSA'])

    def compute_IF(self):
        for method_name in self.hvp_dict:
            if_tmp_dict = {}
            if method_name == "Full-Rank_HyperINF":
                start_time = time()
                for tr_id in self.tr_grad_dict_flat:
                    if_tmp_value = 0
                    for weight_name in self.val_grad_avg_dict_flat:
                        if_tmp_value += torch.sum(self.hvp_dict[method_name][weight_name]*self.tr_grad_dict_flat[tr_id][weight_name].to(self.hvp_dict[method_name][weight_name].device))
                    if_tmp_dict[tr_id]= -if_tmp_value.cpu()
                
                print("Time for Dot Products: ", time()-start_time)
            else:
                start_time = time()
                for tr_id in self.tr_grad_dict:
                    if_tmp_value = 0
                    for weight_name in self.val_grad_avg_dict:
                        if_tmp_value += torch.sum(self.hvp_dict[method_name][weight_name]*self.tr_grad_dict[tr_id][weight_name].to(self.hvp_dict[method_name][weight_name].device))
                    if_tmp_dict[tr_id]= -if_tmp_value.cpu()
                print(f"N: {self.n_train}, Time for Dot Product: {time() - start_time}")
                # print(-if_tmp_value)
                
            self.IF_dict[method_name] = pd.Series(if_tmp_dict, dtype=float).to_numpy()    

    def save_result(self, noise_index, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['noise_index']=noise_index
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)

class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self):
        self.time_dict = defaultdict(list)
        self.hvp_dict = defaultdict(list)
        self.IF_dict = defaultdict(list)

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())

    def compute_hvps(self, lambda_const_param=10):
        self.compute_hvp_identity()
        self.compute_hvp_proposed(lambda_const_param=lambda_const_param)

    def compute_hvp_identity(self):
        start_time = time()
        self.hvp_dict["identity"] = self.val_grad_dict.copy()
        self.time_dict["identity"] = time() - start_time

    def compute_hvp_proposed(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict=defaultdict(dict)
        for val_id in tqdm(self.val_grad_dict.keys()):
            for weight_name in self.val_grad_dict[val_id]:
                # lambda_const computation
                S=torch.zeros(len(self.tr_grad_dict.keys()))
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    S[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

                # hvp computation
                hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                hvp_proposed_dict[val_id][weight_name] = hvp
        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time()-start_time

    def compute_IF(self):
        for method_name in self.hvp_dict:
            print("Computing IF for method: ", method_name)
            if_tmp_dict = defaultdict(dict)
            for tr_id in self.tr_grad_dict:
                for val_id in self.val_grad_dict:
                    if_tmp_value = 0
                    for weight_name in self.val_grad_dict[0]:
                        if_tmp_value += torch.sum(self.hvp_dict[method_name][val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                    if_tmp_dict[tr_id][val_id]=-if_tmp_value

            self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)   

    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)
