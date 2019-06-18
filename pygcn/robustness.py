# For now assume the model is all GCN layers, all with Relu except the last.
# We'll also just do l-inf norm for the sake of convenience.
# See paper for variable definitions.

import numpy as np
import torch
import torch.nn.functional as F
import pickle 
import math
from scipy.sparse import coo_matrix, kron

from pygcn.layers import GraphConvolution
from pygcn.utils import kronecker, tensor_product, kronecker_sparse

class GCNBoundsTwoLayer():
    """
    Special full bounds for two layers only. Specifically avoids using full kronecker products
    to improve memory usage.
    This class mostly serves as an automatic run and storage for the bounds. All methods
    are static and can be used independently of a class instance.
    Assumes the adjacency matrix has had the identity added and been row normalized.
    Args:
        model (pytorch model): 
        x (pytorch tensor, N x I): All features of the nodes of the graph. Each row corresponds
                                   to the features of that node. 
        model (pytorch model): Sequential pytorch model consisting of GCN layers. See models.py
        adj (pytorch tensor, N x N): Adjacency matrix of the model. Must be row normalized.
        eps (float): Epsilon of the norm ball associated with the verification.
        targets (list of int): Target nodes. All of the nodes who's indicies are in this list will
                               have their output bounds computed and stored by this class.
                               Will calculate all nodes if None.
        perturb_targets (list of int): Nodes we will perturb when doing the bounds calculation.
                                       All of the nodes who's indicies are in this list will
                                       have their features perturbed by epsilon. If None, will
                                       apply to all nodes.
        elision (bool): If true, will apply the elision trick to the last layer of weights.
                        Each returned row will show the bounds on the weights where the ground 
                        truth class has it's weights subtracted by the weights associated
                        for the label in the current index.
        labels (pytorch tensor, N x J): Labels used to determine ground truth for elision weights.
                                        If None, uses the model's predictions as "ground truth".
        xl (pytorch tensor, N x I): Manually input lower bound on the input data. Intended to allow the 
                                    user to do feature bounding. Take care to ensure that this a copied
                                    tensor, and not a pointer to the original data. Also make sure you only
                                    perturb the inputs that you give in the perturb_targets list.
        xu (pytorch tensor, N x I): Manually input upper bound on the input data. Intended to allow the 
                                    user to do feature bounding. Take care to ensure that this a copied
                                    tensor, and not a pointer to the original data. Also make sure you only
                                    perturb the inputs that you give in the perturb_targets list.
        p_n (float): The norm of the epsilon ball allowed for the input features.
        no_kron (bool): If true, will not use the kronecker product for computing the second layer bounds.
                        Will instead do a tensor product for each perturbed point during calculation.
        sparse_kron (bool): If true, will not calculate the second layer kronecker product as a sparce matrix.
    Vars:
        See all stored variables at end of __init__. Access as needed.
        LB and UB contain the upper and lower bound respectively for each layer.
        You probably care about the last element of each, LB[-1] and UB[-1], for the final bound.
    """
    def __init__(self, model, x, adj, eps, targets=None, perturb_targets=None, elision=False, 
                 labels=None, xl=None, xu=None, p_n=float('inf'), no_kron=False, sparse_kron=False):
        if elision:
            if labels is not None:
                gt = labels
            else:
                _, gt = torch.max(model.forward(x), 1)
        else:
            gt = None
        weights = self.extract_parameters(model, elision)
        l, u = self.compute_first_layer_preac_bounds(x, eps, weights, adj, perturb_targets, p_n, xl=xl, xu=xu)
        LB = [l]
        UB = [u]
        alpha_u, alpha_l, beta_u, beta_l = [], [], [], []
        au, al, bu, bl = self.compute_activation_bound_variables(LB[-1], UB[-1])
        alpha_u.append(au)
        alpha_l.append(al)
        beta_u.append(bu)
        beta_l.append(bl)
        # lb, ub, lmd, omg, delta, theta 
        lb, ub = self.compute_bound_and_variables(weights, adj, x, eps, 
                                                  alpha_u, alpha_l, beta_u, beta_l, 
                                                  targets, perturb_targets, elision, 
                                                  gt, p_n, no_kron, sparse_kron)
        LB.append(lb)
        UB.append(ub)

        # Store variables
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.targets = targets
        self.perturb_targets = perturb_targets
        self.elision = elision
        self.labels = gt
        self.xl = xl
        self.xu = xu
        self.p_n = p_n
        self.no_kron = sparse_kron
        self.sparse_kron = sparse_kron
        
        self.weights = weights
        self.alpha_u = alpha_u
        self.alpha_l = alpha_l
        self.beta_u = beta_u
        self.beta_l = beta_l
        # self.lmd = lmd
        # self.omg = omg
        # self.delta = delta
        # self.theta = theta
        self.LB = LB
        self.UB = UB

    # Return the weights of each GCN layer
    # Bias not supported.
    @staticmethod
    def extract_parameters(model, elision):
        weights = []
        for layer in model:
            if isinstance(layer, GraphConvolution):
                weights.append(layer.weight)
            else:
                raise ValueError("Only GCN layers supported.")
        if len(weights) > 2:
            raise ValueError("Only Two GCN layers supported.")
        # Change to elision weights
        if elision:
            w = weights[1]
            J = w.shape[1]
            w_e = -w.repeat(1, J)  # Attacker contribution
            for j1 in range(J):
                for j2 in range(J):
                    w_e[:, j1*J+j2] += w[:, j1]
            weights[1] = w_e
        return weights
        
    # Return l and u for each neuron preactivation in the first layer.
    @staticmethod
    def compute_first_layer_preac_bounds(x, eps, weights, adj, perturb_targets, p_n, xl=None, xu=None):
        N = x.shape[0]
        # if perturb_targets is not None:
        #     l = x.clone()
        #     u = x.clone()
        #     for n in perturb_targets:
        #         l[n] -= eps
        #         u[n] += eps
        # else:
        #     l = x.clone() - eps
        #     u = x.clone() + eps
        if xl is not None:
            l = xl
        if xu is not None:
            u = xu
        w = weights[0]
        I, J = w.shape
        next_l, next_u = torch.zeros(N, J), torch.zeros(N, J)
        q_n = int(1.0/ (1.0 - 1.0/p_n)) if p_n != 1 else np.inf
        # lt, ut = torch.zeros(N, I, J), torch.zeros(N, I, J)
        for j in range(J):
            if xl is not None or xl is not None:
                lt = (l * (w[:, j] > 0).float().repeat(N, 1) + 
                    u * (w[:, j] <= 0).float().repeat(N, 1))
                ut = (u * (w[:, j] > 0).float().repeat(N, 1) + 
                    l * (w[:, j] <= 0).float().repeat(N, 1))
                next_l[:, j:j+1] = adj.mm(lt).mm(w[:,j:j+1])  # One-element slice to keep dimensions
                next_u[:, j:j+1] = adj.mm(ut).mm(w[:,j:j+1])
            else:
                dualnorm = torch.norm(w[:, j], q_n)
                xw = x.mm(w[:,j:j+1])
                if perturb_targets:
                    xw_l = xw.clone()
                    xw_u = xw.clone()
                    for n in perturb_targets:
                        xw_l[n] -= eps*dualnorm
                        xw_u[n] += eps*dualnorm
                else:
                    xw_l = xw - eps*dualnorm
                    xw_u = xw + eps*dualnorm
                # next_l[:, j:j+1] = next_x - eps*dualnorm
                # next_u[:, j:j+1] = next_x + eps*dualnorm
                next_l[:, j:j+1] = adj.mm(xw_l)
                next_u[:, j:j+1] = adj.mm(xw_u)
        return next_l, next_u

    # Return lower alpha, upper alpha, lower beta, and upper beta.
    # Assumes Relu.
    # Corresponds to CROWN non-linear relaxation bounds
    @staticmethod
    def compute_activation_bound_variables(l, u):
        alpha_u_i = torch.zeros(l.shape) + (l > 0).float()  # Set proper values for all linear activations
        alpha_l_i = torch.zeros(l.shape) + (l > 0).float()
        beta_u_i = torch.zeros(l.shape)
        beta_l_i = torch.zeros(l.shape)
        nla = (l < 0) * (u > 0)  # Indexes of all that aren't linear (non-linear activations)
        alpha_u_i[nla] = u[nla] / (u[nla] - l[nla])
        alpha_l_i[nla] = (u[nla] > -l[nla]).float()  # If u > -l, slope is 1. Otherwise 0. These are the CROWN bounds.
        beta_u_i[nla] = -l[nla]
        # beta_l_i always remains 0
        return alpha_u_i, alpha_l_i, beta_u_i, beta_l_i

    # Computes the full bound for the second layer, plus variables used in computation.
    @staticmethod
    def compute_bound_and_variables(weights, adj, x, eps, alpha_u, alpha_l, beta_u, beta_l, 
                                    targets, perturb_targets, elision, gt, p_n, no_kron, sparse_kron):
        N = adj.shape[0]
        I = alpha_u[0].shape[1]
        J = weights[-1].shape[1]
        # lmd_l, omg_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
        # delta_l, theta_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
        if elision:
            J_org = int(math.sqrt(J))  # Original J
            UB, LB = torch.zeros(N, J_org), torch.zeros(N, J_org)
        else:
            UB, LB = torch.zeros(N, J), torch.zeros(N, J)
        if targets is not None:
            targs = torch.Tensor(targets).long()
        else:
            targs = torch.Tensor(list(range(N))).long()
        if perturb_targets is not None:
            p_targs = torch.Tensor(perturb_targets).long()
        else:
            p_targs = torch.Tensor(list(range(N))).long()
        if not no_kron:
            if sparse_kron:
                w0_vec = kronecker_sparse(adj.t().contiguous()[p_targs], weights[0])
            else:
                w0_vec = kronecker(adj.t().contiguous()[p_targs], weights[0])
        q_n = int(1.0/ (1.0 - 1.0/p_n)) if p_n != 1 else float('inf')
        for j in range(J):
            lmd_l = (alpha_u[0] * (weights[-1][:, j] > 0).float().repeat(N, 1) +
                     alpha_l[0] * (weights[-1][:, j] <= 0).float().repeat(N, 1))
            omg_l = (alpha_l[0] * (weights[-1][:, j] > 0).float().repeat(N, 1) +
                     alpha_u[0] * (weights[-1][:, j] <= 0).float().repeat(N, 1))
            delta_l = (beta_u[0] * (weights[-1][:, j] > 0).float().repeat(N, 1) +
                       beta_l[0] * (weights[-1][:, j] <= 0).float().repeat(N, 1))
            theta_l = (beta_l[0] * (weights[-1][:, j] > 0).float().repeat(N, 1) +
                       beta_u[0] * (weights[-1][:, j] <= 0).float().repeat(N, 1))
            lmd_l_kron = lmd_l.view(-1, 1)
            omg_l_kron = omg_l.view(-1, 1)
            # Upper bound
            ub1 = adj.mm(x).mm(weights[0]) * lmd_l  # First layer mult
            ub0 = adj.mm(ub1).mm(weights[1][:, j:j+1]) # Second layer mult
            ubb = adj.mm(lmd_l * delta_l).mm(weights[1][:, j:j+1])  # bias
            # Lower bound
            lb1 = adj.mm(x).mm(weights[0]) * omg_l  # First layer mult
            lb0 = adj.mm(lb1).mm(weights[1][:, j:j+1]) # Second layer mult
            lbb = adj.mm(omg_l * theta_l).mm(weights[1][:, j:j+1])  # bias
            for i in targs:
                if elision and gt[i] != int(j / J_org):
                    continue
                w1_vec = tensor_product(adj.t().contiguous()[:, i], weights[1][:, j]).view(-1, 1)
                if no_kron:
                    ubeps, lbeps = 0, 0
                    for t in p_targs:
                        w0_vec_row = kronecker(adj.t().contiguous()[t:t+1], weights[0])
                        ubeps_mat = w0_vec_row.mm(w1_vec * lmd_l_kron)
                        ubeps += eps * torch.norm(ubeps_mat, p=q_n)
                        lbeps_mat = w0_vec_row.mm(w1_vec * omg_l_kron)
                        lbeps += -eps * torch.norm(lbeps_mat, p=q_n)
                        del ubeps_mat
                        del lbeps_mat
                        del w0_vec_row
                else:
                    ubeps_mat = w0_vec.mm(w1_vec * lmd_l_kron)
                    ubeps = eps * torch.norm(ubeps_mat, p=q_n)
                    lbeps_mat = w0_vec.mm(w1_vec * omg_l_kron)
                    lbeps = -eps * torch.norm(lbeps_mat, p=q_n)
                if elision:
                    UB[i, j - gt[i]*J_org] = ubeps + ub0[i, 0] + ubb[i, 0]
                    LB[i, j - gt[i]*J_org] = lbeps + lb0[i, 0] + lbb[i, 0]
                else:
                    UB[i, j] = ubeps + ub0[i, 0] + ubb[i, 0]
                    LB[i, j] = lbeps + lb0[i, 0] + lbb[i, 0]
                # Ensure deletion of large matricies
                del w1_vec
        return LB[targs], UB[targs] # , [lmd_l], [omg_l], [delta_l], [theta_l] 


class GCNIntervalBounds():
    """
    Interval bounds for comparison
    """
    def __init__(self, model, x, adj, eps, targets=None, perturb_targets=None, elision=False, 
                 labels=None, xl=None, xu=None):
        if elision:
            if labels is not None:
                gt = labels
            else:
                _, gt = torch.max(model.forward(x), 1)
        else:
            gt = None
        weights = self.extract_parameters(model, elision)
        LB, UB = self.compute_interval_bounds(x, eps, weights, adj, perturb_targets, targets,
                                              elision, gt, xl=xl, xu=xu)

        # Store variables
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.targets = targets
        self.perturb_targets = perturb_targets
        self.elision = elision
        self.labels = gt
        self.xl = xl
        self.xu = xu
        
        self.weights = weights
        self.LB = LB
        self.UB = UB

    # Return the weights of each GCN layer
    # Bias not supported.
    @staticmethod
    def extract_parameters(model, elision):
        weights = []
        for layer in model:
            if isinstance(layer, GraphConvolution):
                weights.append(layer.weight)
            else:
                raise ValueError("Only GCN layers supported.")
        if len(weights) > 2:
            raise ValueError("Only Two GCN layers supported.")
        # Change to elision weights
        if elision:
            w = weights[-1]
            J = w.shape[-1]
            w_e = -w.repeat(1, J)  # Attacker contribution
            for j1 in range(J):
                for j2 in range(J):
                    w_e[:, j1*J+j2] += w[:, j1]
            weights[-1] = w_e
        return weights
        
    # Return l and u for each neuron preactivation in the first layer.
    @staticmethod
    def compute_interval_bounds(x, eps, weights, adj, perturb_targets, targets, 
                                elision, labels, xl=None, xu=None):
        LB, UB = [], []
        N = x.shape[0]
        if perturb_targets is not None:
            l = x.clone()
            u = x.clone()
            for n in perturb_targets:
                l[n] -= eps
                u[n] += eps
        else:
            l = x.clone() - eps
            u = x.clone() + eps
        if xl is not None:
            l = xl
        if xu is not None:
            u = xu
        for k in range(len(weights)):
            w = weights[k]
            I, J = w.shape
            next_l, next_u = torch.zeros(N, J), torch.zeros(N, J)
            # print(w.shape)
            # print(l.shape)
            for j in range(J):
                # print((w[:, j] > 0).float().repeat(N, 1).shape)
                lt = (l * (w[:, j] > 0).float().repeat(N, 1) + 
                      u * (w[:, j] <= 0).float().repeat(N, 1))
                ut = (u * (w[:, j] > 0).float().repeat(N, 1) + 
                      l * (w[:, j] <= 0).float().repeat(N, 1))
                next_l[:, j:j+1] = adj.mm(lt).mm(w[:,j:j+1])  # One-element slice to keep dimensions
                next_u[:, j:j+1] = adj.mm(ut).mm(w[:,j:j+1])
            if k != len(weights) - 1:
                next_l = F.relu(next_l)
                next_u = F.relu(next_u)
            LB.append(next_l)
            UB.append(next_u)
            l = next_l
            u = next_u
        # Select target and proper elision labels
        if elision:
            N, J = LB[-1].shape
            J_org = int(math.sqrt(J))  # Original J
            new_l, new_u = torch.zeros(N, J_org), torch.zeros(N, J_org)
            for n in range(N):
                new_u[n, :] = UB[-1][n, labels[n]*J_org:(labels[n]+1)*J_org]
                new_l[n, :] = LB[-1][n, labels[n]*J_org:(labels[n]+1)*J_org]
            LB[-1] = new_l
            UB[-1] = new_u
        if targets is not None:
            for i in range(len(LB)):
                LB[i] = LB[i][targets]
                UB[i] = UB[i][targets]
        return LB, UB
# -----------------------------------------------------------------------------------------------------


# TODO: Some errors in the bound selection via weights. Fix after theorem update.
class GCNBoundsRelaxed():
    # This class mostly serves as an automatic run and storage for the bounds. All methods
    # are static and can be used independently of a class instance.
    # Assumes the adjacency matrix has been normalized and had the identity added.
    # This is the relaxed bounds derivation in the paper.
    def __init__(self, model, x, adj, eps, targets=None):
        weights = self.extract_parameters(model)
        l, u = self.compute_first_layer_preac_bounds(x, eps, weights, adj, targets)
        LB = [l]
        UB = [u]
        alpha_u, alpha_l, beta_u, beta_l = [], [], [], []
        for l in range(len(weights)-1):
            au, al, bu, bl = self.compute_activation_bound_variables(LB[-1], UB[-1])
            alpha_u.append(au)
            alpha_l.append(al)
            beta_u.append(bu)
            beta_l.append(bl)
            Lambda, Omega, J_tilde, lmd, omg, delta, theta = self.compute_linear_bound_variables(weights[:l+2], adj, alpha_u, alpha_l, beta_u, beta_l)
            lb, ub = self.compute_bounds(eps, x, Lambda, Omega, J_tilde, lmd, omg, delta, theta)
            LB.append(lb)
            UB.append(ub)

        # Store variables
        self.targets = targets
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.weights = weights
        self.alpha_u = alpha_u
        self.alpha_l = alpha_l
        self.beta_u = beta_u
        self.beta_l = beta_l
        self.Lambda = Lambda
        self.Omega = Omega
        self.J_tilde = J_tilde
        self.lmd = lmd
        self.omg = omg
        self.delta = delta
        self.theta = theta
        self.LB = LB
        self.UB = UB

    # Return the weights of each GCN layer
    # Bias not supported.
    @staticmethod
    def extract_parameters(model):
        weights = []
        for layer in model:
            if isinstance(layer, GraphConvolution):
                weights.append(layer.weight)
            else:
                raise ValueError("Only GCN layers supported.")
        return weights 

    # Return l and u for each neuron preactivation. Also returns tilde variables for debug purposes.
    # Corresponds to theorem 3.1
    @staticmethod
    def compute_first_layer_preac_bounds(x, eps, weights, adj, targets, xl=None, xu=None):
        N = x.shape[0]
        if targets is not None:
            l = x.clone()
            u = x.clone()
            for n in targets:
                l[n] -= eps
                u[n] += eps
        else:
            l = x.clone() - eps
            u = x.clone() + eps
        if xl is not None:
            l = xl
        if xu is not None:
            u = xu
        w = weights[0]
        I, J = w.shape
        # TODO: Vectorize
        lt, ut = torch.zeros(N, I, J), torch.zeros(N, I, J)
        for j in range(J):
            lt[:, :, j] = (l * (w[:, j] > 0).float().repeat(N, 1) + 
                           u * (w[:, j] <= 0).float().repeat(N, 1))
            ut[:, :, j] = (u * (w[:, j] > 0).float().repeat(N, 1) + 
                           l * (w[:, j] <= 0).float().repeat(N, 1))
        # TODO: Vectorize
        next_l, next_u = torch.zeros(N, J), torch.zeros(N, J)
        for j in range(J):
            next_l[:, j:j+1] = adj.mm(lt[:, :, j]).mm(w[:,j:j+1])  # One-element slice to keep dimensions
            next_u[:, j:j+1] = adj.mm(ut[:, :, j]).mm(w[:,j:j+1])
        return next_l, next_u
        # N = x.shape[0]
        # next_l, next_u = GCNBoundsFull.compute_first_layer_preac_bounds(x, eps, weights, adj, targets)
        # return next_l.view(N, -1), next_u.view(N, -1)

    # Return lower alpha, upper alpha, lower beta, and upper beta.
    # Assumes Relu.
    # Corresponds to CROWN non-linear relaxation bounds
    @staticmethod
    def compute_activation_bound_variables(l, u):
        alpha_u_i = torch.zeros(l.shape) + (l > 0).float()  # Set proper values for all linear activations
        alpha_l_i = torch.zeros(l.shape) + (l > 0).float()
        beta_u_i = torch.zeros(l.shape)
        beta_l_i = torch.zeros(l.shape)
        nla = (l < 0) * (u > 0)  # Indexes of all that aren't linear (non-linear activations)
        alpha_u_i[nla] = u[nla] / (u[nla] - l[nla])
        alpha_l_i[nla] = (u[nla] > -l[nla]).float()  # If u > -l, slope is 1. Otherwise 0. These are the CROWN bounds.
        beta_u_i[nla] = -l[nla]
        # beta_l_i always remains 0
        return alpha_u_i, alpha_l_i, beta_u_i, beta_l_i

    # Return big lambda, big omega, J, small lambda, small omega, delta, theta, lambda tilde, and omega tilde.
    # Corresponds to theorem 4.2
    @staticmethod
    def compute_linear_bound_variables(weights, adj, alpha_u, alpha_l, beta_u, beta_l):
        N = adj.shape[0]
        J = weights[-1].shape[1]
        Lambda = [weights[-1]]
        Omega = [weights[-1]]
        J_tilde = [adj]
        lmd = []
        omg = []
        delta = []
        theta = []
        lmb_tilde = []
        omg_tilde = []
        for ind, w in reversed(list(enumerate(weights[:-1]))):
            # ind += 1  # alphas and betas are have an extra 0 element.
            # small lambda, small omega, delta, theta, and tilde lambda + omega
            I = alpha_u[ind].shape[1]  # Also Lambda[0][0]
            lmd_l, omg_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
            delta_l, theta_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
            lmb_tilde_l, omg_tilde_l = torch.zeros(I, J), torch.zeros(I, J)
            # TODO: Vectorize
            for j in range(J):
                lmd_l[:, :, j] = (alpha_u[ind] * (Lambda[0][:, j] >= 0).float().repeat(N, 1) +
                                  alpha_l[ind] * (Lambda[0][:, j] < 0).float().repeat(N, 1))
                omg_l[:, :, j] = (alpha_l[ind] * (Omega[0][:, j] >= 0).float().repeat(N, 1) +
                                  alpha_u[ind] * (Omega[0][:, j] < 0).float().repeat(N, 1))
                delta_l[:, :, j] = (beta_u[ind] * (Lambda[0][:, j] >= 0).float().repeat(N, 1) +
                                    beta_l[ind] * (Lambda[0][:, j] < 0).float().repeat(N, 1))
                theta_l[:, :, j] = (beta_l[ind] * (Omega[0][:, j] >= 0).float().repeat(N, 1) +
                                    beta_u[ind] * (Omega[0][:, j] < 0).float().repeat(N, 1))
                lmb_tilde_l[:, j] = (torch.max(alpha_u[ind], 0)[0] * (Lambda[0][:, j] >= 0).float() +
                                     torch.min(alpha_l[ind], 0)[0] * (Lambda[0][:, j] < 0).float())
                omg_tilde_l[:, j] = (torch.min(alpha_l[ind], 0)[0] * (Omega[0][:, j] >= 0).float() +
                                     torch.max(alpha_u[ind], 0)[0] * (Omega[0][:, j] < 0).float())
            # J_tilde
            j_tilde_l = adj.mm(J_tilde[0])
            # Lambda and Omega
            Lambda_l = w.mm(lmb_tilde_l * Lambda[0])
            Omega_l = w.mm(omg_tilde_l * Omega[0])
            # Prepend all variables
            Lambda = [Lambda_l] + Lambda
            Omega = [Omega_l] + Omega
            J_tilde = [j_tilde_l] + J_tilde
            lmd = [lmd_l] + lmd
            omg = [omg_l] + omg
            delta = [delta_l] + delta
            theta = [theta_l] + theta
            lmb_tilde = [lmb_tilde_l] + lmb_tilde
            omg_tilde = [omg_tilde_l] + omg_tilde
        return Lambda, Omega, J_tilde, lmd, omg, delta, theta

    # Return global lower and upper bounds for each data point.
    # Corresponds to corollary 4.3
    @staticmethod
    def compute_bounds(eps, xo, Lambda, Omega, J_tilde, lmd, omg, delta, theta):
        N, J = J_tilde[0].shape[0], Lambda[0].shape[1]
        Lambda_1 = Lambda[0]
        Omega_1 = Omega[0]
        J_1 = J_tilde[0]
        # first term, constant
        # L-infty ball, dual to L1
        # jsum = torch.sum(torch.abs(J_1))
        # print("jsum: ", jsum)
        t1_u, t1_l = torch.zeros(N, J), torch.zeros(N, J)
        for j in range(J):
            t1_u[:, j] = eps * torch.sum(torch.abs(Lambda_1[:, j]))
            t1_l[:, j] = -eps * torch.sum(torch.abs(Omega_1[:, j]))
        # L1 ball, dual to L-infty
        # t1_u = eps * torch.max(torch.abs(Lambda_1))
        # t1_l = -eps * torch.max(torch.abs(Omega_1))
        # second term, [n,j] matrix
        t2_u = J_1.mm(xo.mm(Lambda_1))
        t2_l = J_1.mm(xo.mm(Omega_1))
        # third term, [n,j] matrix
        t3_u, t3_l = torch.zeros(N, J), torch.zeros(N, J)
        # TODO: Vectorize
        for i in range(len(J_tilde)-1):
            for j in range(J):
                t3_u[:, j:j+1] += J_tilde[i+1].mm((lmd[i][:, :, j] * delta[i][:, :, j]).mm(Lambda[i+1][:, j:j+1]))
                t3_l[:, j:j+1] += J_tilde[i+1].mm((omg[i][:, :, j] * theta[i][:, :, j]).mm(Omega[i+1][:, j:j+1]))
        return t1_l + t2_l + t3_l, t1_u + t2_u + t3_u


class GCNBoundsFull():
    # This class mostly serves as an automatic run and storage for the bounds. All methods
    # are static and can be used independently of a class instance.
    # Assumes the adjacency matrix has been normalized and had the identity added.
    # These are the full bounds in the paper using the Kronecker product. Note that this
    # can take much more computation power and memory.
    # Targets are the data points (rows) we are applying the pertebations to. Should be list of indicies.
    # If none, will apply pertebation to all points.
    def __init__(self, model, x, adj, eps, targets=None):
        weights = self.extract_parameters(model)
        l, u = self.compute_first_layer_preac_bounds(x, eps, weights, adj, targets)
        LB = [l]
        UB = [u]
        alpha_u, alpha_l, beta_u, beta_l = [], [], [], []
        for l in range(len(weights)-1):
            au, al, bu, bl = self.compute_activation_bound_variables(LB[-1], UB[-1])
            alpha_u.append(au)
            alpha_l.append(al)
            beta_u.append(bu)
            beta_l.append(bl)
            Lambda, Omega, lmd, omg, delta, theta = self.compute_linear_bound_variables(weights[:l+2], adj, alpha_u, alpha_l, beta_u, beta_l)
            lb, ub = self.compute_bounds(eps, x, Lambda, Omega, lmd, omg, delta, theta)
            LB.append(lb)
            UB.append(ub)

        # Store variables
        self.targets = targets
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.weights = weights
        self.alpha_u = alpha_u
        self.alpha_l = alpha_l
        self.beta_u = beta_u
        self.beta_l = beta_l
        self.Lambda = Lambda
        self.Omega = Omega
        self.lmd = lmd
        self.omg = omg
        self.delta = delta
        self.theta = theta
        self.LB = LB
        self.UB = UB

    @staticmethod
    def extract_parameters(model):
        return GCNBoundsRelaxed.extract_parameters(model)

    @staticmethod
    def compute_first_layer_preac_bounds(x, eps, weights, adj, targets):
        return GCNBoundsRelaxed.compute_first_layer_preac_bounds(x, eps, weights, adj, targets)
        # w = kronecker(adj.t().contiguous(), weights[0])
        # xo = x.view(1, -1)
        # N, I = x.shape
        # NI, NJ = w.shape
        # if targets:
        #     targs_expanded = []
        #     for t in targets:
        #         targs_expanded += list(range(I*t, I*t+I))
        #     targs = torch.Tensor(targs_expanded).long()
        # Ax0 = xo.mm(w).view(-1)
        # LB_first, UB_first = torch.zeros(NJ), torch.zeros(NJ)
        # for j in range(NJ):
        #     if targets is None:
        #         dualnorm_Aj = torch.sum(torch.abs(w[:, j]))  # dual of inf, 1 norm
        #     else:
        #         dualnorm_Aj = torch.sum(torch.abs(w[targs, j]))  # dual of inf, 1 norm
        #     UB_first[j] = Ax0[j]+eps*dualnorm_Aj
        #     LB_first[j] = Ax0[j]-eps*dualnorm_Aj
        # return LB_first, UB_first

    @staticmethod
    def compute_activation_bound_variables(l, u):
        return GCNBoundsRelaxed.compute_activation_bound_variables(l, u)

    @staticmethod
    def compute_linear_bound_variables(weights, adj, alpha_u, alpha_l, beta_u, beta_l):
        N = adj.shape[0]
        J = weights[-1].shape[1]
        NJ = N * J  # Because we're vectorizing to vanilla CROWN, dimensions are now in NI x NJ.
        Lambda = [torch.eye(NJ)]
        Omega = [torch.eye(NJ)]
        lmd = []
        omg = []
        delta = [torch.zeros(NJ, NJ)]
        theta = [torch.zeros(NJ, NJ)]
        for ind, w in reversed(list(enumerate(weights))):
            ind -= 1  # weights have one more element
            w_vec = kronecker(adj.t().contiguous(), w)  # Massive expanded weight matrix
            NI = w_vec.shape[0]
            # lower case variables
            weight_lambda = w_vec.mm(Lambda[0])
            weight_omega = w_vec.mm(Omega[0])
            if ind != -1:
                lmd_l = (alpha_u[ind].view(-1,1).repeat(1, NJ) * (weight_lambda > 0).float() +
                         alpha_l[ind].view(-1,1).repeat(1, NJ) * (weight_lambda <= 0).float())
                omg_l = (alpha_l[ind].view(-1,1).repeat(1, NJ) * (weight_omega > 0).float() +
                         alpha_u[ind].view(-1,1).repeat(1, NJ) * (weight_omega <= 0).float())
                delta_l = (beta_u[ind].view(-1,1).repeat(1, NJ) * (weight_lambda > 0).float() +
                           beta_l[ind].view(-1,1).repeat(1, NJ) * (weight_lambda <= 0).float())
                theta_l = (beta_l[ind].view(-1,1).repeat(1, NJ) * (weight_omega > 0).float() +
                           beta_u[ind].view(-1,1).repeat(1, NJ) * (weight_omega <= 0).float())
            else:
                lmd_l, omg_l = torch.ones(NI, NJ), torch.ones(NI, NJ)
            # Lambda and Omega
            Lambda_l = weight_lambda * lmd_l
            Omega_l = weight_omega * omg_l
            Lambda = [Lambda_l] + Lambda
            Omega = [Omega_l] + Omega
            lmd = [lmd_l] + lmd
            omg = [omg_l] + omg
            if ind != -1:
                delta = [delta_l] + delta
                theta = [theta_l] + theta
        return Lambda, Omega, lmd, omg, delta, theta

    # Return global lower and upper bounds for each data point.
    # Corresponds to corollary 4.3
    @staticmethod
    def compute_bounds(eps, xo, Lambda, Omega, lmd, omg, delta, theta):
        N = xo.shape[0]
        J = int(Lambda[0].shape[1] / N)
        xo = xo.view(1, -1)  # Flatten x to be compatible with vanilla CROWN
        Lambda_0 = Lambda[0]
        Omega_0 = Omega[0]
        # first term, constant
        # L-infty ball, dual to L1
        t1_u, t1_l = torch.zeros(N*J), torch.zeros(N*J)
        for j in range(N*J):
            t1_u[j] = eps * torch.sum(torch.abs(Lambda_0[:, j]))
            t1_l[j] = -eps * torch.sum(torch.abs(Omega_0[:, j]))
        t1_u = t1_u.view(N, J)
        t1_l = t1_l.view(N, J)
        # L1 ball, dual to L-infty
        # t1_u = eps * torch.max(torch.abs(Lambda_0))
        # t1_l = -eps * torch.max(torch.abs(Omega_0))
        # second term, [n,j] matrix
        t2_u = xo.mm(Lambda_0).view(N, J)
        t2_l = xo.mm(Omega_0).view(N, J)
        # third term, [n,j] matrix
        t3_u, t3_l = torch.zeros(N*J), torch.zeros(N*J)
        # TODO: Vectorize
        for i in range(1, len(Lambda)):
            for j in range(N*J):
                t3_u[j] += delta[i-1][:, j].dot(Lambda[i][:, j]) # Delta and theta have no 0 element, so list is shifted
                t3_l[j] += theta[i-1][:, j].dot(Omega[i][:, j])
        t3_u = t3_u.view(N, J)
        t3_l = t3_l.view(N, J)
        # with open("gcn_small_bound_boundcheck_eps1-"+str(int(1/eps))+".pkl", "wb") as f:
        #     pickle.dump({'Ax0_UB': t2_u.view(-1), 
        #                 'Ax0_LB': t2_l.view(-1), 
        #                 'dualnorm_ub': t1_u.view(-1), 
        #                 'dualnorm_lb': t1_l.view(-1),
        #                 'constants_ub': t3_u.view(-1), 
        #                 'constants_lb': t3_l.view(-1),
        #                 }, f)
        return [t1_l + t2_l + t3_l, t1_u + t2_u + t3_u]
