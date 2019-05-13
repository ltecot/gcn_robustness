# For now assume the model is all GCN layers, all with Relu except the last.
# We'll also just do l-inf norm for the sake of convenience.
# See paper for variable definitions.

import torch
import torch.nn.functional as F

from pygcn.layers import GraphConvolution
from pygcn.utils import kronecker

class GCNBoundsRelaxed():
    # This class mostly serves as an automatic run and storage for the bounds. All methods
    # are static and can be used independently of a class instance.
    # Assumes the adjacency matrix has been normalized and had the identity added.
    # This is the relaxed bounds derivation in the paper.
    def __init__(self, model, x, adj, eps):
        weights = self.extract_parameters(model)
        l, u = self.compute_first_layer_preac_bounds(x, eps, weights, adj)
        LB = [l]
        UB = [u]
        alpha_u, alpha_l, beta_u, beta_l = [], [], [], []
        for l in range(len(weights)-1):
            # alpha_u, alpha_l, beta_u, beta_l = self.compute_activation_bound_variables(l, u)
            # Lambda, Omega, J_tilde, lmd, omg, delta, theta, lmd_tilde, omg_tilde = self.compute_linear_bound_variables(weights, adj, alpha_u, alpha_l, beta_u, beta_l)
            # bounds = self.compute_bounds(eps, x, Lambda, Omega, J_tilde, lmd, omg, delta, theta)
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
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.weights = weights
        # self.l = l
        # self.u = u
        # self.l_tilde = l_tilde
        # self.u_tilde = u_tilde
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
        # self.lmd_tilde = lmd_tilde
        # self.omg_tilde = omg_tilde
        # self.bounds = bounds
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
    def compute_first_layer_preac_bounds(x, eps, weights, adj):
        N = x.shape[0]
        l = x - eps
        u = x + eps
        w = weights[0]
        I, J = w.shape
        # TODO: Vectorize
        lt, ut = torch.zeros(N, I, J), torch.zeros(N, I, J)
        for j in range(J):
            lt[:, :, j] = (l * (w[:, j] >= 0).float().repeat(N, 1) + 
                           u * (w[:, j] < 0).float().repeat(N, 1))
            ut[:, :, j] = (u * (w[:, j] >= 0).float().repeat(N, 1) + 
                           l * (w[:, j] < 0).float().repeat(N, 1))
        # TODO: Vectorize
        next_l, next_u = torch.zeros(N, J), torch.zeros(N, J)
        for j in range(J):
            next_l[:, j:j+1] = adj.mm(lt[:, :, j].mm(w[:,j:j+1]))  # One-element slice to keep dimensions
            next_u[:, j:j+1] = adj.mm(ut[:, :, j].mm(w[:,j:j+1]))
        return next_l, next_u

    # Return lower alpha, upper alpha, lower beta, and upper beta.
    # Assumes Relu.
    # Corresponds to CROWN non-linear relaxation bounds
    @staticmethod
    def compute_activation_bound_variables(l, u):
        alpha_u_i = torch.zeros(l.shape) + (l >= 0).float()  # Set proper values for all linear activations
        alpha_l_i = torch.zeros(l.shape) + (l >= 0).float()
        beta_u_i = torch.zeros(l.shape)
        beta_l_i = torch.zeros(l.shape)
        nla = (l < 0) * (u > 0)  # Indexes of all that aren't linear (non-linear activations)
        alpha_u_i[nla] = u[nla] / (u[nla] - l[nla])
        alpha_l_i[nla] = (u[nla] > -l[nla]).float()  # If u > -l, slope is 1. Otherwise 0. These are the CROWN bounds.
        beta_u_i[nla] = -l[nla]  # alpha_u_i[nla] * (-l[nla])
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
    def __init__(self, model, x, adj, eps):
        weights = self.extract_parameters(model)
        l, u = self.compute_first_layer_preac_bounds(x, eps, weights, adj)
        LB = [l]
        UB = [u]
        alpha_u, alpha_l, beta_u, beta_l = [], [], [], []
        for l in range(len(weights)-1):
            # alpha_u, alpha_l, beta_u, beta_l = self.compute_activation_bound_variables(l, u)
            # Lambda, Omega, J_tilde, lmd, omg, delta, theta, lmd_tilde, omg_tilde = self.compute_linear_bound_variables(weights, adj, alpha_u, alpha_l, beta_u, beta_l)
            # bounds = self.compute_bounds(eps, x, Lambda, Omega, J_tilde, lmd, omg, delta, theta)
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
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.weights = weights
        # self.l = l
        # self.u = u
        # self.l_tilde = l_tilde
        # self.u_tilde = u_tilde
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
        # self.lmd_tilde = lmd_tilde
        # self.omg_tilde = omg_tilde
        # self.bounds = bounds
        self.LB = LB
        self.UB = UB

    @staticmethod
    def extract_parameters(model):
        return GCNBoundsRelaxed.extract_parameters(model)

    @staticmethod
    def compute_first_layer_preac_bounds(x, eps, weights, adj):
        return GCNBoundsRelaxed.compute_first_layer_preac_bounds(x, eps, weights, adj)

    @staticmethod
    def compute_activation_bound_variables(l, u):
        return GCNBoundsRelaxed.compute_activation_bound_variables(l, u)
        # alpha_u, alpha_l, beta_u, beta_l = GCNBoundsRelaxed.compute_activation_bound_variables(l, u)
        # Flatten all the variables
        # alpha_u_flat = alpha_u.view(-1)
        # alpha_l_flat = alpha_l.view(-1)
        # beta_u_flat = beta_u.view(-1) 
        # beta_l_flat = beta_l.view(-1) 
        # return alpha_u_flat, alpha_l_flat, beta_u_flat, beta_l_flat

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
            # NI = alpha_u[ind].shape[0]
            w_vec = kronecker(adj, w)  # Massive expanded weight matrix
            NI = w_vec.shape[0]
            # lower case variables
            weight_lambda = w_vec.mm(Lambda[0])
            weight_omega = w_vec.mm(Omega[0])
            # if ind - 1 != 0:
            # print(ind)
            # print(alpha_u[ind].shape)
            # print("weight lambda shape: ", weight_lambda.shape)
            # print("weight lambda: ", weight_lambda[:, 0])
            # print("weight omega shape: ", weight_omega.shape)
            # print("weight omega: ", weight_omega[:, 0])
            # print(len(beta_u))
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
            # Prepend all variables
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
        # print(N, J)
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
        return [t1_l + t2_l + t3_l, t1_u + t2_u + t3_u]
        # return [t1_l, t1_u]


if __name__ == "__main__":
    x = torch.Tensor([[1.0,2.0], [2.0,1.0]])
    weights = [torch.Tensor([[1.0, -1.0], [0.5, -0.6]])]
    adj = torch.Tensor([[2.0, 0.0], [0.0, 1.0]])
    eps = 0.5
    l, u, l_tilde, u_tilde = GCNBoundsFull.compute_preac_bounds(x, eps, weights, adj)
    print(l)
    print(u)

