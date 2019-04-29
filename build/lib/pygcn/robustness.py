# For now assume the model is all GCN layers, all with Relu except the last.
# We'll also just do l-inf norm for the sake of convenience.
# See paper for variable definitions.

import torch
import torch.nn.functional as F

# from pygcn.models import GCN
from pygcn.layers import GraphConvolution

class GCNBounds():
    # This class mostly serves as an automatic run and storage for the bounds. All methods
    # are static and can be used independently of a class instance.
    # Assumes the adjacency matrix has been normalized and had the identity added.
    def __init__(self, model, x, adj, eps):
        weights = self.extract_parameters(model)
        print("compute_preac_bounds")
        l, u, l_tilde, u_tilde = self.compute_preac_bounds(x, eps, weights, adj)
        print("compute_activation_bound_variables")
        alpha_u, alpha_l, beta_u, beta_l = self.compute_activation_bound_variables(l, u)
        print("compute_linear_bound_variables")
        Lambda, Omega, J_tilde, lmd, omg, delta, theta, lmd_tilde, omg_tilde = self.compute_linear_bound_variables(weights, adj, alpha_u, alpha_l, beta_u, beta_l)
        print("compute_bounds")
        bounds = self.compute_bounds(eps, x, Lambda, Omega, J_tilde, lmd, omg, delta, theta)

        # Store variables
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.weights = weights
        self.l = l
        self.u = u
        self.l_tilde = l_tilde
        self.u_tilde = u_tilde
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
        self.lmd_tilde = lmd_tilde
        self.omg_tilde = omg_tilde
        self.bounds = bounds

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
    def compute_preac_bounds(x, eps, weights, adj):
        N = x.shape[0]
        l = [x - eps]
        u = [x + eps]
        l_tilde = []
        u_tilde = []
        for ind, w in enumerate(weights):
            I, J = w.shape
            # TODO: Vectorize
            lt, ut = torch.zeros(N, I, J), torch.zeros(N, I, J)
            for j in range(J):
                lt[:, :, j] = (l[ind] * (w[:, j] >= 0).float().repeat(N, 1) + 
                               u[ind] * (w[:, j] < 0).float().repeat(N, 1))
                ut[:, :, j] = (u[ind] * (w[:, j] >= 0).float().repeat(N, 1) + 
                               l[ind] * (w[:, j] < 0).float().repeat(N, 1))
            # TODO: Vectorize
            next_l, next_u = torch.zeros(N, J), torch.zeros(N, J)
            if ind == 0:
                for j in range(J):
                    # print(w[:,j].shape)
                    # print(lt[:, :, j].shape)
                    # print(lt[:, :, j].mm(w[:,j]).shape)
                    next_l[:, j:j+1] = adj.mm(lt[:, :, j].mm(w[:,j:j+1]))  # One-element slice to keep dimensions
                    next_u[:, j:j+1] = adj.mm(ut[:, :, j].mm(w[:,j:j+1]))
            else:
                for j in range(J):
                    next_l[:, j:j+1] = adj.mm(F.relu(lt[:, :, j]).mm(w[:,j:j+1]))
                    next_u[:, j:j+1] = adj.mm(F.relu(ut[:, :, j]).mm(w[:,j:j+1]))
            l.append(next_l)
            u.append(next_u)
            l_tilde.append(lt)
            u_tilde.append(ut)
        return l, u, l_tilde, u_tilde

    # Return lower alpha, upper alpha, lower beta, and upper beta.
    # Assumes Relu.
    # Corresponds to CROWN non-linear relaxation bounds
    @staticmethod
    def compute_activation_bound_variables(l, u):
        alpha_u = []
        alpha_l = []
        beta_u = []
        beta_l = []
        for l, u in zip(l, u):
            alpha_u_i = torch.zeros(l.shape) + (l >= 0).float()  # Set proper values for all linear activations
            alpha_l_i = torch.zeros(l.shape) + (l >= 0).float()
            beta_u_i = torch.zeros(l.shape)
            beta_l_i = torch.zeros(l.shape)
            nla = (l < 0) * (u > 0)  # Indexes of all that aren't linear (non-linear activations)
            alpha_u_i[nla] = u[nla] / (u[nla] - l[nla])
            alpha_l_i[nla] = (u[nla] > -l[nla]).float()  # If u > -l, slope is 1. Otherwise 0. These are the CROWN bounds.
            beta_u_i[nla] = alpha_u_i[nla] * (-l[nla])
            # beta_l_i always remains 0

            # Store layer variables
            alpha_u.append(alpha_u_i)
            alpha_l.append(alpha_l_i)
            beta_u.append(beta_u_i)
            beta_l.append(beta_l_i)
        return alpha_u, alpha_l, beta_u, beta_l

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
            ind += 1  # alphas and betas are have an extra 0 element.
            # small lambda, small omega, delta, theta, and tilde lambda + omega
            I = alpha_u[ind].shape[1]  # Also Lambda[0][0]
            lmd_l, omg_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
            delta_l, theta_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
            lmb_tilde_l, omg_tilde_l = torch.zeros(I, J), torch.zeros(I, J)
            # TODO: Vectorize
            # for i in range(I):
            #     for j in range(J):
            #         for n in range(N):
            #             lmd_l[n, i, j] = alpha_u[ind][n, i] if Lambda[0][i, j] >= 0 else alpha_l[ind][n, i]
            #             omg_l[n, i, j] = alpha_l[ind][n, i] if Omega[0][i, j] >= 0 else alpha_u[ind][n, i]
            #             delta_l[n, i, j] = beta_u[ind][n, i] if Lambda[0][i, j] >= 0 else beta_l[ind][n, i]
            #             theta_l[n, i, j] = beta_l[ind][n, i] if Omega[0][i, j] >= 0 else beta_u[ind][n, i]
            #         lmb_tilde_l[n, i, j] = torch.max(alpha_u[ind][:, i]) if Lambda[0][i, j] >= 0 else torch.min(alpha_l[ind][:, i])
            #         omg_tilde_l[n, i, j] = torch.min(alpha_l[ind][:, i]) if Omega[0][i, j] >= 0 else torch.max(alpha_u[ind][:, i])
            for j in range(J):
                # print(ind)
                # print(len(alpha_u))
                # print((Lambda[0][:, j] >= 0).float())
                # print(torch.max(alpha_u[ind], 0))
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
        return Lambda, Omega, J_tilde, lmd, omg, delta, theta, lmb_tilde, omg_tilde

    # Return global lower and upper bounds for each data point.
    # Corresponds to corollary 4.3
    @staticmethod
    def compute_bounds(eps, xo, Lambda, Omega, J_tilde, lmd, omg, delta, theta):
        N, J = J_tilde[0].shape[0], Lambda[0].shape[1]
        Lambda_1 = Lambda[0]
        Omega_1 = Omega[0]
        J_1 = J_tilde[0]
        # first term, constant
        t1_u = eps * torch.sum(torch.abs(Lambda_1))
        t1_l = -eps * torch.sum(torch.abs(Omega_1))
        # second term, [n,j] matrix
        t2_u = J_1.mm(xo.mm(Lambda_1))
        t2_l = J_1.mm(xo.mm(Omega_1))
        # third term, [n,j] matrix
        t3_u, t3_l = torch.zeros(N, J), torch.zeros(N, J)
        # TODO: Vectorize
        for i in range(len(J_tilde)-1):
            for j in range(J):
                # print((lmd[i] * delta[i]).shape)
                # print(Lambda[i+1].shape)
                t3_u[:, j:j+1] += J_tilde[i+1].mm((lmd[i][:, :, j] * delta[i][:, :, j]).mm(Lambda[i+1][:, j:j+1]))
                t3_l[:, j:j+1] += J_tilde[i+1].mm((omg[i][:, :, j] * theta[i][:, :, j]).mm(Omega[i+1][:, j:j+1]))
        return [t1_l + t2_l + t3_l, t1_u + t2_u + t3_u]