# For now assume the model is all GCN layers, all with Relu except the last.
# We'll also just do l-inf norm for the sake of convenience.
# See paper for variable definitions.

import torch.nn.functional as F

from pygcn.models import GCN

class GCNBounds():
    # This class mostly serves as an automatic run and storage for the bounds. All methods
    # are static and can be used independently of a class instance.
    # Assumes the adjacency matrix has been normalized and had the identity added.
    def __init__(self, model, x, adj, eps):
        weights = self.extract_parameters(model)
        l, u, l_tilde, u_tilde = self.compute_preac_bounds(x, eps, weights, adj)
        alpha_u, alpha_l, beta_u, beta_l = self.compute_activation_bound_variables(l, u)
        Lamda, Omega, J, lmd, omg, delta, theta, lmd_tilde, omg_tilde = self.compute_bound_variables(weights, adj, alpha_u, alpha_l, beta_u, beta_l)
        bounds = self.compute_bounds(lmd, omg, Lamda, Omega, J)

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
        self.Lamda = Lamda
        self.Omega = Omega
        self.J = J
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
    def extract_parameters(self, model):
        weights = []
        for layer in model:
            if isinstance(layer, GCN):
                weights.append(layer.weight)
            else:
                raise ValueError("Only GCN layers supported.")
        return weights 

    # Return l and u for each neuron preactivation. Also returns tilde variables for debug purposes.
    # Corresponds to theorem 3.1
    @staticmethod
    def compute_preac_bounds(self, x, eps, weights, adj):
        N = x.shape[0]
        l = [x - eps]
        u = [x + eps]
        l_tilde = []
        u_tilde = []
        for ind, w in enumerate(weights):
            I, J = w.shape
            # TODO: Vectorize
            lt, ut = torch.zeros(N, I, J), torch.zeros(N, I, J)
            for n in range(N):
                for i in range(I)
                    for j in range(J):
                        lt[n, i, j] = l[ind][n, i] if w[i, j] >= 0 else u[ind][n, i]
                        ut[n, i, j] = u[ind][n, i] if w[i, j] >= 0 else l[ind][n, i]
            # TODO: Vectorize
            next_l, next_u = torch.zeros(N, J), torch.zeros(N, J)
            if ind == 0:
                for j in range(J):
                    next_l[:, j] = adj.mm(lt[j].mm(w))
                    next_u[:, j] = adj.mm(ut[j].mm(w))
            else:
                for j in range(J):
                    next_l[:, j] = adj.mm(F.relu(lt[j]).mm(w))
                    next_u[:, j] = adj.mm(F.relu(ut[j]).mm(w))
            l.append(next_l)
            u.append(next_u)
            l_tilde.append(lt)
            u_tilde.append(ut)
        return l, u, l_tilde, u_tilde

    # Return lower alpha, upper alpha, lower beta, and upper beta.
    # Assumes Relu.
    # Corresponds to CROWN non-linear relaxation bounds
    @staticmethod
    def compute_activation_bound_variables(self, l, u):
        alpha_u = []
        alpha_l = []
        beta_u = []
        beta_l = []
        for l, u in zip(l, u):
            alpha_u_i = torch.zeros(l.shape) + (l >= 0)  # Set proper values for all linear activations
            alpha_l_i = torch.zeros(l.shape) + (l >= 0)
            beta_u_i = torch.zeros(l.shape)
            beta_l_i = torch.zeros(l.shape)
            nla = (l < 0) * (u > 0)  # Indexes of all that aren't linear (non-linear activations)
            alpha_u_i[nla] = u[nla] / (u[nla] - l[nla])
            alpha_l_i[nla] = u[nla] > -l[nla]  # If u > -l, slope is 1. Otherwise 0. These are the CROWN bounds.
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
    def compute_linear_bound_variables(self, weights, adj, alpha_u, alpha_l, beta_u, beta_l):
        N = adj.shape[0]
        Lambda = [weights[-1]]
        Omega = [weights[-1]]
        J = [adj]
        lmd = []
        omg = []
        delta = []
        theta = []
        lmb_tilde = []
        omg_tilde = []
        for ind, w in reversed(enumerate(weights[:-1])):
            # J
            j_l = adj.mm(J[0])
            # small lambda, small omega, delta, and theta
            I, J = Lambda[0].shape
            lmd_l, omg_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
            delta_l, theta_l = torch.zeros(N, I, J), torch.zeros(N, I, J)
            lmb_tilde_l, omg_tilde_l = torch.zeros(I, J), torch.zeros(I, J)
            # TODO: Vectorize
            for n in range(N):
                for i in range(I)
                    for j in range(J):
                        lmd_l[n, i, j] = l[ind][n, i] if w[i, j] >= 0 else u[ind][n, i]
                        ut[n, i, j] = u[ind][n, i] if w[i, j] >= 0 else l[ind][n, i]
        return Lambda, Omega, J, lmd, omg, delta, theta, lmb_tilde, omg_tilde

    # Return global lower and upper bounds for each data point.
    # Corresponds to corollary 4.3
    @staticmethod
    def compute_bounds(self, lmd, omg, Lamda, Omega, J):
        return None