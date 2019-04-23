# For now assume the model is all GCN layers, all with Relu except the last.
# We'll also just do l-inf norm for the sake of convenience.
# See paper for variable definitions.

import torch.nn.functional as F

from pygcn.models import GCN

class GcnBounds():
    # This class mostly serves as an automatic run and storage for the bounds. All methods
    # are static and can be used independently of a class instance.
    # Assumes the adjacency matrix has been normalized and had the identity added.
    def __init__(self, model, x, adj, eps):
        weights = self.extract_parameters(model)
        l, u, l_tilde, u_tilde = self.compute_preac_bounds(x, eps, weights, adj)
        lmd, omg, delta, theta = self.compute_activation_bound_variables(l, u)
        Lamda, Omega, J, lmd_tilde, omg_tilde = self.compute_bound_variables(weights, mod_adj)
        bounds = self.compute_bounds(lmd, omg, Lamda, Omega, J)

        # Store variables
        self.model = model
        self.x = x
        self.adj = adj
        self.eps = eps
        self.weights = weights
        self.l = l
        self.u = u
        self.lmd = lmd
        self.omg = omg
        self.delta = delta
        self.theta = theta
        self.Lamda = Lamda
        self.Omega = Omega
        self.J = J
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

    # Return l and u for each neuron preactivation
    @staticmethod
    def compute_preac_bounds(self, x, eps, weights, adj):
        n = x.shape[0]
        l = [x - eps]
        u = [x + eps]
        l_tilde = []
        u_tilde = []
        for ind, w in enumerate(weights):
            i, j = w.shape
            # TODO: Vectorize
            lt, ut = torch.zeros((n, i, j)), torch.zeros((n, i, j))
            for nn in range(n):
                for ii in range(i)
                    for jj in range(j):
                        lt[n, i, j] = l[ind][n, i] if w[i, j] >= 0 else u[ind][n, i]
                        ut[n, i, j] = u[ind][n, i] if w[i, j] >= 0 else l[ind][n, i]
            if ind == 0:
                next_l = adj.mm(lt.mm(w))
                next_u = adj.mm(ut.mm(w))
            else:
                next_l = adj.mm(F.relu(lt).mm(w))
                next_u = adj.mm(F.relu(ut).mm(w))
            l.append(next_l)
            u.append(next_u)
            l_tilde.append(lt)
            u_tilde.append(ut)
        return l, u, l_tilde, u_tilde

    # Return small lambda, small omega, delta, and theta
    # Assumes Relu.
    @staticmethod
    def compute_activation_bound_variables(self, l, u):
        lmd = []
        omg = []
        delta = []
        theta = []
        for l, u in zip(l, u):
            pass
        return None, None, None, None, None

    # Return small lambda, small omega, big lambda, big omega, and the J's.
    @staticmethod
    def compute_linear_bound_variables(self, weights, adj):
        return None, None, None, None, None

    # Return global lower and upper bounds for each data point.
    @staticmethod
    def compute_bounds(self, lmd, omg, Lamda, Omega, J):
        return None