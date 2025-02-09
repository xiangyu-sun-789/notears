import sys

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
import pandas as pd
from sklearn import preprocessing
from lingam.utils import make_dot


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True, variable_names=None, no_time_inverse_edges=False):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.variable_names = variable_names
        self.no_time_inverse_edges = no_time_inverse_edges

        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

        print('self.variable_names: ', self.variable_names)
        print('self.no_time_inverse_edges: ', self.no_time_inverse_edges)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)

                    if self.no_time_inverse_edges == True:
                        i_time_step = int(self.variable_names[i][-1])
                        j_time_step = int(self.variable_names[j][-1])

                        # restrict the weights to be 0 if it is from a variable at a later step (i) to a variable at an early step (j)
                        if i_time_step > j_time_step:
                            bound = (0, 0)

                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class NotearsSobolev(nn.Module):
    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, self.d - 1)
        # h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for iteration in range(max_iter):
        print('iteration: ', iteration)
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def NOTEARS_draw_DAGs(adjacency_matrix, file_name, variable_names):
    # direction of the adjacency matrix needs to be transposed.
    # in LINGAM, the adjacency matrix is defined as column variable -> row variable
    # in NOTEARS, the W is defined as row variable -> column variable
    dot = make_dot(np.transpose(adjacency_matrix), labels=variable_names)

    dot.format = 'png'
    dot.render(file_name)


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=6)

    # import notears.utils as ut
    # ut.set_random_seed(123)
    #
    # n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    # B_true = ut.simulate_dag(d, s0, graph_type)
    # np.savetxt('W_true.csv', B_true, delimiter=',')
    #
    # X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')

    # ----- Sports data -----

    if len(sys.argv) < 2:
        raise Exception("Specify where the data files are.")

    running_mode = str(sys.argv[1])

    if running_mode == 'local':
        data_directory = '/Users/shawnxys/Desktop/SFU_Vault/preprocessed_sports_data/'
        results_directory = './results/'

    elif running_mode == 'lab':
        data_directory = '/Local-Scratch/shawnxys/SFU_Vault/preprocessed_sports_data/'
        results_directory = '/Local-Scratch/shawnxys/causal_sports_results/results/'

    else:
        raise Exception("Specify where the data files are: {local, compute_canada, lab}")

    features_directory = data_directory + 'features_two_steps.csv'
    action_shots_directory = data_directory + 'actions_shot_two_steps.csv'
    rewards_directory = data_directory + 'rewards_two_steps.csv'

    # load the data
    features_df = pd.read_csv(features_directory)
    actions_shot_df = pd.read_csv(action_shots_directory)
    rewards_df = pd.read_csv(rewards_directory)

    features_shots_rewards_df = pd.concat([features_df, actions_shot_df, rewards_df], axis=1)

    # adjust_home_away
    features_shots_rewards_df = features_shots_rewards_df.drop(columns=['home_1', 'away_1', 'away_2'])

    # adjust_reward
    features_shots_rewards_df = features_shots_rewards_df.drop(columns=['reward_1'])

    # drop all columns other than shot_1, shot_2, reward_2
    # features_shots_rewards_df = features_shots_rewards_df[['shot_1', 'shot_2', 'reward_2']]

    variable_names = [s for s in features_shots_rewards_df.columns]

    # data that will be used to run the algorithm
    X = features_shots_rewards_df.to_numpy()
    # X = features_shots_rewards_df.iloc[:100].to_numpy()

    # data standardization
    scaler = preprocessing.StandardScaler().fit(X)
    print(X.std(axis=0))  # std over columns, https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    X = scaler.transform(X)
    print(X.std(axis=0))

    d = X.shape[1]

    print('Start...')

    # w_threshold = 0.3  # default
    w_threshold = 0.0
    print('w_threshold: ', w_threshold)

    # if add temporal restriction of no time inverse edges
    no_time_inverse_edges = False
    # no_time_inverse_edges = True

    model = NotearsMLP(dims=[d, 10, 1], bias=True, variable_names=variable_names,
                       no_time_inverse_edges=no_time_inverse_edges)
    W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01, w_threshold=w_threshold)

    # assert ut.is_dag(W_est)

    if no_time_inverse_edges:
        file_name = 'nonlinear_notears_DAGs_' + str(w_threshold) + '_prior_knowledge'
    else:
        file_name = 'nonlinear_notears_DAGs_' + str(w_threshold)

    np.savetxt(file_name + '.csv', W_est, delimiter=',')
    # acc = ut.count_accuracy(B_true, W_est != 0)
    # print(acc)

    NOTEARS_draw_DAGs(W_est, file_name, variable_names)


if __name__ == '__main__':
    main()

    print('Done.')
