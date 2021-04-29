import sys

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from sklearn import preprocessing
import pandas as pd

from notears.nonlinear import NOTEARS_draw_DAGs


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
    # from notears import utils
    # utils.set_random_seed(1)
    #
    # n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    # B_true = utils.simulate_dag(d, s0, graph_type)
    # W_true = utils.simulate_parameter(B_true)
    # np.savetxt('W_true.csv', W_true, delimiter=',')
    #
    # X = utils.simulate_linear_sem(W_true, n, sem_type)
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

    print('Start...')

    # w_threshold = 0.3  # default
    w_threshold = 0.0
    print('w_threshold: ', w_threshold)

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2', w_threshold=w_threshold)

    # assert utils.is_dag(W_est)

    file_name = 'linear_notears_DAGs_' + str(w_threshold)

    np.savetxt(file_name + '.csv', W_est, delimiter=',')
    # acc = utils.count_accuracy(B_true, W_est != 0)
    # print(acc)

    NOTEARS_draw_DAGs(W_est, file_name, variable_names)
