from server_problem import *
from algos import nfq,preprocess_episodes_nfq
from utils import make_samples
from plot import plot_policy_update_server


env = ServerProblem()
#env = ServerProblem2()
gamma = 0.95

def nfq_main(env,nb_iter,max_iter_by_episode=500):
    k = 0
    nactions = env.num_actions
    pi = None
    pi_policy = 'uniform'

    train_results, greedy_results = [], []

    while k < nb_iter:
        D = make_samples(env, pi, n=50, max_iter=max_iter_by_episode, pi_policy=pi_policy, epsilon=0.2)

        train_results.append(evaluate(D))
        D = preprocess_episodes_nfq(D)

        pi = nfq(D,nactions,gamma,nlayers=2,nhidden=5,L1_reg_coeff=1e-5, L2_reg_coeff=1e-5,n_iter=15,learning_rate=None)
        pi_policy = 'egreedy'

        #Evaluating greedy policy
        D = make_samples(env, pi, n=50, max_iter=max_iter_by_episode, pi_policy='greedy')
        greedy_results.append(evaluate(D))
        k += 1

    return train_results,greedy_results

results = nfq_main(env,3)

plot_policy_update_server(env,results, 'nfqmain')