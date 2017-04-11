from server_problem import ServerProblem,evaluate_server
from algos import get_sklearn_func, capi,get_mlp_learn_func
from utils import onehot,get_fixed_value_func,get_scheduled_value_func
from plot import plot_policy_update_server
import numpy as np
import pickle

env = ServerProblem()
#env = ServerProblem2()

#epsilon_func = get_scheduled_value_func([0,2,5],[1.0,0.5,0.2],"iter_count")
epsilon_func = get_fixed_value_func(0.2)


#pi_eval = 'lstdq'
phi = lambda x,a : np.concatenate([x,onehot(env.num_actions,a)])


pi_eval = 'nfq'
nfqconfig = {'nb_iter': 10,
             'nlayers': 2,
             'nhidden': 10,
             'l1_reg': 1e-5,
             'l2_reg': 1e-5,
             'learning_rate': None}
gamma = 0.95


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.classification import KNeighborsClassifier

#learn = get_sklearn_func(weight_examples=True)
#learn = get_sklearn_func(learner=LDA,weight_examples=False)
learn = get_sklearn_func(learner=DecisionTreeClassifier,weight_examples=True)

learn = get_sklearn_func(learner=KNeighborsClassifier,weight_examples=False)
#learn = get_mlp_learn_func()


label = "nfq+logistic_reg"
label = "nfq+decisiontree"
label = "nfq+KNN"
#label = "nfq+MLP"

pi, train_stats, greedy_stats, it_stats = capi(env,learn,gamma,pi_0=None,num_episodes=200,pi_eval=pi_eval, max_iter_by_episode=200,epsilon_func=epsilon_func,nfq_config=nfqconfig,phi=phi,reuse_feats=False,evaluate_func=evaluate_server,min_iteration=3)

file_name = "server_nfq_logistic_2.pkl"
file_name = "server_nfq_knn_2.pkl"
#file_name = "server_nfq_dectree_3.pkl"
#file_name = "server_nfq_mlp_6.pkl"


results = pickle.dump((train_stats,greedy_stats, it_stats),open(file_name, 'wb'))
plot_policy_update_server(env, (train_stats, greedy_stats,it_stats), label)
