from utils import onehot, get_scheduled_value_func,get_fixed_value_func, evaluate
from walk import LineWalk
from algos import capi, get_sklearn_func, default_mlp_config
import pickle
from plot import plot_policy_update_linewalk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import svm
import numpy as np

nb_state = 10
gamma = 0.99

env = LineWalk(nb_state)

#epsilon_func = get_scheduled_value_func([0,2,5],[1.0,0.5,0.2],"iter_count")
epsilon_func = get_fixed_value_func(0.2)

#pi_eval='lstdq'
#phi = lambda x,a : np.concatenate([onehot(env.nb_states,x[0]),onehot(env.num_actions,a)])
#phi = lambda x, y : np.concatenate([onehot(nb_state,x), onehot(2,y)])

pi_eval,phi ='nfq',None
nfqconfig=default_mlp_config
nfqconfig = {'nb_iter': 10,
             'nlayers': 0,
             'nhidden': 10,
             'l1_reg': 1e-5,
             'l2_reg': 1e-5,
             'learning_rate': None}

from sklearn.neighbors.classification import KNeighborsClassifier

if True :
    labels = ['test']
    saves = 'test.pkl'
    learns = [get_sklearn_func(learner=KNeighborsClassifier,weight_examples=False)]
else :
    learn0 = get_sklearn_func(weight_examples=True)
    learn1 = get_sklearn_func(learner=LDA,weight_examples=False)
    learn2 = get_sklearn_func(learner=DecisionTreeClassifier,weight_examples=False)
    learn3 = get_sklearn_func(learner=KNeighborsClassifier,weight_examples=False)
    labels = ["nfq+logistic_reg","nfq+lda","nfq+dectree","nfq+KNN"]
    saves = ["linewalk_nfq_logistic_8.pkl","linewalk_nfq_lda_1.pkl","linewalk_nfq_dectree.pkl","linewalk_nfq_knn.pkl"]
    learns = [learn0,learn1,learn2,learn3]

for lbl,file_name,learn in zip(labels,saves,learns):

    pi, train_stats, greedy_stats, it_stats = capi(env,learn,gamma,pi_0=None,pi_eval=pi_eval, num_episodes=5, max_iter_by_episode=50,epsilon_func=epsilon_func,nfq_config=nfqconfig,phi=phi,min_iteration=3,max_iteration=5)
    results = pickle.dump((train_stats,greedy_stats, it_stats),open(file_name, 'wb'))
    plot_policy_update_linewalk(env, (train_stats, greedy_stats,it_stats), lbl)