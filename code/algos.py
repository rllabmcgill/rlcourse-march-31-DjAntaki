import numpy as np
from utils import onehot, uniform_probs,make_samples, evaluate, get_fixed_value_func
from mlp import make_mlp, T, run_mlp, params
from operator import itemgetter
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
def simulate(env,s,a):
    env.current_state = s
    return env.step(a)

def lstdq(D, k, phi, gamma, pi):
    """
    http://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf

    :param D: Source of sample (list of (state, action))
    :param k: Number of basis functions
    :param phi: Basis functions
    :param gamma: Discount factor
    :param pi: Policy whose value function is sought
    :return: the weights that describes our policy
    """
    A = np.zeros((k, k))
    b = np.zeros((k,))

    for (s, a, r, s_p) in D:
        next_best_action = np.argmax(pi(s_p))
        A = A + phi(s, a) * (phi(s, a) - gamma * phi(s_p, next_best_action)).T
        b = b + phi(s, a) * r

    pinv = np.linalg.pinv(A) * b
    w_pi = np.dot(pinv, b)
    return w_pi

def rollout(env,s,a,gamma,pi,K,T):

    Q = np.zeros((K,))

    for k in range(K):
        s_p, r = simulate(env,s,a)
        Q[k], s = r, s_p
        for t in range(1,T):
            s_p, r = simulate(env,s,np.argmax(pi(s)))
            Q[k] += gamma**t * r
            s = s_p
    return np.average(Q)

default_mlp_config = {'nlayers':2,
 'nhidden':100,'l1_reg':1e-6,'l2_reg':1e-6,'learning_rate':1e-5}

def get_mlp_learn_func(mlpconfig=default_mlp_config):
    nlayers, nhidden = mlpconfig['nlayers'], mlpconfig['nhidden']
    L1_reg_coeff, L2_reg_coeff = mlpconfig['l1_reg'], mlpconfig['l2_reg']
    lr = mlpconfig['learning_rate']
    def learn(TS):
        X = np.array(map(lambda x: np.concatenate([x[0], [x[1]]]), TS))
        Y = np.array(map(lambda x: -1 if x[2] == 0 else 1, TS))

        id = uuid.uuid4()
        params.bindNew(id)
        ninput= len(X[0])

        model = make_mlp(ninput, nlayers, nhidden, n_out=1, out_activation=T.tanh)  # ,out_activation=T.nnet.relu)
        forward_pass = run_mlp(X, Y, model, L1_reg_coeff, L2_reg_coeff, learning_rate=lr, uuid=id)
        model.predict = lambda x: forward_pass(x)

        def pi(s, a):
            if type(a) is int:
                return model.predict(np.concatenate([s, [a]]).reshape(1, -1))
            else:
                return model.predict(np.concatenate([s, [a]]))

        # pi = lambda s,a : model.predict(np.concatenate([s,[a]]))
        y_pred = model.predict(X)
        y_pred = list(map(lambda x: -1 if x<0 else 1,y_pred))
        stats = [metrics.accuracy_score(Y, y_pred), metrics.average_precision_score(Y, y_pred), metrics.f1_score(Y,y_pred)]
       # print("training stats")
       # print(stats)
        return pi, stats

    return learn


def get_sklearn_func(learner=LogisticRegression, weight_examples=True,**kwargs):
    def learn(TS):
        X = np.array(map(itemgetter(0),TS))
        Y = np.array(map(itemgetter(1),TS))
        model = learner()
        if weight_examples :
            examples_weights = np.array(map(itemgetter(2),TS))
            model.fit(X, Y, sample_weight=examples_weights)
        else:
            model.fit(X, Y)

        def pi(x):
            if x.ndim == 1:
                return model.predict(x.reshape(1,-1))
            else :
                return model.predict(x)
        #pi = lambda x : model.predict(x)

        y_pred = model.predict(X)
        stats = [metrics.accuracy_score(Y,y_pred),metrics.average_precision_score(Y,y_pred),metrics.f1_score(Y,y_pred)]
        # print("training stats")
        # print(stats)
        return pi,stats

    return learn


def capi(env,learn,gamma,pi_0=None,pi_eval='lstdq',num_episodes=50, max_iter_by_episode=500,epsilon_func=get_fixed_value_func(0.2),min_iteration=0,max_iteration=15,evaluate_func=evaluate,**kwargs):
    """

    :param env:
    :param learn: function that takes in input a binary classification set and .
    :param gamma: Discount factor
    :param max_iter_by_episode : Maximum number of iteration allowed by episode generated
    :param epsilon_func : function that return the epsilon to be used. Is given the iteration count in input.
    :param pi_0: starting policy, uniform over action if None
    :param pi_eval: Determine which policy evaluation procedure to use
    :param max_iteration : The maximum number of policy update.
    :param evaluate_func : Takes a list of episodes in input and compute statistic relevant to the problem.

    Conditional params:
    :param K: if pi_eval == 'rollout', number of trajectories
    :param T: if pi_eval == 'rollout', trajectories length
    :param nfq_config if pi_eval == 'nfq', dictionnary which contains configuration for the nfq function
    :param phi : if pi_eval == 'lstdq' :

    :return:
    pi : policy function
    train_stats :
    greedy_stats : Evaluation of
    it_stats :

    """

    keep_old_consistent_examples = False #Something i have tried, im not sure it helps. It will create a bigger dataset at each iteration.

    nactions = env.num_actions
    iter_count = -1

    # monitored stats will be stored in these variables
    train_stats = []
    greedy_stats = []
    it_stats = []
    similarity_ratio_threshold = 0.98 # If policy is too similar to previous, we will stop optimization.

    # Initializing stuff with regards to policy evaluation step.
    if pi_eval == 'rollout' :
        Q_hat_pi = np.zeros((env.num_states,nactions))
        K, T = kwargs['K'], kwargs['T']
    elif pi_eval == 'nfq':
        nfqconfig = kwargs.get('nfq_config',{})
        reuse_features = kwargs.get('reuse_feats',False)
        nfq_iter = nfqconfig.get('nb_iter',10)
        l1 = nfqconfig.get('l1_reg', 0)
        l2 = nfqconfig.get('l2_reg', 0)
        nfq_nlayers = nfqconfig.get('nlayers',2)
        nhidden = nfqconfig.get('nhidden',10)
        nfq_learning_rate = nfqconfig.get('learning_rate', None)
    elif pi_eval == 'lstdq':
        phi = kwargs['phi']
    else :
        raise NotImplementedError()

    if pi_0 is None:
        pi_prime = lambda x : uniform_probs((nactions,))
    else :
        pi_prime = pi_0

#    Initial greedy evaluation
    print("Generating greedy samples")
    D = make_samples(env, pi_prime, n=num_episodes, max_iter=max_iter_by_episode, pi_policy='greedy')
    print("Finished generating samples.")
    greedy_stats.append(evaluate_func(D))

    kept_D, Old_Q_func = [], None
    forwardpass = None

    while True:
        iter_count += 1
        print 'main loop iter %i'%iter_count
        pi = pi_prime
        TS = [] # Training set, each elements are 4-tuples indicating (state,action,example sign, example weight)

        print("Generating training samples")
        D = make_samples(env, pi, n=num_episodes, max_iter=max_iter_by_episode, pi_policy='egreedy',epsilon=epsilon_func(iter_count=iter_count))
        print("Finished generating samples.")
        train_stats.append(evaluate_func(D))

        Q_func = None
        if pi_eval == 'lstdq':
            D = preprocess_episodes_lspi(D)
            k = len(phi(D[0][0], 0))
            w = lstdq(D, k, phi, gamma, pi)
            Q_func = lambda s : np.array([np.dot(phi(s, a),w) for a in range(nactions)])
        elif pi_eval == 'rollout':
            for d in D:
                s = d[0]
                for a in range(nactions):
                    Q_hat_pi[s][a] = rollout(env, s, a, gamma, pi, K, T)
            Q_func = lambda s: np.array([Q_hat_pi[s][a] for a in range(nactions)])
        elif pi_eval == 'nfq':
            D = preprocess_episodes_nfq(D)
            if reuse_features :
                Q_func, forwardpass = nfq(D,nactions,gamma=gamma,nlayers=nfq_nlayers,nhidden=nhidden, L1_reg_coeff=l1, L2_reg_coeff=l2,n_iter=nfq_iter,learning_rate =nfq_learning_rate,get_last_layer_features=True)
            else :
                Q_func = nfq(D, nactions, gamma=gamma, nlayers=nfq_nlayers, nhidden=nhidden,
                                          L1_reg_coeff=l1, L2_reg_coeff=l2, n_iter=nfq_iter,
                                          learning_rate=nfq_learning_rate,get_last_layer_features=False)


        else :
            raise NotImplementedError()

        #Function that process state-action pair to classifier input
        if pi_eval == 'lstdq':
            make_input = phi
        elif pi_eval == "nfq" and reuse_features :
            raise NotImplementedError()
            make_input = lambda s, a: np.concatenate([np.array(s, dtype=np.float), onehot(nactions, a)])

        else:
            make_input = lambda s, a: np.concatenate([np.array(s, dtype=np.float), onehot(nactions, a)])

        print("Creating dataset for classifer...")
        for d in D:
            s = d[0]

            qvals = Q_func(s)
            a_star = np.argmax(qvals)

            if all([qvals[a] < qvals[a_star] for a in range(nactions) if not a == a_star]):
                TS.append((make_input(s,a_star),1,np.max([qvals[a_star]-qvals])))

            for a in range(nactions):
                if not a == a_star and qvals[a] < qvals[a_star]:
                    TS.append((make_input(s, a), 0, qvals[a_star]-qvals[a]))


        if keep_old_consistent_examples and iter_count > 0:
            c = 0
            for i,d in enumerate(kept_D):
                s = d[0]

                new_qvals = Q_func(s)
                old_qvals = Old_Q_func(s)
                a_star = np.argmax(qvals)

                for a in range(nactions):
                    if not a == a_star and new_qvals[a] < new_qvals[a_star] and old_qvals[a] < old_qvals[a_star]:
                        TS.append((make_input(s, a), 0, qvals[a_star]-qvals[a]))
                if all([new_qvals[a] < new_qvals[a_star] and old_qvals[a] < old_qvals[a_star] for a in range(nactions) if
                        not a == a_star]):
                    TS.append((make_input(s, a_star), 1, np.max([qvals[a_star] - qvals])))
                    c += 1
                else :
                    kept_D.pop(i)
            print('kept %i examples'%c)


        print("Learning action classifier model...")
        predict_func, stats = learn(TS)
        pi_prime = lambda s : np.array([predict_func(make_input(s,a)) for a in range(nactions)])

        print("Generating samples with greedy policy...")
        D2 = make_samples(env, pi_prime, n=num_episodes, max_iter=max_iter_by_episode, pi_policy='greedy')
        print("Finished generating samples.")
        greedy_stats.append(evaluate_func(D2))

        # Evaluating similarity between policy

        same, n = 0, len(D)
        for d in D:
            s,a = d[0], d[1]
            if np.argmax(pi_prime(s)) == np.argmax(pi(s)) :
                same += 1

        similarity_ratio = float(same)/n
        it_stats.append([similarity_ratio])
        print('policy similarity ratio', similarity_ratio)
        if (min_iteration < iter_count and similarity_ratio >= similarity_ratio_threshold) or iter_count > max_iteration:
            break

        if keep_old_consistent_examples:
            kept_D = D

            #kept_D += D
            Old_Q_func = Q_func

    return pi, train_stats, greedy_stats, it_stats

import uuid

def nfq(D,nactions,gamma,nlayers=2,nhidden=100,L1_reg_coeff=0, L2_reg_coeff=0,n_iter=100,learning_rate=None,get_last_layer_features=False):
    """
    #http://ml.informatik.uni-freiburg.de/_media/publications/rieecml05.pdf

    :param D: list of 3-tuple (state,action,reward)
    :param gamma:
    :param nlayers:
    :param nhidden:
    :param n_iter:
    :return:
    """
    k=0

    transform = lambda d: np.concatenate([d[0], onehot(nactions,d[1])])

    ninput = len(transform(D[0]))

    id = uuid.uuid4()
    params.bindNew(id)

    model = make_mlp(ninput, nlayers, nhidden, n_out=1,out_activation=lambda x:x)#,out_activation=T.nnet.relu)

    params.bindNew()

    X = np.array(map(transform, D))

    Q = lambda s : uniform_probs((nactions,))

    forward_pass = None
    while k < n_iter:
        print('nfq iter %i'%k)
        Y = np.array(map(lambda x: x[2] + gamma * np.max(Q(x[0])), D))

        forward_pass = run_mlp(X, Y, model, L1_reg_coeff, L2_reg_coeff, learning_rate=learning_rate, uuid=id)
        if get_last_layer_features:
            Q = lambda s: [forward_pass(transform((s, a))[0][:, None].T) for a in range(nactions)]
        else :
            Q = lambda s : [forward_pass(transform((s,a))[:,None].T) for a in range(nactions)]
        k +=1
        if k == 10 :
            break
    if get_last_layer_features:
        return Q, forward_pass
    else :
        return Q

def preprocess_episodes_lspi(episodes):
    D = []
    for j in range(len(episodes)):
        S,A,R = episodes[j]
        D.append([(S[i], A[i], R[i], S[i + 1]) for i in range(len(S) - 1)])
    D = reduce(lambda x,y:x+y,D,[])
    return D

def preprocess_episodes_nfq(episodes):
    D = []
    for j in range(len(episodes)):
        S,A,R = episodes[j]
        D += [(S[i], A[i], R[i]) for i in range(len(S) - 1)]
    return D



def test():
    from server_problem import ServerProblem
    env = ServerProblem()

    learn = get_sklearn_func()

#    pi_eval = 'lstdq'
    pi_eval = 'nfq'
    capi(env,learn,gamma=0.95, pi_0=None,pi_eval=pi_eval, epsilon=0.2,nfq_iter=2)


def test2():
    from server_problem import ServerProblem, evaluate
    env = ServerProblem()

    learn = get_mlp_learn_func()
#    learn = get_learn_func()

#    pi_eval = 'lstdq'

    nfqconfig = {'nb_iter': 2,
                 'l1_reg': 1e-6,
                 'l2_reg': 1e-6,
                 'learning_rate': None}
    pi_eval = 'nfq'
    capi(env,learn,gamma=0.95, pi_0=None,pi_eval=pi_eval,nfq_config=nfqconfig,evaluate_func=evaluate)




if __name__ == "__main__":
    test2()