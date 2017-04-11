import numpy as np
from numpy.random import beta, poisson, binomial, normal, uniform, chisquare,shuffle

def evaluate(episodes):
    nb_episodes = len(episodes)
    iter_taken = np.zeros((nb_episodes,))
    total_reward = np.zeros((nb_episodes,))
    for n in range(nb_episodes):
        S,A,R = episodes[n]
        iter_taken[n] = len(S) -1
        total_reward[n] = np.sum(R)
    print('average iter taken', np.average(iter_taken))
    print('average reward', np.average(total_reward))
    return iter_taken, total_reward

def evaluate_random_agent(env,n=1000, max_iter=100):
    print("random agent :")
    Ds = make_samples(env, pi=None, n=n, max_iter=max_iter, pi_policy='uniform')
    return evaluate(Ds)

def splitdata(X,Y,percent):
    a,cut = np.array(range(len(X))), int(percent*len(X))
    shuffle(a)
    X_train, Y_train = X[a[:cut]], Y[a[:cut]]
    X_valid, Y_valid = X[a[cut:]], Y[a[cut:]]
    return X_train, Y_train, X_valid, Y_valid

def make_samples(env, pi, n, max_iter=1000, pi_policy='greedy',**kwargs):
    D = []
    for _ in range(n):

#        env.render()

        A, R = [], []
        S = [env.reset()]

        i = 0
        s = S[0]
        while True:
            i += 1
            if pi_policy == 'greedy':
                action = np.argmax(pi(s))
            elif pi_policy == 'sample':
                action = sample(pi(s))
            elif pi_policy == 'uniform':
                action = np.random.choice(range(env.num_actions))
            elif pi_policy == "egreedy":
                action = egreedy_sample(pi(s),kwargs['epsilon'])
            s, reward, isterminal, info= env.step(action)

            A.append(action)
            S.append(s)
            R.append(reward)

            if isterminal or i >= max_iter:
                break
        D.append([S,A,R])
    return D

def movingaverage(values, window):
    #taken from https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


def onehot(num_el,index):
    return np.eye(num_el,num_el)[index]

#
#


def uniform_probs(shape):
    size = np.product(shape)
    return 1.0/size * np.ones(shape)

def softmax(x,temp=1):
    e_x = np.exp(x/temp)
    return e_x / e_x.sum(axis=0)


def sample(weights):
    return np.random.choice(range(len(weights)),p=weights)

def egreedy_sample(policy, epsilon=1e-2):
    """ returns argmax with prob (1-epsilon), else returns a random index"""
    if np.random.binomial(1,1-epsilon) == 1:
        return np.argmax(policy)
    else :
        return np.random.choice(range(len(policy)))

def esoft(weights, epsilon=1e-2,temp=1):
    a_star = np.argmax(weights)
    num_action = len(weights)
    soft_probs = epsilon/num_action * softmax(weights,temp=temp)
    return (1-np.sum(soft_probs))*onehot(num_action,a_star) + soft_probs

#

def get_fixed_value_func(v):
    def get_value(*args, **kwargs):
        return v
    return get_value

def get_scheduled_value_func(times,values,key=None):
    """ times : the scheduled time at which the returned value change.
        values : the values to be returned
        key : if none then looks at first argument

        TODO : less sketchy code"""

    assert sorted(times) == times
    times_len = len(times)
    def get_value(*args, **kwargs):
        t = kwargs[key]
        sched_value_index = np.max(filter(lambda x:t >= times[x],range(times_len)))
        return values[sched_value_index]
    return get_value


def get_beta_func(a, b):
    def f():
        return beta(a, b)
    return f#, float(a) / (a + b)

def get_binomial_func(p, r=1):
    def f():
        return r * binomial(1, p)
    mean = p * r
    return f#, mean

def get_gaussian_func(mu, std):
    def f():
        return normal(mu, std)
    return f#, mu

def get_poisson_func(lam):
    def f():
        return poisson(lam)
    return f#, lam

def get_chisquare_func(df):
    def f():
        return chisquare(df)
    return f#, df

def get_uniform_func(a, b):
    def f():
        return uniform(a, b)
    return f#, (a + b) / 2.0
