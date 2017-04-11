
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
from utils import evaluate_random_agent
from walk import LineWalk
from server_problem import ServerProblem

def plot_policy_update_linewalk(env, results, label, max_iter=50):
    """ results is a tuple (train_results, greedy_results). Both are list of episodes where each episode is a tuple (S,A,R) which are matrix"""
    rdn_avg_iter_taken, rdn_avg_r = map(np.average,evaluate_random_agent(env,max_iter=max_iter))
    good = True
    if good:
        good_avg_iter_taken = env.nb_states -1
        good_avg_r = 100 + good_avg_iter_taken* -1
#    good_avg_iter_taken, _, good_avg_nb_bad_decisions, good_avg_r = map(np.average,evaluate_good_agent(env,max_iter))
    nb_updates = len(results[0])
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    ax1.set_title("Average reward in fct. of # of policy update done")
    ax1.set_xlabel("# of policy update completed")
    ax1.set_ylabel("Reward")

    ax2.set_title("Average nb of iteration taken in function of # of policy updates done")
    ax2.set_xlabel("# of policy update completed")
    ax2.set_ylabel("Avg. iter taken")



    ax1.plot((0, nb_updates-1), (rdn_avg_r, rdn_avg_r ), 'r-',label='random')
    ax2.plot((0, nb_updates-1), (rdn_avg_iter_taken, rdn_avg_iter_taken), 'r-',label='random')

    if good :
        ax1.plot((0, nb_updates-1), (good_avg_r, good_avg_r), 'g-',label='optimal policy')
        ax2.plot((0, nb_updates-1), (good_avg_iter_taken, good_avg_iter_taken), 'g-', label='optimal policy')

    xpoints = xrange(nb_updates)
    xpoints2 = xrange(nb_updates)
    train_results, greedy_results, it_results = results

    print(nb_updates)
    ax1.plot(xpoints, map(np.average,map(itemgetter(1),train_results)), 'k-',label=label)
    ax1.plot(xpoints2, map(np.average,map(itemgetter(1),greedy_results[:-1])) ,'k:',label="greedy eval "+label)
    ax2.plot(xpoints, map(np.average,map(itemgetter(0),train_results)), 'k-',label=label)
    ax2.plot(xpoints2, map(np.average,map(itemgetter(0),greedy_results[:-1])) ,'k:',label="greedy eval "+label)


    ax3.set_title("Policy similarity in function of # of policy updates done")
    ax3.set_xlabel("# of policy update completed")
    ax3.set_ylabel("Policy similarity")

    ax3.plot(xpoints2,[0] + map(itemgetter(0),it_results[:-1]))



    legend = ax1.legend(loc='lower right', shadow=True)
    legend = ax2.legend(loc='right', shadow=True)


    plt.show()

def plot_policy_update_server(env, results, label, max_iter=200):
    """ results is a tuple (train_results, greedy_results). Both are list of episodes where each episode is a tuple (S,A,R) which are matrix"""
    from server_problem import evaluate_good_agent_server,evaluate_random_agent_server

    rdn_avg_iter_taken, _, rdn_avg_nb_bad_decisions, rdn_avg_r = map(np.average, evaluate_random_agent_server(env, max_iter=max_iter))
    good_avg_iter_taken, _, good_avg_nb_bad_decisions, good_avg_r = map(np.average, evaluate_good_agent_server(env, max_iter=max_iter))
    nb_updates = len(results[0])
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    ax1.set_title("Average reward in fct. of # of policy update done")
    ax1.set_xlabel("# of policy update completed")
    ax1.set_ylabel("Reward")

#    ax2.set_title("Average nb of iteration taken in function of # of policy updates done")
#    ax2.set_xlabel("# of policy update completed")
    ax2.set_title("Average nb of tasks discarded in function of # of policy updates done")
    ax2.set_xlabel("# of policy update completed")

    ax2.set_ylabel("Avg. nb of tasks discarded")



    ax1.plot((0, nb_updates), (rdn_avg_r, rdn_avg_r ), 'r-',label='random')
    ax1.plot((0, nb_updates), (good_avg_r, good_avg_r), 'g-',label='optimal policy')

#    ax2.plot((0, nb_updates), (rdn_avg_iter_taken, rdn_avg_iter_taken), 'r-',label='random')
#    ax2.plot((0, nb_updates), (good_avg_iter_taken, good_avg_iter_taken), 'g-', label='optimal policy')

    ax2.plot((0, nb_updates), (rdn_avg_nb_bad_decisions, rdn_avg_nb_bad_decisions), 'r-',label='random')
    ax2.plot((0, nb_updates), (good_avg_nb_bad_decisions, good_avg_nb_bad_decisions), 'g-', label='optimal policy')


    xpoints = xrange(nb_updates)
    xpoints2 = xrange(nb_updates+1)
    train_results, greedy_results, it_results = results

    print(nb_updates)
    ax1.plot(xpoints, map(np.average,map(itemgetter(3),train_results)), 'k-',label=label)
    ax1.plot(xpoints2, map(np.average,map(itemgetter(3),greedy_results)) ,'k:',label="greedy eval "+label)

#    ax2.plot(xpoints, map(np.average,map(itemgetter(0),train_results)), 'k-',label=label)
#    ax2.plot(xpoints2, map(np.average,map(itemgetter(0),greedy_results)) ,'k:',label="greedy eval "+label)
    ax2.plot(xpoints, map(np.average, map(itemgetter(2), train_results)), 'k-', label=label)
    ax2.plot(xpoints2, map(np.average,map(itemgetter(2),greedy_results)) ,'k:',label="greedy eval "+label)


    ax3.set_title("Policy similarity in function of # of policy updates done")
    ax3.set_xlabel("# of policy update completed")
    ax3.set_ylabel("Policy similarity")

    ax3.plot(xpoints2,[0] + map(itemgetter(0),it_results))

    legend = ax1.legend(loc='lower right', shadow=True)
    legend = ax2.legend(loc='right', shadow=True)


    plt.show()


if __name__=="__main__":
    import pickle

    if True :
        file_name = "linewalk_nfq_dectree.pkl"
        results = pickle.load(open(file_name,'rb'))
        env = LineWalk(10)
        plot_policy_update_linewalk(env, results, "capi (nfq-dectree)")
    else :
        file_name = "server_nfq_dectree_1.pkl"
        results = pickle.load(open(file_name,'rb'))
        env = ServerProblem()
        plot_policy_update_server(env, results, "server problem")