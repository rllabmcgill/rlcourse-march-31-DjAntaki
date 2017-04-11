# I got lazy and a significant part of run_mlp function comes from : https://github.com/mfaruqui/dwl
#

from models import *
from rprop import rprop_plus_updates
import sys, time
import numpy as np
from theano import tensor as T
from utils import splitdata
import os
from operator import itemgetter

def make_mlp(ninput,nlayers, nhidden, n_out, hidden_activation=T.nnet.relu, out_activation=T.nnet.softmax):

    nins = [ninput] + [nhidden] * nlayers
    model = StackModel(
        [HiddenLayer(nin, nout, hidden_activation, init="uniform", name='_t%d'%i)
         for i,nin,nout in zip(range(len(nins)),nins[:-1], nins[1:])] +
        [HiddenLayer(nins[-1], n_out, out_activation, name='output')])
    sys.stderr.write("\nNumber of nodes:-\n  Input: {0}\n  Hidden: {1}\n  Output: {2}\n".format(ninput, nhidden, n_out))
    return model

def run_mlp(X,Y, model, L1_reg_coeff, L2_reg_coeff, error='leastsquare', learning_rate=None,n_epochs=70,uuid=None,get_last_layer=False):
    """
    run full batch update with RProp return the trained model. Keeps 20% for validation.
    """
    print(X)
    X, Y = np.array(X,dtype='float32'), np.array(Y,dtype='float32')

    X_train, Y_train, X_valid, Y_valid = splitdata(X,Y,0.8)

    x = T.matrix('x', dtype='float32')
    y = T.vector('y', dtype='float32')


    out, varlist = model.apply(x)


    if error == 'leastsquare':
        L = T.sum((out.T - y)**2).mean()
    elif error == 'x-entropy':
        L = -T.sum(y.T * T.log(out) + (1 - y.T) * T.log(1 - out), axis=1).mean()
    else :
        raise NotImplementedError()

    l1reg = T.sum(abs(T.concatenate([l.W.flatten() for l in model.layers])))
    l2reg = T.sum((T.concatenate([l.W.flatten() for l in model.layers]))**2)

    cost = L + L1_reg_coeff * l1reg + L2_reg_coeff * l2reg
    #print('out',out.eval({x:np.ones((100,13),dtype=np.float32)}).shape)
    #print(np.zeros(100,dtype=np.float32).shape)
    #print(L.eval({x:np.ones((100,13),dtype=np.float32),y:np.zeros(100,dtype=np.float32)}))
    #print(l1reg.eval())#{x:np.ones((100,13),dtype=np.float32)}))
    #print(l2reg.eval())#{x:np.ones((100,13),dtype=np.float32)}))

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    if uuid is None:
        param_list = params.param_list
    else :
        param_list = params.param_groups[uuid]

    gparams = T.grad(cost, param_list)
    gparams = map(lambda x : T.cast(x,dtype='float32'),gparams)
#    gparams = [T.grad(cost, param) for param in params.param_list]

#    print('gparam',gparams[0].eval({x:np.ones((100,13),dtype=np.float32),y:np.zeros(100,dtype=np.float32)}).dtype)
#    print(params.param_list[0].dtype)
    if learning_rate is not None:
        # SGD
        sys.stderr.write('\nUsing SGD with learning rate: {0}\n'.format(learning_rate))
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(param_list, gparams)]
    else:
        # RPROP+
        sys.stderr.write('\nUsing RPROP+...\n')
        updates = [(param, param_updated) for (param, param_updated) in rprop_plus_updates(param_list, gparams)]

    sys.stderr.write('... training\n')

    # early-stopping parameters
    patience = 10 # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                      # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0

    train_func = theano.function(inputs=[x,y], outputs=[cost]+gparams+map(itemgetter(1),updates), updates=updates,allow_input_downcast=True)
    valid_func = theano.function(inputs=[x,y], outputs=cost)


    avg_cost = valid_func(X_train,Y_train)
    valid_loss = valid_func(X_valid,Y_valid)

    print >> sys.stderr, ('epoch %i, train cross entropy %f validation cross entropy %f' %
                 (epoch, avg_cost, valid_loss))

    while (epoch < n_epochs):
        epoch = epoch + 1
        iter = epoch

        o = train_func(X_train,Y_train)
        valid_loss = valid_func(X_valid, Y_valid)

        avg_cost = o[0]

        if learning_rate is None:
            print >> sys.stderr, ('epoch %i, train cross entropy %f validation cross entropy %f' %(epoch, avg_cost, valid_loss))

        else :
            print >> sys.stderr, ('epoch %i, train cross entropy %f validation cross entropy %f learning_rate %f' %
                 (epoch, avg_cost, valid_loss,learning_rate))

        # if we got the best validation score until now
        if valid_loss < best_validation_loss:
            #improve patience if loss improvement is good enough
            if valid_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, iter + patience_increase)

            best_validation_loss = valid_loss
            best_iter = iter
            #params.exportToFile("best.pkl")

        if patience <= iter:
            sys.stderr.write("\nEarly stopping...\n")
            break

    end_time = time.clock()
    print >> sys.stderr, (('Optimization complete. Best validation score of %f %% obtained at iteration %i, epochs ran %i ') %
              (best_validation_loss, best_iter + 1, epoch))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.))

    if get_last_layer:
        print(varlist[-2])
        return theano.function([x],[out.flatten(), varlist[-2]],allow_input_downcast=True) #model
    else :
        return theano.function([x],out.flatten(),allow_input_downcast=True) #model