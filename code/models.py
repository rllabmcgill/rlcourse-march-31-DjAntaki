import numpy
import theano
import pickle
T = theano.tensor

def gradient_descent(params, grads, lr):
    up = [(i, i - lr * gi) for i, gi in zip(params, grads)]
    return up

class SharedGenerator:
    count = 0
    def __init__(self):
        self.reset()
        self.init_tensor_x = T.scalar()
        self.init_minibatch_x = numpy.float32(0)
        self.isJustReloadingModel = False

    def reset(self):
        self.param_list = [] # currently bound list of parameters
        self.param_groups = {} # groups of parameters
        self.param_costs = {} # each group can have attached costs
        self.bind(self.param_list)

    def bind(self, params, name="default"):
        if type(params)==str:
            self.param_list = self.param_groups[params]
            return
        self.param_list = params
        self.param_groups[name] = params
        if name not in self.param_costs:
            self.param_costs[name] = []
    def bindNew(self, name='default'):
        p = []
        self.bind(p, name)
        return p

    def __call__(self, name, shape, init='uniform', **kwargs):
        #print("init",name,shape,init,kwargs)
        if type(init).__module__ == numpy.__name__: # wtf numpy
            values = init
        elif init == "uniform" or init == "glorot" or init == 'tanh':
            k = numpy.sqrt(6./numpy.sum(shape)) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "bengio" or init == 'relu':
            p = kwargs['inputDropout'] if 'inputDropout' in kwargs and kwargs['inputDropout'] else 1
            k = numpy.sqrt(6.*p/shape[0]) if 'k' not in kwargs else kwargs['k']
            values = numpy.random.uniform(-k,k,shape)
        elif init == "one":
            values = numpy.ones(shape)
        elif init == "zero":
            values = numpy.zeros(shape)
        elif init == 'ortho':
            def sym(w):
                import numpy.linalg
                from scipy.linalg import sqrtm, inv
                return w.dot(inv(sqrtm(w.T.dot(w))))
            values = numpy.random.normal(0,1,shape)
            values = sym(values).real

        else:
            print(type(init))
            raise ValueError(init)
        s = theano.shared(numpy.float32(values), name=name)
        self.param_list.append(s)

        s.uid = name + '_'+str(self.count)
        self.count += 1
        return s

    def exportToFile(self, path):
        exp = {}
        for g in self.param_groups:
            exp[g] = [i.get_value() for i in self.param_groups[g]]
        pickle.dump(exp, open(path,'wb'), -1)

    def importFromFile(self, path):
        exp = pickle.load(open(path,'rb'))
        for g in exp:
            for i in range(len(exp[g])):
                print(g, exp[g][i].shape)
                self.param_groups[g][i].set_value(exp[g][i])
    def attach_cost(self, name, cost):
        self.param_costs[name].append(cost)
    def get_costs(self, name):
        return self.param_costs[name]
    def get_all_costs(self):
        return [j  for i in self.param_costs for j in self.param_costs[i]]
    def get_all_names(self):
        print([i for i in self.param_costs])
        print(self.param_costs.keys())
        return self.param_costs.keys()

    def computeUpdates(self, lr, gradient_method=gradient_descent):
        updates = []
        for i in self.param_costs:
            updates += self.computeUpdatesFor(i, lr, gradient_method)
        return updates

    def computeUpdatesFor(self, name, lr, gradient_method=gradient_descent):
        if name not in self.param_costs or \
           not len(self.param_costs[name]):
            return []
        cost = sum(self.param_costs[name])
        grads = T.grad(cost, self.param_groups[name])
        updates = gradient_method(self.param_groups[name], grads, lr)

        return updates


shared = SharedGenerator()
params = shared

class HiddenLayer:
    def __init__(self, n_in, n_out, activation, init="uniform", name=''):
        self.W = shared("W"+name, (n_in, n_out), init)
        self.b = shared("b"+name, (n_out,), "zero")
        self.activation = activation

    def __call__(self, x, *args):
        return self.activation(T.dot(x,self.W) + self.b)

    def apply(self, x, *args):
        if isinstance(x,Var):
            x = x.val
        return Var(val=self.activation(T.dot(x,self.W) + self.b))


class Var:
    def __init__(self, **kwargs):
        self.dict = kwargs
    def __getitem__(self, k):
        return self.dict[k]
    def __getattr__(self, k):
        return self.dict[k]
    def get(self, k, v):
        return self.dict.get(k,v)
    def getVal(self, k, v):
        return self.dict[k].val if k in self.dict else v
    def __repr__(self):
        return 'Var<%s>'%str(self.dict)
    def __str__(self):
        return 'Var<%s>'%str(self.dict)
#    def __coerce__(self, other):
#        return None

class VarList:
    def __init__(self, lst):
        self.lst = lst
    def __getitem__(self, k):
        l = [i[k] for i in self.lst if k in i.dict]
        if not len(l) and k[-1]=='s': # plural!
            l = [i[k[:-1]] for i in self.lst if k[:-1] in i.dict]
        #if len(l)== 0:
        #    raise AttributeError(k)
        return VarList(map(lambda x:Var(val=x) if not isinstance(x,Var) and not isinstance(x,VarList) else x, l))
    def __getattr__(self, k):
        return self[k]
    def __repr__(self):
        return 'VarList %s'%str(self.lst)
    def __str__(self):
        return 'VarList %s'%str(self.lst)
    @property
    def vals(self):
   #     return [i.val for i in self.lst]
        return [i.val if not isinstance(i.val,Var) else i.val.val for i in self.lst]

class StackModel:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, *x):
        for l in self.layers:
            x = l(*x)
            if type(x) != list and type(x) != tuple:
                x = [x]
        return x

    def apply(self, x):
        x = Var(val=x)
        fpass = [x]
        for l in self.layers:
            x = l.apply(x)
            fpass.append(x)
        return x.val, VarList(fpass)

def initialise_shared(i):
    params = {}
    for args in i:
        params[args[0]] = shared(*args)
    return params
