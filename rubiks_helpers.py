'''
Helper functions
(c) 16.9.2020 mha
'''

from rubiks_cube import *
#from rubiks_cube_faces import *
import numpy as np
import torch

from numba import jit, njit
from numba.types import byte
    
GPU = torch.cuda.is_available()
if not GPU:
    print('NOT USING GPU!')
    

def policy2oh(p, n=12):
    'Converts a number to onehot format.'
    oh = np.zeros((len(p), n))
    for i in range(len(p)):
        oh[i, p[i]] = 1.
    return oh


def numpy2torch(x, dtype='float32'):
    'Converts a numpy array to a torch array'
    x = torch.from_numpy(x.astype(dtype))
    if GPU:
        x = x.cuda()
    return x

def torch2numpy(x):
    'Converts a torch array to a numpy array'
    return x.detach().cpu().numpy()

# Abbreviation for converting functions
t2np = torch2numpy
np2t = numpy2torch

# Dicionary for moves between chars and ints (e.g. 'R' <-> 2)
moves = ['F', 'B', 'R', 'L', 'U', 'D'] + ['f', 'b', 'r', 'l', 'u', 'd']
moves_rev = { k: m for k, m in enumerate(moves) }


#@njit
def argmax(list, num):
    'Generalization of np.argmax with more than one input'
    list = list.copy()
    #res = np.empty(num, dtype=byte)
    res = np.empty(num, dtype='int')
    for n in range(num):
        m, k = -np.inf, -1
        for i in range(len(list)):
            if list[i] > m:
                m = list[i]
                k = i
        res[n] = k
        list[k] = -np.inf
    return res


def update_mt(mt, n, tau):
    '''updates the mean teacher by the network
    mt: Mean teacher network
    n: Target network
    tau: Update constanst like 0.9'''
    mtdict = mt.state_dict()
    ndict = n.state_dict()
    for k in mtdict.keys():
        mtdict[k] = tau * mtdict[k] + (1-tau) * ndict[k]
    mt.load_state_dict(mtdict)
    
    
    
def SolutionLink(moves):
    '''Print a link to a website which shows the solution in 3D
    '''
    s = ''
    for m in moves: # e. g. DrU becomes D_R-_U
        s += m
        if m.islower():
            s += '-'
        s += '_'
    print('https://alg.cubing.net/?alg=' + s + '&type=alg')
    

   
    
    
from heapq import heappush, heappop

def hashable(item):
    'This function makes arrays and tuples hashable. nescessary for sets and dictionarys (which do not accept np.arrays as keys)'
    if type(item) is tuple:
        return tuple(hashable(it) for it in item)
    try:
        return item.tobytes()
    except:
        return hash(item)
    

class SolvingQueue:
    'In the solving routines we have a priority queue in which the priority changes from time to time. This class supports a discrete number of priorities.'
    def __init__(self, n=5):
        self.n = n # number of sub queues
        self.queue = [ [] for _ in range(self.n) ]
        self.items = set()
    def push(self, item, values):
        assert len(values)==self.n
        tiebreaker = np.random.rand()
        hash_item = hashable(item)
        if hash_item in self.items:
            return
        for i in range(self.n):
            heappush(self.queue[i], (values[i], tiebreaker, item))
        self.items.add(hash_item)
    def pop(self, i):
        assert len(self) > 0, 'Queue is empty!'
        while True:
            _, _, item = heappop(self.queue[i])
            if hashable(item) in self.items: # assure that item was not taken out by another queue `i`.
                break
        self.items.remove(hashable(item))
        return item
    def __len__(self):
        return len(self.items)