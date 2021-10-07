'''
Batchgen and Semisupervised batchgen
(c) 13.9.2020 mha
'''

import os, gzip, pickle
import numpy as np

from rubiks_helpers import *


nval = 1500


_precached = dict()


def combineBatches(mblist):
    xlist, zlist = zip(*mblist)
    xs = np.concatenate(xlist, axis=0)
    zs = np.concatenate(zlist, axis=0)
    return xs, zs

    
class BatchGen:
    def __init__(self, mbsize, files='data/rubik_shuffle_d20_cp_n100k_%d.dat', nval=nval, cache0=True, verbose=1):
        self.mbsize = mbsize
        self.nval = nval
        self.verbose = verbose
        self.collection0 = None
        
        # find out how many data files exist
        n = 0
        while os.path.exists(files%n):
            n += 1
        assert n > 0, 'There is no such file!'
        
        self.files = files
        self.nfiles = n
        
        self.cache0 = cache0
        if cache0: # cache the zeroth file
            self.precache()
            
        self.reset()
        
        # Find out if data has x or x and z
        if len(self.collection[0]) == 2:
            self.next = self.next_dual
        else:
            self.next = self.next_single
        
    def getvalbatch(self):
        'Provides the validation batch'
        xs = np.zeros((self.nval, 6, 6, 3, 3))
        zs = np.zeros((self.nval, 12, 6, 6, 3, 3))
        for i in range(self.nval):
            xs[i], zs[i] = _precached[self.files%0][i]
        return xs, zs
        
    def reset(self):
        self.setfile(0)
        self.k = 0
        
    def precache(self):
        'Caches the file 0 globally, so it only has to be loaded once and not when the BatchGen is initated'
        global _precached
        if self.files%0 not in _precached.keys():
            _precached[self.files%0] = self.load(0)
        
    def load(self, k):
        'Loads the files i and caches it for delivering minibatches.'
        fn = self.files % k
        print(f'Loading file "{fn}"')
        with gzip.open(fn, 'rb') as f:
                collection = pickle.load(f)
        print('Finished loading')
        return collection
    
    def setfile(self, k):
        if k == 0 and self.cache0:
            print('Loading precached file 0!')
            self.collection = _precached[self.files%0]
            istart = self.nval
        else:
            self.collection = self.load(k)
            istart = 0
            
        ilist = np.random.permutation(range(istart, len(self.collection)))
        self.minibatches = [ ilist[k*self.mbsize:(k+1)*self.mbsize] for k in range(len(ilist)//self.mbsize) ]
        self.k = k
        
    def nextfile(self):
        k = (self.k+1)%self.nfiles
        self.setfile(k)
        
    def next_dual(self):
        if len(self.minibatches) == 0:
            self.nextfile()
        mb = self.minibatches.pop()
        size = self.mbsize
        xs = np.zeros((size, 6, 6, 3, 3))#, dtype='uint8'
        zs = np.zeros((size, 12, 6, 6, 3, 3))
        for i, k in enumerate(mb):
            x, z = self.collection[k]
            # encode x
            xs[i] = x
            # encode z
            zs[i] = z
            #for r in range(12): # this loop goes through all moves which can be done on the cube
            #    zs[i,r] = z[r]
        return xs, zs
        
    def next_single(self):
        if len(self.minibatches) == 0:
            self.nextfile()
        mb = self.minibatches.pop()
        size = self.mbsize
        xs = np.zeros((size, 6, 6, 3, 3))
        zs = np.zeros((size, 12, 6, 6, 3, 3))
        for i, k in enumerate(mb):
            x = self.collection[k]
            # encode x
            xs[i] = x
            # encode z
            faces = oh2faces(x)
            for r in range(12): # this loop goes through all moves which can be done on the cube
                faces2 = apply(faces, r)
                oh = faces2oh(faces2)
                zs[i,r] = oh
        return xs, zs
        
        
collection = None
    
# Batch generator for neural network training
def batchgen(size, istart=nval, iend=0, verbose='batchgen'):
    '''size: minibatchsize
    istart, iend: Start- und Endindex der herausgegebenen Samples
    verbose: Wenn überschrieben, dann mit einem String des Batchgen Namens (z. B. Supervised)'''
    global collection
    if collection is None:
        import pickle
        collection, _ = pickle.load(open('data/rubik_shuffle15_10k.pkl', 'rb'))
        print('Loaded file shuffle15 with len', len(collection))
    iend = len(collection)
    print(iend)
        
    ep = 0
    while True:
        ep += 1
        ilist = range(istart, iend)
        ilist = np.random.permutation(ilist)
        minibatches = [ ilist[k*size:(k+1)*size] for k in range(len(ilist)//size) ]
        for mb in minibatches:
            xs = np.zeros((size, 6, 6, 3, 3))
            zs = np.zeros((size, 12, 6, 6, 3, 3))
            for i, k in enumerate(mb):
                x, z = collection[k]
                # encode x
                oh = faces2oh(x)
                xs[i] = oh
                # encode z
                for r in range(12): # this loop goes through all moves which can be done on the cube
                    oh = faces2oh(z[r])
                    zs[i,r] = oh
            yield xs, zs
        if verbose:
            print(f'{verbose}: finished one epoch ({ep})!')
            
            
# Batch generator for neural network training
def semibatchgen(size, istart=0, iend=0, verbose='semisuper-batchgen'):
    '''size: minibatchsize
    istart, iend: Start- und Endindex der herausgegebenen Samples
    verbose: Wenn überschrieben, dann mit einem String des Batchgen Namens (z. B. Supervised)'''
    ep = 0
    while True:
        ep += 1
        ilist = range(istart, iend)
        ilist = np.random.permutation(ilist)
        minibatches = [ ilist[k*size:(k+1)*size] for k in range(len(ilist)//size) ]
        for mb in minibatches:
            xs = np.zeros((size, 6, 6, 3, 3))
            zs = np.zeros((size, 6, 6, 3, 3))
            for i, k in enumerate(mb):
                x, z = semisupervised[k]
                # encode x
                oh = faces2oh(x)
                xs[i] = oh
                # encode z
                oh = faces2oh(z)
                zs[i] = oh
            yield xs, zs
        if verbose:
            print(f'{verbose}: finished one epoch ({ep})!')