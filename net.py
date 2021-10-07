
import torch
from torch import nn

from rubiks_cube_states import *
from rubiks_helpers import *

def randombatch(n=100):
    'For test reasons'
    return np2t(np.asarray([ state2oh(shuffle(n=30)) for _ in range(n) ]))

def inference_speed(net):
    'Inference speed of test batch in ms'
    from time import time
    batch = randombatch()
    t0 = time()
    for _ in range(100):
        net(batch)
    t1 = time()
    
    batch = randombatch()
    t2 = time()
    with torch.no_grad():
        for _ in range(100):
            net(batch)
    t3 = time()
    return (t1-t0)*10, (t3-t2)*10
    

GPU = True
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.iter = 1 # number of training iterations
        self.stats = [] # statistics of cubes solved while learning
        
        activation = nn.LeakyReLU()
        
        self.shared = nn.Sequential(
            nn.Linear(oh_len, 4096),
            nn.ReLU(),   # little mistake
            nn.Linear(4096, 2048),
            activation,
        )
        
        self.policyhead = nn.Sequential(
            nn.Linear(2048, 512),
            activation,
            nn.Linear(512, 12),
            nn.LogSoftmax(dim=-1)
        )
        
        self.valuehead = nn.Sequential(
            nn.Linear(2048, 512),
            activation,
            nn.Linear(512, 1)
        )
        
        self.solvablehead = nn.Sequential(
            nn.Linear(2048, 512),
            activation,
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        if GPU:
            self.cuda()
        
    def forward(self, x, value=True, policy=True, solvable=False):
        b = len(x)
        try:
            x = x.reshape(b, oh_len)
        except:
            print('Couldnt reshape x, x.shape is', x.shape)
            raise
        x = self.shared(x)
        result = tuple()
        if value:
            v = 10*self.valuehead(x)
            result += (v,)
        if policy:
            p = self.policyhead(x)
            result += (p,)
        if solvable:
            s = self.solvablehead(x)
            result += (s,)
        if len(result) == 1:
            return result[0]
        else:
            return result
        
        
        
        

    
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.iter = 1 # number of training iterations
        self.stats = [] # statistics of cubes solved while learning
        
        activation = nn.LeakyReLU()
        
        
        self.shared = nn.Sequential(
            nn.Linear(oh_len, 4096),
            activation,
            nn.Linear(4096, 4096),
            activation,
            nn.Linear(4096, 2048),
            activation,
        )
        
        self.policyhead = nn.Sequential(
            nn.Linear(2048, 512),
            activation,
            nn.Linear(512, 12),
            nn.LogSoftmax(dim=-1)
        )
        
        self.valuehead = nn.Sequential(
            nn.Linear(2048, 512),
            activation,
            nn.Linear(512, 1)
        )
        
        self.solvablehead = nn.Sequential(
            nn.Linear(2048, 512),
            activation,
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        if GPU:
            self.cuda()
        
    def forward(self, x, value=True, policy=True, solvable=False):
        b = len(x)
        try:
            x = x.reshape(b, oh_len)
        except:
            print('Couldnt reshape x, x.shape is', x.shape)
            raise
        x = self.shared(x)
        result = tuple()
        if value:
            v = 10*self.valuehead(x)
            result += (v,)
        if policy:
            p = self.policyhead(x)
            result += (p,)
        if solvable:
            s = self.solvablehead(x)
            result += (s,)
        if len(result) == 1:
            return result[0]
        else:
            return result
        
        
        
        
        

relu = nn.functional.relu
F = torch.nn.functional

class AttentionLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        
        self.qkv = nn.Conv1d(n, 2*n//8 + n, 1) # all in one layer: q and k have n//8 channels, v has n channels
        
        ###self.postlayer1 = nn.Conv1d(n, n, 1)
        ###self.postlayer2 = nn.Conv1d(n, n, 1)
        self.dense1 = nn.Linear(n, n)
        self.dense2 = nn.Linear(n, n)
        
        self.bn1 = nn.BatchNorm1d(n)
        self.bn2 = nn.BatchNorm1d(n)
    
    def forward(self, x):
        xr = x
        
        b, c, hw = x.shape
        
        x = relu(self.bn1(x))
        
        q, k, v = torch.split(self.qkv(x), [c//8, c//8, c], dim=1)
        beta = torch.bmm(q.permute(0,2,1), k) # has dimensions b, h*w, h*w
        beta = beta / np.sqrt(c//8)
        
        beta = F.softmax(beta, dim=1)
        self.last_beta = beta
        
        x = torch.bmm(v, beta)
        x = x.reshape(b, c, hw)
        
        xr = xr + x
        
        x = self.bn2(xr)
        x = x.permute(0, 2, 1).reshape(b*hw, c)
        x = self.dense2(relu(self.dense1(x)))
        x = x.reshape(b, hw, c).permute(0, 2, 1)
        
        xr = xr + x
        
        return xr
    
    

class Net_Att(nn.Module):
    def __init__(self):
        super().__init__()
        
        n=256 # für seitenweise attention
        #n=64 # für faceweise attention
        
        self.n = n
        
        self.iter = 1 # number of training iterations
        self.stats = [] # statistics of cubes solved while learning
        
        self.embed_c = nn.Conv1d(6*9, n, 1)
        #self.embed_c = nn.Conv1d(6, n, 1)
        self.embed_p = nn.Parameter(torch.zeros(1, n, 6))
        #self.embed_p = nn.Parameter(torch.zeros(1, n, 6*9))
        self.pre1 = nn.Conv1d(n, n, 1)
        
        self.al1 = AttentionLayer(n)
        self.al2 = AttentionLayer(n)
        
        self.dense1 = nn.Linear(n*6, 512)
        #self.dense1 = nn.Linear(n*6*9, 512)
        
        self.dense2 = nn.Linear(512, 1)
        
        if GPU:
            self.cuda()
        
        
    def forward(self, x, value=True, policy=False, solvable=False):
        assert value==True and policy==False and solvable==False, 'Only value supported!'
        b = len(x)
        
        x = x.permute(0, 1, 3, 4, 2) # b c s h w --> b c h w s (mit s der Seite)
        x = x.reshape(b, 6*9, 6)
        #x = x.reshape(b, 6, 9*6)
        
        x = self.embed_c(x)
        x = x + self.embed_p
        #x = self.pre1(relu(x))
        
        x = self.al1(x)
        x = self.al2(x)
        x = x.reshape(b, self.n*6)
        #x = x.reshape(b, self.n*9*6)
        x = relu(self.dense1(x))
        v = 10*self.dense2(x)
        
        return v

    