'''
Accelerated version of the rubiks cube. Implemented by functions rather than permutations. Only for 3x3x3 Cube.
(c) 20.9.2020 mha
'''
from rubiks_cube import *
import pickle
from numba import jit, njit, prange
from numba.types import byte

moves = ['F', 'B', 'R', 'L', 'U', 'D'] + ['f', 'b', 'r', 'l', 'u', 'd']
moves_big = ['F', 'B', 'R', 'L', 'U', 'D'] + ['f', 'b', 'r', 'l', 'u', 'd'] + ['X', 'Y', 'Z'] + ['i', 'j', 'k']
moves_big_rev = { m: k for k, m in enumerate(moves_big) }



# New Cube in faces format
_cube = RubiksCube(3)
_faces0 = _cube.getfaces()
del _cube

def newfaces():
    '''Returns a solved cube in faces format'''
    return _faces0.copy()

'''
# Generate dictionary for applying the moves in faces format, for multiple moves like "URUR"
def _generate_dict(nrots):
    c = RubiksCube()
    _d = dict()
    _k = set()
    for i in range(nrots+1):
        for j in range(len(moves_big)**i):
            n = j
            ms = ''
            for _ in range(i):
                ms += moves_big[n%len(moves_big)]
                n //= len(moves_big)
            print(ms)
            _k.add(ms)
            for k in range(6*3*3):
                faces = np.zeros(6*3*3, dtype='uint8')
                faces[k] = 1
                faces = faces.reshape((6,3,3))
                c.fromfaces(faces)
                for m in ms:
                    c.rotate(m)
                targetfaces = c.getfaces()
                l = np.argmax(targetfaces.flatten())
                if k != l:
                    _d[ms, k] = l
    return _d, _k

#_d, _k = _generate_dict(nrots=3)
#with open('moves_faces.pkl', 'wb') as f:
#    pickle.dump((_d, _k), f)
with open('moves_faces.pkl', 'rb') as f:
    _d, _k = pickle.load(f)
'''


# Generate dictionary for applying the moves in faces format
def _generate_dict():
    c = RubiksCube()
    _d = dict()
    _k = set()
    for m in moves_big:
        _k.add(m)
        for k in range(6*3*3):
            faces = np.zeros(6*3*3, dtype='uint8')
            faces[k] = 1
            faces = faces.reshape((6,3,3))
            c.fromfaces(faces)
            c.rotate(m)
            targetfaces = c.getfaces()
            l = np.argmax(targetfaces.flatten())
            if k != l:
                _d[m, k] = l
    return _d, _k

_d, _k = _generate_dict()


def apply(faces, m):
    '''Applies the move m (string or int) to the cube in faces format'''
    faces = faces.flatten()
    resfaces = np.zeros_like(faces)
    #if type(m) is not str:
    if not isinstance(m, str):
        m = moves_big[m]
    assert m in _k, f'Unknown move {m}!' 
    for k in range(6*3*3):
        if (m, k) in _d.keys():
            l = _d[m,k]
            resfaces[l] = faces[k]
        else:
            resfaces[k] = faces[k]
    return resfaces.reshape((6,3,3))



@jit(cache=True)
def faces2oh(faces):
    '''Brings the cube into a one hot format. I. e.
    6x3x3 of numbers in 0..5  --->  6x6x3x3 of 0 and 1 '''
    #oh = np.zeros((6, 6, 3, 3), dtype='uint8')
    oh = np.zeros((6, 6, 3, 3), dtype=byte)
    for l in range(6):
        for i in range(3):
            for j in range(3):
                c = faces[l,i,j]
                oh[c,l,i,j] = 1
    return oh


def oh2faces_slow(oh):
    '''Converts the cube information from one hot format to faces format. I. e.
    6x6x3x3 of 0 and 1  --->  6x3x3 of numbers in 0..5 '''
    assert len(oh.shape) == 4, 'oh is supposed to have 4 dimensions!'
    faces = np.zeros((6, 3, 3), dtype='uint8')
    for l in range(6):
        for i in range(3):
            for j in range(3):
                faces[l,i,j] = np.argmax(oh[:,l,i,j])
    return faces


@njit(cache=True)
def oh2faces_fast(oh):
    faces = np.zeros((6, 3, 3), dtype=byte)
    for l in range(6):
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    if oh[k,l,i,j] > 0:
                        faces[l,i,j] = k
                        break
    return faces


oh2faces = oh2faces_fast


@jit(cache=True)
def Bfaces2oh(faces):
    '''Brings the cube into a one hot format. I. e.
    20 of numbers in 0..23  --->  20x24 of 0 and 1 '''
    bnum = len(faces)
    oh = np.zeros((bnum, 6, 6, 3, 3), dtype=np.uint8)
    for b in range(bnum):
        oh[b] = faces2oh(faces[b])
    return oh


def shuffle(faces=None, n=15):
    '''Performs a random move n times'''
    if isinstance(faces, type(None)):
        faces = _faces0
    for _ in range(n):
        m = np.random.choice(moves[:12])
        faces = apply(faces, m)
    return faces


def issolved(faces):
    '''Checks if the cube is solved'''
    for k in range(6): 
        if np.min(faces[k]) != np.max(faces[k]):
            return False
    return True


def draw(faces):
    c = RubiksCube(3)
    assert faces.shape == (6, 3, 3), f'Format is not ´faces´, shape is {faces.shape}!'
    c.fromfaces(faces)
    c.draw()


def colornormalization(faces):
    '''Permutes the colors such that the middle pieces are in the right order again
    Rotations around x/y/z together with this function will act as identity on the solved cube
    and augment other cubes
    '''
    d = {}
    for k in range(6):
        d[faces[k, 1, 1]] = _faces0[k, 1, 1]
    for k in range(6):
        for i in range(3):
            for j in range(3):
                faces[k, i, j] = d[faces[k, i, j]]
    return faces




# Generate random augmentations
def _generate_augmentations(n):
    c = RubiksCube()
    _d_p = dict() # moves
    _d_c = dict() # colors
    
    for i in range(n):
        moves = np.random.choice(['x', 'y', 'z', 'mx', 'my', 'mz'], 20)
        for k in range(6*3*3):
            
            # position permutation
            faces = np.zeros(6*3*3, dtype='uint8')
            faces[k] = 1
            faces = faces.reshape((6,3,3))
            c.fromfaces(faces)
            for m in moves:
                c.rotate(m)
            targetfaces = c.getfaces()
            l = np.argmax(targetfaces.flatten())
            if k != l:
                _d_p[i, k] = l
                
            # color permutation
            c.reset()
            for m in moves:
                c.rotate(m)
            faces = c.getfaces()
            for k in range(6):
                _d_c[i, faces[k, 1, 1]] = _faces0[k, 1, 1]
    return _d_p, _d_c
            
            
_num_augment = 200
#_d_aug_p, _d_aug_c = _generate_augmentations(_num_augment)
#with open('augmentations.pkl', 'wb') as f:
#    pickle.dump((_d_aug_p, _d_aug_c), f)
with open('augmentations.pkl', 'rb') as f:
    _d_aug_p, _d_aug_c = pickle.load(f)

    
def augment(faces, i=None):
    if i is None:
        i = np.random.randint(_num_augment)
        
    faces = faces.flatten()
    resfaces = np.zeros_like(faces)
    for k in range(6*3*3):
        if (i,k) in _d_aug_p.keys():
            l = _d_aug_p[i,k]
            resfaces[l] = _d_aug_c[i,faces[k]]
        else:
            resfaces[k] = _d_aug_c[i,faces[k]]
    return resfaces.reshape((6,3,3))

    
def augment_oh(oh, i=None):
    return faces2oh(augment(oh2faces(oh), i))