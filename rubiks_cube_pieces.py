'''
Accelerated version of the rubiks cube. Implemented by functions rather than permutations. Only for 3x3x3 Cube.
Instead of using the faces format (6x9 faces of the 26 pieces), here we use a representation of the 20 movable pieces.
(c) 17.2.2021 mha
'''

from rubiks_cube import *

from numba import jit, njit
from numba.types import byte


moves = ['F', 'B', 'R', 'L', 'U', 'D'] + ['f', 'b', 'r', 'l', 'u', 'd']
moves_big = ['F', 'B', 'R', 'L', 'U', 'D'] + ['f', 'b', 'r', 'l', 'u', 'd'] + ['X', 'Y', 'Z']
moves_big_rev = { m: k for k, m in enumerate(moves_big) }


# New Cube in faces format
_cube = RubiksCube(3)
_state0 = _cube.getpieces()


def newstate():
    '''Returns a solved cube in faces format'''
    return _state0.copy()

def issolved(state):
    '''Checks if the cube is solved'''
    try:
        return np.all(state == _state0)
    except:
        return np.all(state == _oh0)


def draw(state):
    assert state.shape == (20,), f'Wrong format, shape is {state.shape}!'
    _cube.frompieces(state)
    _cube.draw()


def shuffle(state=None, n=30):
    '''Performs a random move n times'''
    if isinstance(state, type(None)):
        state = _state0
    for _ in range(n):
        m = np.random.choice(moves[:12])
        state = apply(state, m)
    return state


def augment(state):
    _cube.frompieces(state)
    _cube.rotate(np.random.choice(['', 'x', 'xx', 'X']))
    if np.random.rand() < 2/6:
        _cube.rotate(np.random.choice(['', 'yy']))
    else:
        _cube.rotate('y')
        _cube.rotate(np.random.choice(['', 'x', 'xx', 'X']))
    _cube.colornormalization()
    return _cube.getpieces()


@jit(cache=True)
def pieces2oh(pieces):
    '''Brings the cube into a one hot format. I. e.
    20 of numbers in 0..23  --->  20x24 of 0 and 1 '''
    #oh = np.zeros((20, 24), dtype='uint8')
    oh = np.zeros((20, 24), dtype=byte)
    for n in range(20):
        k = pieces[n]
        oh[n,k] = 1
    return oh

@jit(cache=True)
def Bpieces2oh(pieces):
    '''Brings the cube into a one hot format. I. e.
    20 of numbers in 0..23  --->  20x24 of 0 and 1 '''
    bnum = len(pieces)
    oh = np.zeros((bnum, 20, 24), dtype=byte)
    for b in range(bnum):
        oh[b] = pieces2oh(pieces[b])
    return oh


def oh2pieces_slow(oh):
    '''Converts the cube information from one hot format to pieces format. I. e.
    20x24 of 0 and 1  --->  20 of numbers in 0..23 '''
    assert len(oh.shape) == 2, 'oh is supposed to have 4 dimensions!'
    pieces = np.argmax(oh, axis=1)
    return pieces


@njit(cache=True)
def oh2pieces_fast(oh):
    '''Converts the cube information from one hot format to pieces format. I. e.
    20x24 of 0 and 1  --->  20 of numbers in 0..23 '''
    pieces = np.zeros(20, dtype=byte)
    for n in range(20):
        for k in range(24):
            if oh[n,k] > 0:
                pieces[n] = k
                break
    return pieces


oh2pieces = oh2pieces_fast
_oh0 = pieces2oh(_state0)


# Generic function terminology
oh2state = oh2pieces
state2oh = pieces2oh
shape_state = (20, 24)


# Generate dictionary for applying the moves in faces format
def _generate_dict():
    colors_corners = [ (c1, c2, c3) for c1 in [0,1] for c2 in [2,3] for c3 in [4,5] ]
    pos_corners = [ (i, j, k) for i in [-1, 1] for j in [-1, 1] for k in [-1, 1] ]
    colors_edges = [ (c1, c2) for c1 in range(6) for c2 in range(6) if c1 < c2 and c1//2 != c2//2 ]
    pos_edges = [ (i, j, k) for i in range(3) for j in range(3) for k in range(3) if abs(i-1)+abs(j-1)+abs(k-1)==2 ]
    
    state0 = _state0
    cube = RubiksCube()
    _d_pos = dict()
    _d_ori = dict()
    _k = set()
    
    ready = False
    while not ready:
        cube.shuffle()
        state0 = cube.getpieces()
    
        for m in moves_big:
            _k.add(m)
            cube.frompieces(state0)
            cube.rotate(m)
            state = cube.getpieces()
            
 
            # Corners and edges in the same loop, see that N is either 8 (for corners) or 12 (for edges)
            for n in range(20):
                N = 8 if n<8 else 12
                colors0 = state0[n] % N
                ori0 = state0[n] // N
                for l in range(20): 
                    if (l<8)!=(n<8): continue # only compare corners with corners and edges with edges
                    colors = state[l] % N
                    ori = state[l] // N
                    if colors == colors0:
                        if n!=l: # only keep track about pieces which are moved
                            _d_pos[m, n] = l
                            _d_ori[m, n, ori0] = ori
             

        # Check if orientation dict for edges is ready
        ready = True
        for m in moves_big:
            for n in range(8):
                if not (m, n) in _d_pos.keys():
                    continue
                for o in range(3):
                    if not (m, n, o) in _d_ori.keys():
                        ready = False
                        
                        
    return _d_pos, _d_ori, _k



_d_pos, _d_ori, _k = _generate_dict()

def apply_slow(state0, m):
    state = np.zeros(20, dtype='uint8')
    
    for n in range(20):
        N = 8 if n<8 else 12
        if (m, n) in _d_pos.keys(): # check if piece is moved or standing still
            l = _d_pos[m, n]
            ori0 = state0[n] // N
            ori = _d_ori[m, n, ori0]
            state[l] = state0[n]%N + N*ori
        else:
            state[n] = state0[n]
    
    return state
    
    
    
# die schnelle Variante von ´apply´ wird mit jit dekoriert. Da jit sich nicht mit dictionarys verträgt, erstellt folgende
# Funktion den Quellcode für die Funktion ´apply´ weiter unten. Dort ist der Inhalt der dictionary Objekte explizite enthalten.
    
def makesnippet():
    tab = '    '
    lines = []
    lines.append('@njit(cache=True)')
    lines.append('def apply3(state0, m):')
    lines.append(tab + 'state = np.zeros(20, dtype=byte)')
    lines.append('\n')
    lines.append(tab + 'for n in range(20):')
    lines.append(tab + tab + 'N = 8 if n<8 else 12')
    
    lines.append(tab + tab + '#### Beginning part of _d_pos ####')
    lines.append(tab + tab + 'foundinkeys = True')
    lines.append(tab + tab + 'if False: pass')
    for m in moves_big:
        #if m != 'F': break
        lines.append(tab + tab + f'elif m==\'{m}\':')
        lines.append(tab + tab + tab + 'if False: pass')
        for (m2, n), l in _d_pos.items():
            if m!=m2: continue
            l = _d_pos[m, n]
            lines.append(tab + tab + tab + f'elif n=={n}: l={l}')
        lines.append(tab + tab + tab + 'else: foundinkeys = False')
        
    lines.append(tab + tab + '#### Beginning part of _d_ori ####')
    lines.append(tab + tab + 'if foundinkeys:')
    lines.append(tab + tab + tab + 'ori0 = state0[n] // N')
    lines.append(tab + tab + tab + 'if False: pass')
    for m in moves_big:
        lines.append(tab + tab + tab + f'elif m==\'{m}\':')
        lines.append(tab + tab + tab + tab + 'if False: pass')
        for ori0 in range(3):
            lines.append(tab + tab + tab + tab + f'elif ori0=={ori0}:')
            lines.append(tab + tab + tab + tab + tab + 'if False: pass')
            for (m2, n, ori2) in _d_ori.keys():
                if m2!=m or ori2!=ori0: continue
                ori = _d_ori[m, n, ori0]
                lines.append(tab + tab + tab + tab + tab + f'elif n=={n}: ori={ori}')
                
            
    s =  ''.join([l + '\n' for l in lines])
    
    s += '''
            state[l] = state0[n]%N + N*ori
        else:
            state[n] = state0[n]
            
    return state'''

    return s



@njit(cache=True)
def _apply(state0, m):
    state = np.zeros(20, dtype=byte)


    for n in range(20):
        N = 8 if n<8 else 12
        #### Beginning part of _d_pos ####
        foundinkeys = True
        if False: pass
        elif m=='F':
            if False: pass
            elif n==4: l=5
            elif n==5: l=7
            elif n==6: l=4
            elif n==7: l=6
            elif n==16: l=18
            elif n==17: l=16
            elif n==18: l=19
            elif n==19: l=17
            else: foundinkeys = False
        elif m=='B':
            if False: pass
            elif n==0: l=2
            elif n==1: l=0
            elif n==2: l=3
            elif n==3: l=1
            elif n==8: l=9
            elif n==9: l=11
            elif n==10: l=8
            elif n==11: l=10
            else: foundinkeys = False
        elif m=='R':
            if False: pass
            elif n==2: l=6
            elif n==3: l=2
            elif n==6: l=7
            elif n==7: l=3
            elif n==11: l=14
            elif n==14: l=19
            elif n==15: l=11
            elif n==19: l=15
            else: foundinkeys = False
        elif m=='L':
            if False: pass
            elif n==0: l=1
            elif n==1: l=5
            elif n==4: l=0
            elif n==5: l=4
            elif n==8: l=13
            elif n==12: l=8
            elif n==13: l=16
            elif n==16: l=12
            else: foundinkeys = False
        elif m=='U':
            if False: pass
            elif n==1: l=3
            elif n==3: l=7
            elif n==5: l=1
            elif n==7: l=5
            elif n==10: l=15
            elif n==13: l=10
            elif n==15: l=18
            elif n==18: l=13
            else: foundinkeys = False
        elif m=='D':
            if False: pass
            elif n==0: l=4
            elif n==2: l=0
            elif n==4: l=6
            elif n==6: l=2
            elif n==9: l=12
            elif n==12: l=17
            elif n==14: l=9
            elif n==17: l=14
            else: foundinkeys = False
        elif m=='f':
            if False: pass
            elif n==4: l=6
            elif n==5: l=4
            elif n==6: l=7
            elif n==7: l=5
            elif n==16: l=17
            elif n==17: l=19
            elif n==18: l=16
            elif n==19: l=18
            else: foundinkeys = False
        elif m=='b':
            if False: pass
            elif n==0: l=1
            elif n==1: l=3
            elif n==2: l=0
            elif n==3: l=2
            elif n==8: l=10
            elif n==9: l=8
            elif n==10: l=11
            elif n==11: l=9
            else: foundinkeys = False
        elif m=='r':
            if False: pass
            elif n==2: l=3
            elif n==3: l=7
            elif n==6: l=2
            elif n==7: l=6
            elif n==11: l=15
            elif n==14: l=11
            elif n==15: l=19
            elif n==19: l=14
            else: foundinkeys = False
        elif m=='l':
            if False: pass
            elif n==0: l=4
            elif n==1: l=0
            elif n==4: l=5
            elif n==5: l=1
            elif n==8: l=12
            elif n==12: l=16
            elif n==13: l=8
            elif n==16: l=13
            else: foundinkeys = False
        elif m=='u':
            if False: pass
            elif n==1: l=5
            elif n==3: l=1
            elif n==5: l=7
            elif n==7: l=3
            elif n==10: l=13
            elif n==13: l=18
            elif n==15: l=10
            elif n==18: l=15
            else: foundinkeys = False
        elif m=='d':
            if False: pass
            elif n==0: l=2
            elif n==2: l=6
            elif n==4: l=0
            elif n==6: l=4
            elif n==9: l=14
            elif n==12: l=9
            elif n==14: l=17
            elif n==17: l=12
            else: foundinkeys = False
        elif m=='X':
            if False: pass
            elif n==0: l=1
            elif n==1: l=3
            elif n==2: l=0
            elif n==3: l=2
            elif n==4: l=5
            elif n==5: l=7
            elif n==6: l=4
            elif n==7: l=6
            elif n==8: l=10
            elif n==9: l=8
            elif n==10: l=11
            elif n==11: l=9
            elif n==12: l=13
            elif n==13: l=15
            elif n==14: l=12
            elif n==15: l=14
            elif n==16: l=18
            elif n==17: l=16
            elif n==18: l=19
            elif n==19: l=17
            else: foundinkeys = False
        elif m=='Y':
            if False: pass
            elif n==0: l=4
            elif n==1: l=0
            elif n==2: l=6
            elif n==3: l=2
            elif n==4: l=5
            elif n==5: l=1
            elif n==6: l=7
            elif n==7: l=3
            elif n==8: l=12
            elif n==9: l=17
            elif n==10: l=9
            elif n==11: l=14
            elif n==12: l=16
            elif n==13: l=8
            elif n==14: l=19
            elif n==15: l=11
            elif n==16: l=13
            elif n==17: l=18
            elif n==18: l=10
            elif n==19: l=15
            else: foundinkeys = False
        elif m=='Z':
            if False: pass
            elif n==0: l=2
            elif n==1: l=3
            elif n==2: l=6
            elif n==3: l=7
            elif n==4: l=0
            elif n==5: l=1
            elif n==6: l=4
            elif n==7: l=5
            elif n==8: l=11
            elif n==9: l=14
            elif n==10: l=15
            elif n==11: l=19
            elif n==12: l=9
            elif n==13: l=10
            elif n==14: l=17
            elif n==15: l=18
            elif n==16: l=8
            elif n==17: l=12
            elif n==18: l=13
            elif n==19: l=16
            else: foundinkeys = False
        #### Beginning part of _d_ori ####
        if foundinkeys:
            ori0 = state0[n] // N
            if False: pass
            elif m=='F':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==4: ori=0
                    elif n==6: ori=0
                    elif n==17: ori=0
                    elif n==18: ori=0
                    elif n==19: ori=0
                    elif n==7: ori=0
                    elif n==16: ori=0
                    elif n==5: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==5: ori=2
                    elif n==16: ori=1
                    elif n==6: ori=2
                    elif n==7: ori=2
                    elif n==17: ori=1
                    elif n==18: ori=1
                    elif n==4: ori=2
                    elif n==19: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==7: ori=1
                    elif n==5: ori=1
                    elif n==4: ori=1
                    elif n==6: ori=1
            elif m=='B':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=0
                    elif n==10: ori=0
                    elif n==11: ori=0
                    elif n==2: ori=0
                    elif n==9: ori=0
                    elif n==0: ori=0
                    elif n==3: ori=0
                    elif n==8: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=2
                    elif n==8: ori=1
                    elif n==9: ori=1
                    elif n==11: ori=1
                    elif n==10: ori=1
                    elif n==1: ori=2
                    elif n==2: ori=2
                    elif n==3: ori=2
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=1
                    elif n==3: ori=1
                    elif n==0: ori=1
                    elif n==1: ori=1
            elif m=='R':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==6: ori=2
                    elif n==11: ori=1
                    elif n==14: ori=1
                    elif n==15: ori=1
                    elif n==19: ori=1
                    elif n==2: ori=2
                    elif n==7: ori=2
                    elif n==3: ori=2
                elif ori0==1:
                    if False: pass
                    elif n==6: ori=1
                    elif n==11: ori=0
                    elif n==14: ori=0
                    elif n==7: ori=1
                    elif n==2: ori=1
                    elif n==3: ori=1
                    elif n==19: ori=0
                    elif n==15: ori=0
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=0
                    elif n==3: ori=0
                    elif n==7: ori=0
                    elif n==6: ori=0
            elif m=='L':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=2
                    elif n==4: ori=2
                    elif n==13: ori=1
                    elif n==16: ori=1
                    elif n==0: ori=2
                    elif n==5: ori=2
                    elif n==12: ori=1
                    elif n==8: ori=1
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=1
                    elif n==5: ori=1
                    elif n==8: ori=0
                    elif n==12: ori=0
                    elif n==16: ori=0
                    elif n==13: ori=0
                    elif n==1: ori=1
                    elif n==4: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==5: ori=0
                    elif n==4: ori=0
                    elif n==0: ori=0
                    elif n==1: ori=0
            elif m=='U':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=1
                    elif n==10: ori=0
                    elif n==13: ori=0
                    elif n==15: ori=0
                    elif n==18: ori=0
                    elif n==7: ori=1
                    elif n==3: ori=1
                    elif n==5: ori=1
                elif ori0==1:
                    if False: pass
                    elif n==5: ori=0
                    elif n==13: ori=1
                    elif n==7: ori=0
                    elif n==10: ori=1
                    elif n==1: ori=0
                    elif n==18: ori=1
                    elif n==3: ori=0
                    elif n==15: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==3: ori=2
                    elif n==7: ori=2
                    elif n==5: ori=2
                    elif n==1: ori=2
            elif m=='D':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==4: ori=1
                    elif n==6: ori=1
                    elif n==14: ori=0
                    elif n==17: ori=0
                    elif n==2: ori=1
                    elif n==9: ori=0
                    elif n==0: ori=1
                    elif n==12: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=0
                    elif n==9: ori=1
                    elif n==12: ori=1
                    elif n==6: ori=0
                    elif n==14: ori=1
                    elif n==17: ori=1
                    elif n==2: ori=0
                    elif n==4: ori=0
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=2
                    elif n==4: ori=2
                    elif n==0: ori=2
                    elif n==6: ori=2
            elif m=='f':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==4: ori=0
                    elif n==6: ori=0
                    elif n==17: ori=0
                    elif n==18: ori=0
                    elif n==19: ori=0
                    elif n==7: ori=0
                    elif n==16: ori=0
                    elif n==5: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==5: ori=2
                    elif n==16: ori=1
                    elif n==6: ori=2
                    elif n==7: ori=2
                    elif n==17: ori=1
                    elif n==18: ori=1
                    elif n==4: ori=2
                    elif n==19: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==7: ori=1
                    elif n==5: ori=1
                    elif n==4: ori=1
                    elif n==6: ori=1
            elif m=='b':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=0
                    elif n==10: ori=0
                    elif n==11: ori=0
                    elif n==2: ori=0
                    elif n==9: ori=0
                    elif n==0: ori=0
                    elif n==3: ori=0
                    elif n==8: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=2
                    elif n==8: ori=1
                    elif n==9: ori=1
                    elif n==11: ori=1
                    elif n==10: ori=1
                    elif n==1: ori=2
                    elif n==2: ori=2
                    elif n==3: ori=2
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=1
                    elif n==3: ori=1
                    elif n==0: ori=1
                    elif n==1: ori=1
            elif m=='r':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==6: ori=2
                    elif n==11: ori=1
                    elif n==14: ori=1
                    elif n==15: ori=1
                    elif n==19: ori=1
                    elif n==2: ori=2
                    elif n==7: ori=2
                    elif n==3: ori=2
                elif ori0==1:
                    if False: pass
                    elif n==6: ori=1
                    elif n==11: ori=0
                    elif n==14: ori=0
                    elif n==7: ori=1
                    elif n==2: ori=1
                    elif n==3: ori=1
                    elif n==19: ori=0
                    elif n==15: ori=0
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=0
                    elif n==3: ori=0
                    elif n==7: ori=0
                    elif n==6: ori=0
            elif m=='l':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=2
                    elif n==4: ori=2
                    elif n==13: ori=1
                    elif n==16: ori=1
                    elif n==0: ori=2
                    elif n==5: ori=2
                    elif n==12: ori=1
                    elif n==8: ori=1
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=1
                    elif n==5: ori=1
                    elif n==8: ori=0
                    elif n==12: ori=0
                    elif n==16: ori=0
                    elif n==13: ori=0
                    elif n==1: ori=1
                    elif n==4: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==5: ori=0
                    elif n==4: ori=0
                    elif n==0: ori=0
                    elif n==1: ori=0
            elif m=='u':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=1
                    elif n==10: ori=0
                    elif n==13: ori=0
                    elif n==15: ori=0
                    elif n==18: ori=0
                    elif n==7: ori=1
                    elif n==3: ori=1
                    elif n==5: ori=1
                elif ori0==1:
                    if False: pass
                    elif n==5: ori=0
                    elif n==13: ori=1
                    elif n==7: ori=0
                    elif n==10: ori=1
                    elif n==1: ori=0
                    elif n==18: ori=1
                    elif n==3: ori=0
                    elif n==15: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==3: ori=2
                    elif n==7: ori=2
                    elif n==5: ori=2
                    elif n==1: ori=2
            elif m=='d':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==4: ori=1
                    elif n==6: ori=1
                    elif n==14: ori=0
                    elif n==17: ori=0
                    elif n==2: ori=1
                    elif n==9: ori=0
                    elif n==0: ori=1
                    elif n==12: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=0
                    elif n==9: ori=1
                    elif n==12: ori=1
                    elif n==6: ori=0
                    elif n==14: ori=1
                    elif n==17: ori=1
                    elif n==2: ori=0
                    elif n==4: ori=0
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=2
                    elif n==4: ori=2
                    elif n==0: ori=2
                    elif n==6: ori=2
            elif m=='X':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=0
                    elif n==4: ori=0
                    elif n==6: ori=0
                    elif n==10: ori=0
                    elif n==11: ori=0
                    elif n==13: ori=1
                    elif n==14: ori=1
                    elif n==15: ori=1
                    elif n==17: ori=0
                    elif n==18: ori=0
                    elif n==19: ori=0
                    elif n==2: ori=0
                    elif n==7: ori=0
                    elif n==9: ori=0
                    elif n==16: ori=0
                    elif n==0: ori=0
                    elif n==3: ori=0
                    elif n==5: ori=0
                    elif n==12: ori=1
                    elif n==8: ori=0
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=2
                    elif n==5: ori=2
                    elif n==8: ori=1
                    elif n==9: ori=1
                    elif n==12: ori=0
                    elif n==16: ori=1
                    elif n==6: ori=2
                    elif n==11: ori=1
                    elif n==13: ori=0
                    elif n==14: ori=0
                    elif n==7: ori=2
                    elif n==10: ori=1
                    elif n==17: ori=1
                    elif n==1: ori=2
                    elif n==2: ori=2
                    elif n==18: ori=1
                    elif n==4: ori=2
                    elif n==3: ori=2
                    elif n==19: ori=1
                    elif n==15: ori=0
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=1
                    elif n==3: ori=1
                    elif n==7: ori=1
                    elif n==5: ori=1
                    elif n==4: ori=1
                    elif n==0: ori=1
                    elif n==1: ori=1
                    elif n==6: ori=1
            elif m=='Y':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=2
                    elif n==4: ori=2
                    elif n==6: ori=2
                    elif n==10: ori=1
                    elif n==11: ori=1
                    elif n==13: ori=1
                    elif n==14: ori=1
                    elif n==15: ori=1
                    elif n==17: ori=1
                    elif n==18: ori=1
                    elif n==19: ori=1
                    elif n==2: ori=2
                    elif n==7: ori=2
                    elif n==9: ori=1
                    elif n==16: ori=1
                    elif n==0: ori=2
                    elif n==3: ori=2
                    elif n==5: ori=2
                    elif n==12: ori=1
                    elif n==8: ori=1
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=1
                    elif n==5: ori=1
                    elif n==8: ori=0
                    elif n==9: ori=0
                    elif n==12: ori=0
                    elif n==16: ori=0
                    elif n==6: ori=1
                    elif n==11: ori=0
                    elif n==13: ori=0
                    elif n==14: ori=0
                    elif n==7: ori=1
                    elif n==10: ori=0
                    elif n==17: ori=0
                    elif n==1: ori=1
                    elif n==2: ori=1
                    elif n==18: ori=0
                    elif n==4: ori=1
                    elif n==3: ori=1
                    elif n==19: ori=0
                    elif n==15: ori=0
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=0
                    elif n==3: ori=0
                    elif n==7: ori=0
                    elif n==5: ori=0
                    elif n==4: ori=0
                    elif n==0: ori=0
                    elif n==1: ori=0
                    elif n==6: ori=0
            elif m=='Z':
                if False: pass
                elif ori0==0:
                    if False: pass
                    elif n==1: ori=1
                    elif n==4: ori=1
                    elif n==6: ori=1
                    elif n==10: ori=0
                    elif n==11: ori=1
                    elif n==13: ori=0
                    elif n==14: ori=0
                    elif n==15: ori=0
                    elif n==17: ori=0
                    elif n==18: ori=0
                    elif n==19: ori=1
                    elif n==2: ori=1
                    elif n==7: ori=1
                    elif n==9: ori=0
                    elif n==16: ori=1
                    elif n==0: ori=1
                    elif n==3: ori=1
                    elif n==5: ori=1
                    elif n==12: ori=0
                    elif n==8: ori=1
                elif ori0==1:
                    if False: pass
                    elif n==0: ori=0
                    elif n==5: ori=0
                    elif n==8: ori=0
                    elif n==9: ori=1
                    elif n==12: ori=1
                    elif n==16: ori=0
                    elif n==6: ori=0
                    elif n==11: ori=0
                    elif n==13: ori=1
                    elif n==14: ori=1
                    elif n==7: ori=0
                    elif n==10: ori=1
                    elif n==17: ori=1
                    elif n==1: ori=0
                    elif n==2: ori=0
                    elif n==18: ori=1
                    elif n==4: ori=0
                    elif n==3: ori=0
                    elif n==19: ori=0
                    elif n==15: ori=1
                elif ori0==2:
                    if False: pass
                    elif n==2: ori=2
                    elif n==3: ori=2
                    elif n==7: ori=2
                    elif n==5: ori=2
                    elif n==4: ori=2
                    elif n==0: ori=2
                    elif n==1: ori=2
                    elif n==6: ori=2

            state[l] = state0[n]%N + N*ori
        else:
            state[n] = state0[n]
            
    return state



_state_types = set()

def apply(state, m):
    global _state_types
    if not (type(state), type(m)) in _state_types:
        ##print('apply got new input types:', (type(state), type(m)))
        _state_types.add((type(state), type(m)))
        
    for mm in m:
        state = _apply(state, mm)
        
    return state