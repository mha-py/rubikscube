'''
Rubiks cube class
(c) 16.9.2020 mha
'''


import numpy as np
from colorama import Fore, Back # for coloured print output (draw function)

# Rotation operations
rotx = lambda x, y, z: (+x, -z, +y)
roty = lambda x, y, z: (+z, +y, -x)
rotz = lambda x, y, z: (-y, +x, +z)

# Counterrotations
rotxi = lambda x, y, z: (+x, +z, -y)
rotyi = lambda x, y, z: (-z, +y, +x)
rotzi = lambda x, y, z: (+y, -x, +z)

# Mirroring
mirrx = lambda x, y, z: (-x, y, z)
mirry = lambda x, y, z: (x, -y, z)
mirrz = lambda x, y, z: (x, y, -z)

# Conditions/Layers
cond_f = lambda x, y, z: x==+1
cond_b = lambda x, y, z: x==-1
cond_r = lambda x, y, z: y==+1
cond_l = lambda x, y, z: y==-1
cond_u = lambda x, y, z: z==+1
cond_d = lambda x, y, z: z==-1
cond_all = lambda x, y, z: True

colors = [ 'r', 'o', 'g', 'b', 'w', 'y' ]
colors_rev = { c: k for k, c in enumerate(colors) }

colcode = [ Back.RED, Back.MAGENTA, Back.GREEN, Back.BLUE, Back.WHITE, Back.YELLOW ] # for printing

_dict_piecesorientation = dict() # dictionary for the orientation of corner pieces (i. e. how are the colors of the corner pieces arranged?)

def inverse_moves(moves):
    'Returns the inverse of the moves'
    res = []
    for m in reversed(list(moves)):
        res.append(m.swapcase())
    return ''.join(res)


class RubiksCube():
    def __init__(self, N=3):
        self.N = N
        
        # Lets create a list of relevant indizes which we can feed into our for loops
        self._xyzf_inds = []
        off = (N-1)/2
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    (x, y, z) = (i-off, j-off, k-off)
                    if abs(x)!=0:
                        self._xyzf_inds.append((x, y, z, x, 0, 0))
                    if abs(y)!=0:
                        self._xyzf_inds.append((x, y, z, 0, y, 0))
                    if abs(z)!=0:
                        self._xyzf_inds.append((x, y, z, 0, 0, z))
        self.reset()
        self.draw = self.draw_colored
        
        
    def reset(self):
        '''Resets the cube, i. e. brings it to the solved state'''
        N = self.N
        self.cube = 99 * np.ones((N, N, N, 3), dtype='uint8')
        for x, y, z, fx, fy, fz in self._xyzf_inds:
            if   fx != 0: f=0
            elif fy != 0: f=1
            elif fz != 0: f=2
            c = int(2*f + 1.*([x,y,z][f]>0))
            self[x,y,z,fx,fy,fz] = c
        
    def issolved(self):
        '''Checks if the cube is solved and rotated in the right way(!)'''
        faces = self.getfaces()
        for k in range(5): # need only to check 5 faces (6 is also right if first 5 were right)
            if np.min(faces[k]) != np.max(faces[k]):
                return False
        return True
    
    def copy(self):
        '''Creates a copy of this cube'''
        cpy = self.__class__(self.N)
        cpy.cube = self.cube.copy()
        return cpy
        
    def draw_uncolored(self):
        '''Draws the cube as an ascii string.'''
        N = self.N
        
        s = ''
        # Obere Seite
        s += 8*' ' + '+' + 7*'-' + '+' + '\n' # Dekoration
        f = 2; k = +2
        for i in range(3):
            s += 8*' ' + '| '
            for j in range(3):
                c = self.cube[i,j,k,f]
                s += colors[c] + ' '
            s += '|\n'
            
        # Links, vordere, rechte, hintere Seite
        s +=  '+-------+-------+-------+-------+\n'
        for k in reversed(range(N)):
            s += '| '
            # links
            f = 1; j = 0
            for i in range(N):
                c = self.cube[i,j,k,f]
                s += colors[c] + ' '
            s += '| '
                
            # vorne
            f = 0; i = 2
            for j in range(N):
                c = self.cube[i,j,k,f]
                s += colors[c] + ' '
            s += '| '
                
            # rechts
            f = 1; j = 2
            for i in reversed(range(N)):
                c = self.cube[i,j,k,f]
                s += colors[c] + ' '
            s += '| '
                
            # hinten
            f = 0; i = 0
            for j in reversed(range(N)):
                c = self.cube[i,j,k,f]
                s += colors[c] + ' '
            s += '|\n'
        s +=  '+-------+-------+-------+-------+\n'
            
        # untere Seite
        f = 2; k = 0
        for i in reversed(range(N)):
            s += 8*' ' + '| '
            for j in range(3):
                c = self.cube[i,j,k,f]
                s += colors[c] + ' '
            s += '|\n'
        s += 8*' ' + '+' + 7*'-' + '+' + '\n' # Dekoration
        
        print(s)
        
        
    def draw_colored(self):
        '''Draws the cube as an ascii string.'''
        N = self.N
        
        s = ''
        # Obere Seite
        s += 9*' ' + '+' + 8*'-' + '+' + '\n' # Dekoration
        f = 2; k = +2
        for i in range(3):
            s += 9*' ' + '| '
            for j in range(3):
                c = self.cube[i,j,k,f]
                s += colcode[c]
                s += colors[c] + ' '
            s += Back.RESET
            s += ' |\n'
            
        # Links, vordere, rechte, hintere Seite
            s += Back.RESET
        s +=  '+--------+--------+--------+--------+\n'
        for k in reversed(range(N)):
            s += '| '
            # links
            f = 1; j = 0
            for i in range(N):
                c = self.cube[i,j,k,f]
                s += colcode[c]
                s += colors[c] + ' '
            s += Back.RESET
            s += ' | '
                
            # vorne
            f = 0; i = 2
            for j in range(N):
                c = self.cube[i,j,k,f]
                s += colcode[c]
                s += colors[c] + ' '
            s += Back.RESET
            s += ' | '
                
            # rechts
            f = 1; j = 2
            for i in reversed(range(N)):
                c = self.cube[i,j,k,f]
                s += colcode[c]
                s += colors[c] + ' '
            s += Back.RESET
            s += ' | '
                
            # hinten
            f = 0; i = 0
            for j in reversed(range(N)):
                c = self.cube[i,j,k,f]
                s += colcode[c]
                s += colors[c] + ' '
            s += Back.RESET
            s += ' |\n'
        s +=  '+--------+--------+--------+--------+\n'
            
        # untere Seite
        f = 2; k = 0
        for i in reversed(range(N)):
            s += 9*' ' + '| '
            for j in range(3):
                c = self.cube[i,j,k,f]
                s += colcode[c]
                s += colors[c] + ' '
            s += Back.RESET
            s += ' |\n'
        s += 9*' ' + '+' + 8*'-' + '+' + '\n' # Dekoration
        
        print(s)
        
        
    def fromstring(self, s):
        '''Works with an input string like:
        yyo
        yyo
        ybb
        ggo bww ryy grr
        ggg ooo bbb yrr
        ggg ooo bbb yrr
        www
        wwr
        wwr,
        where all empty spaces will be ignored. Ordering of the letters corresponds to the output format of draw().
        '''
        N = self.N
        lines = s.split('\n')
        lines = [ l.replace(' ', '') for l in lines ]
        
        # Obere Seite
        f = 2; k = +2
        for i in range(N):
            l, lines = lines[0], lines[1:]
            for j in range(N):
                self.cube[i,j,k,f] = colors_rev[l[0]]
                l = l[1:]
        
        # Links, vordere, rechte, hintere Seite
        for k in reversed(range(N)):
            l, lines = lines[0], lines[1:]
            # links
            f = 1; j = 0
            for i in range(N):
                self.cube[i,j,k,f] = colors_rev[l[0]]
                l = l[1:]
                
            # vorne
            f = 0; i = 2
            for j in range(N):
                self.cube[i,j,k,f] = colors_rev[l[0]]
                l = l[1:]
                
            # rechts
            f = 1; j = 2
            for i in reversed(range(N)):
                self.cube[i,j,k,f] = colors_rev[l[0]]
                l = l[1:]
                
            # hinten
            f = 0; i = 0
            for j in reversed(range(N)):
                self.cube[i,j,k,f] = colors_rev[l[0]]
                l = l[1:]
            
        # untere Seite
        f = 2; k = 0
        for i in reversed(range(N)):
            l, lines = lines[0], lines[1:]
            for j in range(N):
                self.cube[i,j,k,f] = colors_rev[l[0]]
                l = l[1:]
        self.checkconsistency()
        
        
    def show(self):
        N = self.N
        #colors = [ 'r', 'o', 'g', 'b', 'w', 'y' ]
        clrs = [ (.8, .2, .2), (.9, .5, .0), (.1, .8, .2), (.0, .25, .9), (.9, .9, .9), (.95, .95, .0) ]
        img = np.ones((3*self.N, 4*self.N, 3))
        row = 0
        
        # Obere Seite
        f = 2; k = +2
        for i in range(N):
            col = N
            for j in range(N):
                c = self.cube[i,j,k,f]
                img[row, col] = clrs[c]
                col += 1
            row += 1
            
        # Links, vordere, rechte, hintere Seite
        for k in reversed(range(N)):
            col = 0
            # links
            f = 1; j = 0
            for i in range(N):
                c = self.cube[i,j,k,f]
                img[row, col] = clrs[c]
                col += 1
                
            # vorne
            f = 0; i = 2
            for j in range(N):
                c = self.cube[i,j,k,f]
                img[row, col] = clrs[c]
                col += 1
                
            # rechts
            f = 1; j = 2
            for i in reversed(range(N)):
                c = self.cube[i,j,k,f]
                img[row, col] = clrs[c]
                col += 1
                
            # hinten
            f = 0; i = 0
            for j in reversed(range(N)):
                c = self.cube[i,j,k,f]
                img[row, col] = clrs[c]
                col += 1
                
            row += 1
            
        # untere Seite
        f = 2; k = 0
        for i in reversed(range(N)):
            col = N
            for j in range(N):
                c = self.cube[i,j,k,f]
                img[row, col] = clrs[c]
                col += 1
            row += 1
        
        from plot_utils import showimg_actualsize, upscale
        factor = 32
        img = upscale(img, factor)
        
        showimg_actualsize(img)
        
        
    def getfaces(self):
        '''Returns an array of all faces, i. e. a 6x3x3 array with numbers in 0..5
        '''
        assert self.N == 3
        
        result = np.zeros((6,self.N,self.N), dtype='uint8')
        
        # oben
        f = 2; k = +2
        result[0,:,:] = self.cube[:,:,k,f]
        
        # links
        f = 1; j = 0
        result[1,:,:] = self.cube[:,j,:,f]

        # vorne
        f = 0; i = 2
        result[2,:,:] = self.cube[i,:,:,f]

        # rechts
        f = 1; j = 2
        result[3,:,:] = self.cube[:,j,:,f]

        # hinten
        f = 0; i = 0
        result[4,:,:] = self.cube[i,:,:,f]
        
        # unten
        f = 2; k = 0
        result[5,:,:] = self.cube[:,:,k,f]
        
        return result
        
        
    def fromfaces(self, faces):
        '''Get a cube from the faces
        '''
        assert self.N == 3
        
        # oben
        f = 2; k = +2
        self.cube[:,:,k,f] = faces[0,:,:]
        
        # links
        f = 1; j = 0
        self.cube[:,j,:,f] = faces[1,:,:]

        # vorne
        f = 0; i = 2
        self.cube[i,:,:,f] = faces[2,:,:]

        # rechts
        f = 1; j = 2
        self.cube[:,j,:,f] = faces[3,:,:]

        # hinten
        f = 0; i = 0
        self.cube[i,:,:,f] = faces[4,:,:]

        # unteren
        f = 2; k = 0
        self.cube[:,:,k,f] = faces[5,:,:]
    
                
    def checkconsistency(self):
        from collections import Counter
        self._cube_read = self.cube
        cnt = Counter([ self[x,y,z,fx,fy,fz] for x, y, z, fx, fy, fz in self._xyzf_inds ])
        ##assert len(cnt.keys())==6, 'Inconsistent Cube!'
        ##assert all([n==self.N**2 for key, n in cnt.items()]), 'Inconsistent Cube!'
        
            
        
    def __setitem__(self, inds, value):
        '''Sets the color of the cube
        '''
        x, y, z, fx, fy, fz = inds
        assert abs(fx)+abs(fy)+abs(fz) == 1  # make sure its an orthonormal vector
        # Get face index (max 3 faces of a minicube)
        if   fx != 0: f=0
        elif fy != 0: f=1
        elif fz != 0: f=2
        off = (self.N-1)/2 # Index offset
        ##assert int(x+off)==x+off and int(y+off)==y+off and int(z+off)==z+off # make sure that int does no floor rounding to another int than wanted
        self.cube[int(x+off), int(y+off), int(z+off), f] = value
        
    def __getitem__(self, inds):
        '''Gets the color of old_cube(!)
        '''
        x, y, z, fx, fy, fz = inds
        assert abs(fx)+abs(fy)+abs(fz) == 1  # only one value can be set
        # Get face index (max 3 faces of a minicube)
        if   fx != 0: f=0
        elif fy != 0: f=1
        elif fz != 0: f=2
        off = (self.N-1)/2 # Index offset
        ##assert int(x+off)==x+off and int(y+off)==y+off and int(z+off)==z+off # make sure that int does no floor rounding to another int than wanted
        return self._cube_read[int(x+off), int(y+off), int(z+off), f]

    def _dotransform(self, transform, condition):
        '''Transform: A function which maps x, y, z to new x, y, z
        Condition: Condition on x, y, z to apply the transform

        Example: For applying `F`, transform is the rotation around x axis
        and condition is true if x=+1.
        '''
        self._cube_read = self.cube.copy()
        for (x, y, z, fx, fy, fz) in self._xyzf_inds:
            if not condition(x, y, z):
                continue
            x2, y2, z2 = transform(x, y, z)
            fx2, fy2, fz2 = transform(fx, fy, fz)
            self[x, y, z, fx, fy, fz] = self[x2, y2, z2, fx2, fy2, fz2]
        self.checkconsistency()
    
    
    def rotate(self, move):
        
        # if multiple moves
        if len(move) != 1:
            for m in move:
                self.rotate(m)
            return
        
        # if one move
        if move == 'F':
            self._dotransform(rotx, cond_f)
        if move == 'f':
            self._dotransform(rotxi, cond_f)
        if move == 'B':
            self._dotransform(rotxi, cond_b)
        if move == 'b':
            self._dotransform(rotx, cond_b)

        if move == 'R':
            self._dotransform(roty, cond_r)
        if move == 'r':
            self._dotransform(rotyi, cond_r)
        if move == 'L':
            self._dotransform(rotyi, cond_l)
        if move == 'l':
            self._dotransform(roty, cond_l)

        if move == 'U':
            self._dotransform(rotz, cond_u)
        if move == 'u':
            self._dotransform(rotzi, cond_u)
        if move == 'D':
            self._dotransform(rotzi, cond_d)
        if move == 'd':
            self._dotransform(rotz, cond_d)
            
        if move == 'X':
            self._dotransform(rotx, cond_all)
        if move == 'x':
            self._dotransform(rotxi, cond_all)
        if move == 'Y':
            self._dotransform(roty, cond_all)
        if move == 'y':
            self._dotransform(rotyi, cond_all)
        if move == 'Z':
            self._dotransform(rotz, cond_all)
        if move == 'z':
            self._dotransform(rotzi, cond_all)
            
        if move == 'i':
            self._dotransform(mirrx, cond_all)
        if move == 'j':
            self._dotransform(mirry, cond_all)
        if move == 'k':
            self._dotransform(mirrz, cond_all)
            
            
    def shuffle(self, n=15, verbose=0):
        s = ''
        for _ in range(n):
            rot = np.random.choice(['F', 'B', 'R', 'L', 'U', 'D'])
            if np.random.rand()<0.5:
                rot = rot.lower()
            s += rot
            self.rotate(rot)
        if verbose:
            print(s)
            
    def random_rot(self):
        # (1) Rotate anything but top face (and bottom)
        for _ in range(np.random.randint(0, 4)):
            self.rotate('Z')
        # (2) Rotate the top face (6 possible orientation)
        r = np.random.rand()
        if r < 4/6:
            n = int(6*r) # number of rotation
            self.rotate('X') # rotate top/yellow to the horizontal
            for _ in range(n):
                self.rotate('Z') # rotate top/yellow around
        elif r < 5/6:
            self.rotate('X') # rotate top/yellow down
            self.rotate('X')
        else:
            pass # leave top/yellow on top
        
        # if this function works, you will get an equal distribution of 24 orientation by using:
        
        '''
        d = defaultdict(lambad: 0)
        cube = RubiksCube()
        for _ in range(1000):
            cube.random_rot()
            key = cube.getfaces().tobytes()
            d[key] += 1
        '''
            
            
    def colornormalization(self):
        '''This operation changes the colors of the middle pieces to the standard
        Rotations around x/y/z together with this function will act as identity on the solved cube'''
        self._cube_read = self.cube
        
        center_pieces = [
            (+1, 0, 0, +1, 0, 0),
            (-1, 0, 0, -1, 0, 0),
            (0, +1, 0, 0, +1, 0),
            (0, -1, 0, 0, -1, 0),
            (0, 0, +1, 0, 0, +1),
            (0, 0, -1, 0, 0, -1)
        ]
        
        # dictionary which will tell us how to transform the colors to restore the normal (white bottom, orange front, etc.)
        # color ordering, negating any rotations around x/y/z axis.
        d = {}
        for x, y, z, fx, fy, fz in center_pieces:
            if   fx != 0: f=0
            elif fy != 0: f=1
            elif fz != 0: f=2
            c = int(2*f + 1.*([x,y,z][f]>0))  # this is the color we except at that center piece
            d[self[x,y,z,fx,fy,fz]] = c
            
        # using this dictionary, we can transform the old colors into new colors, which have the right color orientation
        for x, y, z, fx, fy, fz in self._xyzf_inds:
            self[x,y,z,fx,fy,fz] = d[self[x,y,z,fx,fy,fz]]
            
        # if this worked, a rotation around x/y/z and this function applied afterwards will act together as identity
        
        
    def getpieces(self):
        assert self.N == 3
        global _dict_piecesorientation
        
        pieces = -99*np.ones(20, dtype='uint8')
        n = 0
        
        # Corner pieces
        # Since opposite sides are 0,1; 2,3; 4,5, the corner pieces' colors is a triplet of these, e.g. (0,3,4)
        colors_corners = [ (c1, c2, c3) for c1 in [0,1] for c2 in [2,3] for c3 in [4,5] ]
        pos_corners = [ (i, j, k) for i in [-1, 1] for j in [-1, 1] for k in [-1, 1] ]
        for i, j, k in pos_corners:
            for m, (c1, c2, c3) in enumerate(colors_corners):
                for f in range(3):
                    if self.cube[i+1,j+1,k+1,f] not in [c1, c2, c3]:
                        break  # not the piece we were looking for
                else:  # --> the piece matches the colors
                    colors = m # this encodes the specific piece, i. e. the pieces color, into 0..7.
                    orientation = np.argmin(self.cube[i+1,j+1,k+1,:]) # orientation of the piece, its in 0..2
                    pieces[n] = (8*orientation + colors)
                    n += 1
                    
                    # save the colors of this piece in the dictionary (since we save only the orientation which leaves two colors unknown when restoring)
                    # needed only once! but doesnt harm to repeat
                    piececolors = self.cube[i+1,j+1,k+1,:].copy()
                    ax0 = np.argmin(piececolors) # axis for c1
                    piececolors[ax0] = 9999
                    ax1 = np.argmin(piececolors) # axis for c2
                    piececolors[ax1] = 9999
                    ax2 = np.argmin(piececolors) # axis for c3
                    
                    M = np.zeros((3,3))
                    M[0, ax0] = [i,j,k][ax0]
                    M[1, ax1] = [i,j,k][ax1] 
                    M[2, ax2] = [i,j,k][ax2]
                    sign = np.linalg.det(M)
                    
                    if (c1,c2,c3) in _dict_piecesorientation.keys():
                        assert _dict_piecesorientation[c1,c2,c3] == sign, f'Orientation inconsistency at corner pieces! c1,c2,c3 = {c1,c2,c3}'
                    else:
                        _dict_piecesorientation[c1,c2,c3] = sign
                    

        # Edge pieces
        # the edge pieces' colors are duplets which are not
        colors_edges = [ (c1, c2) for c1 in range(6) for c2 in range(6) if c1 < c2 and c1//2 != c2//2 ]
        pos_edges = [ (i, j, k) for i in range(3) for j in range(3) for k in range(3) if abs(i-1)+abs(j-1)+abs(k-1)==2 ]
        for i, j, k in pos_edges:
            for m, (c1, c2) in enumerate(colors_edges):
                for f in range(3):
                    if [i,j,k][f] == 1:
                        continue
                    if self.cube[i,j,k,f] not in [c1, c2]:
                        break
                else: # --> the piece matches the colors
                    colors = m
                    left_dim = np.argmin([abs(i-1), abs(j-1), abs(k-1)])
                    orientation = np.argmin(self.cube[i,j,k,:]) # in 0..2
                    if orientation > left_dim: orientation += -1 # in 0..1
                    pieces[n] = 12*orientation + colors
                    n += 1
        return pieces

    
    
                    
    def frompieces(self, pieces):
        assert self.N == 3
        
        colors_corners = [ (c1, c2, c3) for c1 in [0,1] for c2 in [2,3] for c3 in [4,5] ]
        pos_corners = [ (i, j, k) for i in [-1, 1] for j in [-1, 1] for k in [-1, 1] ]
        
        # Corner pieces
        for n in range(8):
            colors = pieces[n] % 8
            orientation = pieces[n] // 8
            i, j, k = pos_corners[n]
            c1, c2, c3 = colors_corners[colors]
            
            M = np.zeros((3,3))
            ax0 = orientation
            ax1 = (orientation+1)%3
            ax2 = (orientation+2)%3
            M[0, ax0] = [i,j,k][ax0]
            M[1, ax1] = [i,j,k][ax1]
            M[2, ax2] = [i,j,k][ax2]
            sign = np.linalg.det(M)
            
            if _dict_piecesorientation[c1,c2,c3] != sign:
                c1, c2, c3 = c1, c3, c2
            for f in range(3):
                self.cube[i+1,j+1,k+1,(f+orientation)%3] = [c1,c2,c3][f]
            
                
                
        colors_edges = [ (c1, c2) for c1 in range(6) for c2 in range(6) if c1 < c2 and c1//2 != c2//2 ]
        pos_edges = [ (i, j, k) for i in range(3) for j in range(3) for k in range(3) if abs(i-1)+abs(j-1)+abs(k-1)==2 ]
        
        # Edge pieces
        for n in range(12):
            colors = pieces[n+8] % 12
            orientation = pieces[n+8] // 12
            i, j, k = pos_edges[n]
            c1, c2 = colors_edges[colors]
            if orientation==1:
                c1, c2 = c2, c1
            left_dim = np.argmin([abs(i-1), abs(j-1), abs(k-1)])
            for f in range(3):
                if [i,j,k][f] == 1: # edge has no face looking in dimension f
                    continue
                l = f - (f>left_dim)
                self.cube[i,j,k,f] = [c1,c2][l]
                          