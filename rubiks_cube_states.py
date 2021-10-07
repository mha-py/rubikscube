
'''
from rubiks_cube_faces import *

newstate = newfaces
state2oh = faces2oh
oh2state = oh2faces
Bstate2oh = Bfaces2oh
shape_state = (6,6,3,3)
oh_len = 6*3*3*6
cube2state = lambda c: c.getfaces()'''


from rubiks_cube_pieces import *

#newstate = newpieces
state2oh = pieces2oh
oh2state = oh2pieces
Bstate2oh = Bpieces2oh
shape_state = (20,24)
oh_len = 20*24
cube2state = lambda c: c.getpieces()

def state2cube(s):
    c = RubiksCube()
    c.frompieces(s)
    return c