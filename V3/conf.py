import numpy as np

T          = 3.2
N          = 1000
wMa = 1000000
wMt = 1000000
Pa = 10

# Creation Markeur

print('Choose type of creation')
CMb = input()
CM = int(CMb)

# CM = 0                                # Comment/Decomment to activate/desactivate
# CM = 1                                # Comment/Decomment to activate/desactivate

# Function
# if CM == 0:                             # Creation base on the angle
#     fct_CM = lambda t : np.cos(t)
# elif CM == 1:                           # Creation base on the Torque
#     fct_CM = lambda t : np.tanh(t)
# else :
#     print('This option does not exist, choose an other one')
#     CMb = input()
#     CM = int(CMb)
