import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from casadi import *
import time
import biorbd
from BiorbdViz import BiorbdViz
# from implementation_belaise_V1 import Nibv1, Tibv1

model = biorbd.Model("/home/lim/Documents/code/Models/V3/arm26.bioMod")

T = 3.2
N = 151
# T = Tibv1
# N = Nibv1


def fct_Tarticulaire(etat, activation) :                           # Retourne Liste 2x1, car 2 couples articulaires
    VecStateDyn = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
    for k in range(model.nbMuscleTotal()):
        Vec = VecStateDyn[k]
        Vec.setActivation(activation[k])
    Result = model.muscularJointTorque(VecStateDyn, np.array([etat[0], etat[0 + model.nbDof()]]), np.array([etat[1], etat[1 + model.nbDof()]]))
    return Result

# def fct_obj_markeurs(Q, Data_Markeur, k):
#     Markers = Function('markers', [Q], model.markers(Q)).expand()
#     markers = Markers(Q)
#     Jm = 0
#     for nMark in range(model.nbMarkers()):
#


def fcn_objective_markers_Lea(wMa, wMt, markers, M_real):
    # tracking position marker
    # wMa = scaling anatomical marker
    # wMt = scaling technical marker
    # q = les positions généralisées
    # M_real = positions des markers

                                    # markers position
    # markers = [model.markers(Q)[j] for j in range(model.nbMarkers())]


    Jm = 0
    for nMark in range(model.nbMarkers()):
        if model.marker(nMark).isAnatomical():
            print("in anatam")
            Jm += wMa * (markers[0, nMark] - M_real[nMark, 0]) * (markers[0, nMark] - M_real[nMark, 0])            # probleme planaire en xy
            Jm += wMa * (markers[1, nMark] - M_real[nMark, 1]) * (markers[1, nMark] - M_real[nMark, 1])
            Jm += wMa * (markers[2, nMark] - M_real[nMark, 2]) * (markers[2, nMark] - M_real[nMark, 2])
            # Jm += wMa * (markers[:, nMark] - M_real[nMark])
        else:
            Jm += wMt * (markers[0, nMark] - M_real[nMark, 0]) * (markers[0, nMark] - M_real[nMark, 0])
            Jm += wMt * (markers[1, nMark] - M_real[nMark, 1]) * (markers[1, nMark] - M_real[nMark, 1])
            Jm += wMt * (markers[2, nMark] - M_real[nMark, 2]) * (markers[2, nMark] - M_real[nMark, 2])
            # Jm += wMt * (markers[:, nMark] - M_real[nMark])
    return Jm


def fct_EtoA(exitation):
    return exitation

def fct_Tmuscle(etat, activation) :                                 # Retourne Liste 2x3, car 2 articulation, 3 directions
    return [[1*etat*activation, 1, 1], [2*etat*activation, 2, 2]]

# T = Tibv1
# N = Nibv1