import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from casadi import *
import time
import biorbd
from BiorbdViz import BiorbdViz
# from implementation_belaise_V1 import Nibv1, Tibv1
import conf as conf

model = biorbd.Model("/home/lim/Documents/code/Models/V7/arm26.bioMod")

T          = conf.T
N          = conf.N

VecSDyn = biorbd.VecBiorbdMuscleStateDynamics(model.nbMuscleTotal())
act = MX.sym('act', 1, 1)
setAct = lambda vec,act : vec.setActivation(act)


def fct_Tarticulaire(etat, activation):                           # Retourne Liste 2x1, car 2 couples articulaires
    for k in range(model.nbMuscleTotal()):
        setAct(VecSDyn[k], activation[k])
    Result = model.muscularJointTorque(VecSDyn, etat[:model.nbDof()], etat[model.nbDof():])
    return Result


def fcn_objective_markers_Lea(wMa, wMt, markers, M_real):
    # tracking position marker
    # wMa = scaling anatomical marker
    # wMt = scaling technical marker
    # q = les positions généralisées
    # M_real = positions des markers

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
