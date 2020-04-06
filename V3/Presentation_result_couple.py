import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from casadi import *
import time
import biorbd
from BiorbdViz import BiorbdViz
import conf as conf

model = biorbd.Model("/home/lim/Documents/code/Models/V7/arm26.bioMod")


# Initialisation / Importation donnee base

T = conf.T
N = conf.N
wMa = conf.wMa
Pa = conf.Pa
CM = conf.CM
dN = T / N

if CM == 1:
    Tart_opt = np.load('Couple_Opt_V7-Couple.npy')          # Couple articulaire - OPTIMISER
    DataMarkeur = np.load('DataMarkeur-Couple.npy')         #position Markeur Reel - CREER Virtuel
    Ncmv = len(DataMarkeur)
elif CM == 0:
    Tart_opt = np.load('Couple_Opt_V7.npy')                 # Couple articulaire - OPTIMISER
    DataMarkeur = np.load('DataMarkeur.npy')                #position Markeur Reel - CREER Virtuel
    Ncmv = len(DataMarkeur)
elif CM == 2:
    Tart_opt = np.load('Couple_Opt_V7.3-Activation.npy')
    DataMarkeur = np.load('DataMarkeur-Activation.npy')
    Ncmv = len(DataMarkeur)

Nb_Markeur = model.nbMarkers()                      # Nombre de Markeur Model
Nb_Torque = model.nbGeneralizedTorque()             # Nombre de couple articulaire a trouver du Model
J = 0
Jm = 0
Jc = 0
Je = 0

# Recuperation data a partir fichier optimise


# L = len(Tart_opt) / (model.nbQ() + model.nbQdot() + Nb_Torque)
# X = [[Tart_opt[k], Tart_opt[k + 2]] for k in range(L)]
# V = [[Tart_opt[k + 1], Tart_opt[k + 3]] for k in range(L)]
# T = [[Tart_opt[k + 4], Tart_opt[k + 5]] for k in range(L)]

Q1_opt = Tart_opt[0::6]
Q2_opt = Tart_opt[1::6]
Q1dot_opt = Tart_opt[2::6]
Q2dot_opt = Tart_opt[3::6]
T1_opt = Tart_opt[4::6]
T2_opt = Tart_opt[5::6]


# Traitement donnees - Calcul Erreur entre marker opt et creer


tps = [k * dN for k in range(N + 1)]
Q_opt = np.array([Q1_opt, Q2_opt]).T

#recreer position marker creer a partir de data Q creer
Q = MX.sym('Q', model.nbQ(), N+1)
MQ = Function('markers', [Q], [model.markers(Q)]).expand()
# markers_opt = [[MQ(Q_opt[pas])[Nmarker] for Nmarker in range(len(MQ(Q_opt[pas])))] for pas in range(len(Q1_opt))]
markers_opt = [np.array(MQ(Q_opt[pas]).T) for pas in range(len(Q1_opt))]
DataM = [DataMarkeur[int((k)*Ncmv/N)] for k in range(N)]                        #Tri pour avoir N data markeur equi-reparti
DataM.append(DataMarkeur[-1])                                                   #Pour avoir la même taille que markeurs_opt, et avoir le dernier point du coup

ErreurX = [sum(np.abs([DataM[k][i][0] - markers_opt[k][i][0] for i in range(Nb_Markeur)])) for k in range(N+1)]         #Erreur a chaque pas des marqueurs suivant X
ErreurY = [sum(np.abs([DataM[k][i][1] - markers_opt[k][i][1] for i in range(Nb_Markeur)])) for k in range(N+1)]         #Erreur a chaque pas des marqueurs suivant Y
ErreurZ = [sum(np.abs([DataM[k][i][2] - markers_opt[k][i][2] for i in range(Nb_Markeur)])) for k in range(N+1)]         #Erreur a chaque pas des marqueurs suivant Z
print(f'Erreur X : {sum(ErreurX)/len(ErreurX)}')
print(f'Erreur Y : {sum(ErreurY)/len(ErreurY)}')
print(f'Erreur Z : {sum(ErreurZ)/len(ErreurZ)}')

MarkcrX = [[DataM[k][i][0] for k in range(N+1)] for i in range(Nb_Markeur)]
MarkcrY = [[DataM[k][i][1] for k in range(N+1)] for i in range(Nb_Markeur)]
MarkcrZ = [[DataM[k][i][2] for k in range(N+1)] for i in range(Nb_Markeur)]
MarkopX = [[markers_opt[k][i][0] for k in range(N+1)] for i in range(Nb_Markeur)]
MarkopY = [[markers_opt[k][i][1] for k in range(N+1)] for i in range(Nb_Markeur)]
MarkopZ = [[markers_opt[k][i][2] for k in range(N+1)] for i in range(Nb_Markeur)]


# Calcul de la fct-obj


Jc = 1 * T1_opt.dot(T1_opt)
# Je = 0.1 * Q1_opt[:-1].dot(Q1_opt[:-1]) + 0.1 * Q2_opt[:-1].dot(Q2_opt[:-1])
# Je += 0.1 * Q1dot_opt[:-1].dot(Q1dot_opt[:-1]) + 0.1 * Q2dot_opt[:-1].dot(Q2dot_opt[:-1])
Je = 0
Jm = wMa * (sum(ErreurX) + sum(ErreurY) + sum(ErreurZ))
J = Jc + Jm + Je
print(f'fct-obj - Couple = {Jc} - soit {100 * Jc / J} %')
print(f'fct-obj - Markeur = {Jm} - soit {100 * Jm / J} %')
print(f'fct-obj - Etat = {Je} - soit {100 * Je / J} %')
print(f'fct-obj = {J}')


# Affichage plt


# Erreur Markeur Creer et optimise
plt.plot(tps, ErreurX, label='X')
plt.plot(tps, ErreurY, label='Y')
plt.plot(tps, ErreurZ, label='Z')
plt.title(f'Erreur Markeur V3 - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.figure()
# plt.ion()

#Markers comparaison Opt / Creer X - Y - Z
for i in range(Nb_Markeur) :
    plt.subplot(int(Nb_Markeur/2), int(Nb_Markeur/2), 1 + i)
    plt.plot(tps, np.array(MarkcrX[i]).squeeze(), 'x', label='Cre X')
    plt.plot(tps, np.array(MarkopX[i]).squeeze(), 'v', label='Opt X')
    plt.plot(tps, np.array(MarkcrY[i]).squeeze(), 'x', label='Cre Y')
    plt.plot(tps, np.array(MarkopY[i]).squeeze(), 'v', label='Opt Y')
    plt.plot(tps, np.array(MarkcrZ[i]).squeeze(), 'x', label='Cre Z')
    plt.plot(tps, np.array(MarkopZ[i]).squeeze(), 'v', label='Opt Z')
    plt.title(f'Conpaaison Markeur {i} - W = {wMa} - N = {N}')
plt.gcf().subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.9, wspace = 0.155, hspace = 0.25)
plt.legend(loc='best')
plt.figure()

plt.plot(tps, np.array(MarkcrX[2]).squeeze(), 'x', label='Cre X')
plt.plot(tps, np.array(MarkopX[2]).squeeze(), 'v', label='Opt X')
plt.plot(tps, np.array(MarkcrY[2]).squeeze(), 'x', label='Cre Y')
plt.plot(tps, np.array(MarkopY[2]).squeeze(), 'v', label='Opt Y')
plt.plot(tps, np.array(MarkcrZ[2]).squeeze(), 'x', label='Cre Z')
plt.plot(tps, np.array(MarkopZ[2]).squeeze(), 'v', label='Opt Z')
plt.title(f'Comparaison Markeur {2} - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.figure()

# Afficher Position / Vitesse / Couple de Q1 et Q2
plt.subplot(2, 2, 1)
plt.plot(tps, Q1_opt, label='Q_1')                      # Theta 1
plt.plot(tps, Q1dot_opt, label='Qdot_1')                # Vitesse Theta 1
plt.title(f'Etat optimise V3 - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.subplot(2, 2, 2)
plt.plot(tps[:-1], T1_opt, label='T_1')                      # Couple 1
plt.title(f'Couple optimise - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.subplot(2, 2, 3)
plt.plot(tps, Q2_opt, label='Q_2')                      # Theta 2
plt.plot(tps, Q2dot_opt, label='Qdot_2')                # Vitesse Theta 2
plt.title(f'Etat optimise - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.subplot(2, 2, 4)
plt.plot(tps[:-1], T2_opt, label='T_2')                 # Couple 2
plt.title(f'Couple optimise - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.gcf().subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.9, wspace = 0.155, hspace = 0.25)

plt.show(block=True)


# Affichage Model

#Pb ligne 96, veux pas lire le model : Wrong number or type of arguments for overloaded function 'Joints_meshPoints'.

# qs = Q_opt.T                                                                        # argument = les differents angles de ton model par DDL
# np.save("visual", qs.T)                                                             # qs.T pour transposer de qs
# b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V2/arm26.bioMod")            # Va chercher ton modele
# b.load_movement(qs.T)
# b.exec()
