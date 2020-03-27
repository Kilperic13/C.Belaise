import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from casadi import *
import time
import biorbd
from BiorbdViz import BiorbdViz
import conf as conf

model = biorbd.Model("/home/lim/Documents/code/Models/V6/arm26.bioMod")


# Initialisation / Importation donnee base

T = conf.T
N = conf.N
wMa = conf.wMa
Pa = conf.Pa
DataMarkeur = np.load('DataMarkeur-Couple.npy')   #position Markeur Reel - CREER Virtuel
Ncmv = len(DataMarkeur)        #
dN = T / N                                          # Pas de temps
Nb_Markeur = model.nbMarkers()                      # Nombre de Markeur Model
Nb_Torque = model.nbGeneralizedTorque()             # Nombre de couple articulaire a trouver du Model
Nb_Muscle = model.nbMuscleTotal()
Tart_opt = np.load('Couple_Opt_V6-Couple.npy')                # Couple articulaire - OPTIMISER


# Recuperation data a partir fichier optimise

Periode = int(len(Tart_opt) / N)
print(f'Periode : {Periode}')
if Periode != 10 :
    print('Changer de N')
    print('Periode different de 10')

Q1_opt = Tart_opt[0::Periode]
Q2_opt = Tart_opt[1::Periode]
Q1dot_opt = Tart_opt[2::Periode]
Q2dot_opt = Tart_opt[3::Periode]
Act_opt = [Tart_opt[i + 4::Periode] for i in range(Nb_Muscle)]
#A1_opt = Tart_opt[4::Periode]
#A2_opt = Tart_opt[5::Periode]
#A3_opt = Tart_opt[6::Periode]
#A4_opt = Tart_opt[7::Periode]
#A5_opt = Tart_opt[8::Periode]
#A6_opt = Tart_opt[9::Periode]
#Act_opt = [A1_opt, A2_opt, A3_opt, A4_opt, A5_opt, A6_opt]


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


# Ja = Pa * (A1_opt.dot(A1_opt) + A2_opt.dot(A2_opt) + A3_opt.dot(A3_opt) + A4_opt.dot(A4_opt) + A5_opt.dot(A5_opt) + A6_opt.dot(A6_opt))
Ja = 0
for i in range(Nb_Muscle) :
    Ja += Pa * Act_opt[i].dot(Act_opt[i])
# Je = 0.1 * Q1_opt[:-1].dot(Q1_opt[:-1]) + 0.1 * Q2_opt[:-1].dot(Q2_opt[:-1])
# Je += 0.1 * Q1dot_opt[:-1].dot(Q1dot_opt[:-1]) + 0.1 * Q2dot_opt[:-1].dot(Q2dot_opt[:-1])
Je = 0
Jm = wMa * (sum(ErreurX) + sum(ErreurY) + sum(ErreurZ))
J = Ja + Jm + Je
print(f'fct-obj - Activation = {Ja} - soit {100 * Ja / J} %')
print(f'fct-obj - Markeur = {Jm} - soit {100 * Jm / J} %')
print(f'fct-obj - Etat = {Je} - soit {100 * Je / J} %')
print(f'fct-obj = {J}')


# Affichage plt


# Erreur Markeur Creer et optimise
plt.plot(tps, ErreurX, label='X')
plt.plot(tps, ErreurY, label='Y')
plt.plot(tps, ErreurZ, label='Z')
plt.title(f'Erreur Markeur V6 - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.figure()
# plt.ion()

#Markers comparaison Opt / Creer X - Y - Z
AxeXmkr = 4
if Nb_Markeur % 4 == 0 :
    AxeYmkr = Nb_Markeur//4
else :
    AxeYmkr = Nb_Markeur//4 +1
for i in range(Nb_Markeur) :
    plt.subplot(AxeYmkr, AxeXmkr, 1 + i)
    plt.plot(tps, np.array(MarkcrX[i]).squeeze(), 'x', label='Cre X')
    plt.plot(tps, np.array(MarkopX[i]).squeeze(), 'v', label='Opt X')
    plt.plot(tps, np.array(MarkcrY[i]).squeeze(), 'x', label='Cre Y')
    plt.plot(tps, np.array(MarkopY[i]).squeeze(), 'v', label='Opt Y')
    plt.plot(tps, np.array(MarkcrZ[i]).squeeze(), 'x', label='Cre Z')
    plt.plot(tps, np.array(MarkopZ[i]).squeeze(), 'v', label='Opt Z')
    plt.title(f'Comparaison Markeur {i} - W = {wMa} - N = {N}')
plt.gcf().subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.9, wspace = 0.155, hspace = 0.25)
plt.legend(loc='best')
plt.figure()

plt.plot(tps, np.array(MarkcrX[1]).squeeze(), 'x', label='Cre X')
plt.plot(tps, np.array(MarkopX[1]).squeeze(), 'v', label='Opt X')
plt.plot(tps, np.array(MarkcrY[1]).squeeze(), 'x', label='Cre Y')
plt.plot(tps, np.array(MarkopY[1]).squeeze(), 'v', label='Opt Y')
plt.plot(tps, np.array(MarkcrZ[1]).squeeze(), 'x', label='Cre Z')
plt.plot(tps, np.array(MarkopZ[1]).squeeze(), 'v', label='Opt Z')
plt.title(f'Comparaison Markeur {1} - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.figure()

#Activation Musculaire
AxeXmcl = 4
if Nb_Muscle % 4 == 0 :
    AxeYmcl = Nb_Muscle//4
else :
    AxeYmcl = Nb_Muscle//4 +1
for i in range(Nb_Muscle) :
    plt.subplot(AxeYmcl, AxeXmcl, 1 + i)
    plt.step(tps[:-1], Act_opt[i], label='Act')
    plt.title(f'Activation Musculaire {i + 1} - W = {wMa} - N = {N}')
plt.gcf().subplots_adjust(left = 0.05, bottom = 0.05, right = 0.95, top = 0.9, wspace = 0.155, hspace = 0.25)
plt.legend(loc='best')
plt.figure()

# Afficher Position / Vitesse
plt.subplot(2, 1, 1)
plt.plot(tps, Q1_opt, label='Q_1')                      # Theta 1
plt.plot(tps, Q1dot_opt, label='Qdot_1')                # Vitesse Theta 1
plt.title(f'Etat optimise V6 - W = {wMa} - N = {N}')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(tps, Q2_opt, label='Q_2')                      # Theta 2
plt.plot(tps, Q2dot_opt, label='Qdot_2')                # Vitesse Theta 2
plt.title(f'Etat optimise - W = {wMa} - N = {N}')
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
