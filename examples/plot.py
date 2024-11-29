import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_data

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + '/tmp'

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['figure.dpi'] = 170
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_full = load_data("tmp/fulldynamics.npz") # kinodynamics_f6 fulldynamics centroidal_f6
xs = data_full["xs"]
us = data_full["us"]
com = data_full["com"]
FL_force = data_full["FL_force"]
FR_force = data_full["FR_force"]
RL_force = data_full["RL_force"]
RR_force = data_full["RR_force"]
FL_trans = data_full["FL_trans"]
FR_trans = data_full["FR_trans"]
RL_trans = data_full["RL_trans"]
RR_trans = data_full["RR_trans"]
FL_trans_ref = data_full["FL_trans_ref"]
FR_trans_ref = data_full["FR_trans_ref"]
RL_trans_ref = data_full["RL_trans_ref"]
RR_trans_ref = data_full["RR_trans_ref"]
L_measured = data_full["L_measured"]
solve_time = data_full["solve_time"]

Tx_full = len(xs)
Tl_full = len(FL_trans)
ttx_full = np.linspace(0,Tx_full * 0.001, Tx_full)
ttl_full = np.linspace(0,Tl_full * 0.01, Tl_full)


# CoM
plt.figure('CoM')
plt.subplot(311)
plt.ylabel('x')
plt.plot(ttl_full, com[:,0], label = 'CoM_x')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(312)
plt.ylabel('y')
plt.plot(ttl_full, com[:,1], label = 'CoM_y')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(313)
plt.ylabel('z')
plt.plot(ttl_full, com[:,2], label = 'CoM_z')
plt.grid(True)
plt.legend(loc="upper left")


plt.figure('Angular momentum')
plt.plot(ttl_full, L_measured[:,2], label = 'Full')
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Ang. mom. along z")
plt.legend(loc="lower left")
# Foot force

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5),
                        layout="constrained")
axs[0].plot(ttl_full, FL_force[:,0])
axs[0].set_title('FL foot')
axs[0].set_ylabel('X force (N)')
axs[0].grid(True)
axs[1].plot(ttl_full, FL_force[:,1])
axs[1].set_ylabel('Y force (N)')
axs[1].grid(True)
#axs[1,0].set_title('Left foot')
axs[2].plot(ttl_full, FL_force[:,2])
axs[2].set_ylabel('Z force (N)')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)
#axs[2,0].set_title('Fz left')

# Foot translation

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(3.5, 2.5),
                        layout="constrained")
axs[0,0].plot(ttl_full, FL_trans[:,0])
axs[0,0].plot(ttl_full, FL_trans_ref[:,0], 'r')
axs[0,0].set_title('FL x')
axs[0,0].grid(True)
axs[1,0].plot(ttl_full, FL_trans[:,1])
axs[1,0].plot(ttl_full, FL_trans_ref[:,1], 'r')
axs[1,0].grid(True)
axs[1,0].set_title('FL y')
axs[2,0].plot(ttl_full, FL_trans[:,2])
axs[2,0].plot(ttl_full, FL_trans_ref[:,2], 'r')
axs[2,0].grid(True)
axs[2,0].set_title('FL z')
axs[0,1].plot(ttl_full, FR_trans[:,0])
axs[0,1].plot(ttl_full, FR_trans_ref[:,0], 'r')
axs[0,1].grid(True)
axs[0,1].set_title('FR x')
axs[1,1].plot(ttl_full, FR_trans[:,1])
axs[1,1].plot(ttl_full, FR_trans_ref[:,1], 'r')
axs[1,1].grid(True)
axs[1,1].set_title('FR y')
axs[2,1].plot(ttl_full, FR_trans[:,2])
axs[2,1].plot(ttl_full, FR_trans_ref[:,2], 'r')
axs[2,1].grid(True)
axs[2,1].set_title('FR z')

# Joint power
nq = 19
nv = 18
nu = 12
power_full = []

for i in range(Tx_full):
    pp3 = np.abs(us[i] * xs[i][nq + 6:])
    power_full.append(np.sum(pp3))

plt.figure()
#plt.title('Total joint power')
plt.grid(True)
plt.ylabel('Dissipated power $(kg.m^2.s^{-3})$')
plt.xlabel('Time (s)')
plt.plot(ttx_full, power_full,label = "Full dynamics")
plt.legend(loc = "upper left")
print("Mean power for fulldynamics is " + str(np.mean(power_full)))
print("Dissipated energy for fulldynamics is " + str(np.sum(power_full) * 0.001))
torque = [[] for i in range(3)]
torque_ric = [[] for i in range(3)]
for i in range(len(us)):
    torque[0].append(us[i][0])
    torque[1].append(us[i][1])
    torque[2].append(us[i][2])

plt.figure()
plt.grid(True)
plt.ylabel('Torque N/m')
plt.xlabel('Time (s)')
plt.plot(ttx_full, torque[0],label = "FL hip")
plt.plot(ttx_full, torque[1],label = "FL thigh")
plt.plot(ttx_full, torque[2],label = "FL ankle")
plt.legend(loc = "upper left")
plt.show()
