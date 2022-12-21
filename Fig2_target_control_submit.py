import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


import scipy
from scipy.integrate import odeint, solve_ivp, DOP853
from scipy.interpolate import interp1d


from multiprocessing import Pool
from time import time

import TMM

matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams.update({'axes.linewidth': 0.5})
matplotlib.rcParams.update({'ytick.direction': 'in'})
matplotlib.rcParams.update({'xtick.direction': 'in'})
matplotlib.rcParams.update({'ytick.major.width': 0.5})
matplotlib.rcParams.update({'xtick.major.width': 0.5})
matplotlib.rcParams.update({'ytick.minor.width': 0.5})
matplotlib.rcParams.update({'xtick.minor.width': 0.5})
matplotlib.rcParams.update({'figure.figsize': [3.375 *2, 2.5 * 1.8]})
matplotlib.rcParams.update({'savefig.pad_inches': 0.02}) #default 0.1
matplotlib.rcParams.update({'savefig.format': 'png'}) #default 0.1
matplotlib.rcParams.update({'figure.subplot.bottom': 0.05,
                            'figure.subplot.hspace': 0,
                            'figure.subplot.left': 0.05,
                            'figure.subplot.right': 0.95,
                            'figure.subplot.top': 0.95,
                            'figure.subplot.wspace': 0}) #default 0.125, 0.2, 0.125, 0.9, 0.88, 0.2




#%% Design

a = 1
omega_b = 2*np.pi / a
dt = a *0.01 

Tp = 20

tau = np.arange(-(Tp)*a, (Tp)*a+dt/10, dt)

delta = 0.01 / Tp
omega = np.linspace(-3*omega_b, 3*omega_b, 3001)
domega = omega[1]-omega[0]



N_list = 11
A = 105*np.pi / (256*omega_b)
S0 = (16/7)*A * np.linspace(0,1,N_list)
coeff_b = S0/16
coeff_a = (A-7*coeff_b)/4

S_list = (np.abs(omega/omega_b)<2).reshape(1,-1) * ((4-(omega/omega_b)**2)**2).reshape(1,-1) * (coeff_a.reshape(-1,1)@ ((omega/omega_b)**2).reshape(1,-1) + coeff_b.reshape(-1,1))
S_fw_list = S_list * (delta**2)
R_fw_list = np.real([np.sum([s*np.exp(-1j*om*tau)*domega/(2*np.pi) for s, om in zip(S, omega)], axis=0) for S in S_fw_list])


A = 20*np.pi/(27 * omega_b)
S2w = 4*A/5 * np.linspace(0, 1, N_list)
coeff_b = S2w/4
coeff_a = (A - 5*coeff_b)/2
S_list = ((np.abs(omega/omega_b)<3)* (omega/omega_b)**2 * (3 - np.abs(omega)/omega_b)).reshape(1,-1) * (coeff_a.reshape(-1,1)@((np.abs(omega/omega_b)-2)**2).reshape(1,-1) + coeff_b.reshape(-1,1) )
S_bw_list = S_list * (delta**2)
R_bw_list = np.real([np.sum([s*np.exp(-1j*om*tau)*domega/(2*np.pi) for s, om in zip(S, omega)], axis=0) for S in S_bw_list])




t_idx = np.arange(-round(Tp/dt), 0)
Rcenter_idx = np.argmin(np.abs(tau))



def int_1d_stat(R):
    S0 = np.sum([r*np.exp(1j*0*tau0) *dt for r, tau0 in zip(R, tau)])
    S2w = np.sum([r*np.exp(1j*2*omega_b*tau0) *dt for r, tau0 in zip(R, tau)])
    return omega_b*S0, omega_b*S2w


with Pool(30) as pool:
    Int_1d_stat_ctrlfw = np.array(pool.map(int_1d_stat, R_fw_list))
    Int_1d_stat_ctrlbw = np.array(pool.map(int_1d_stat, R_bw_list))



#%% covariance matrix

t =  np.arange(0*a, (Tp-dt/10)*a, dt)
tic = time()
tau0_idx = np.argmin(np.abs(tau))

def row(R, i):
    return np.array(R[tau0_idx+(0-i):tau0_idx+(len(t)-i)] )

def complex_block(mat):
    return 0.5 * np.block([[np.real(mat), -np.imag(mat)],[np.imag(mat), np.real(mat)]])

with Pool(30) as pool:
    cov_alpha_fw_list = [np.array(pool.starmap(row, zip([R]*len(t), range(len(t))))) for R in R_fw_list]
    cov_alpha_bw_list = [np.array(pool.starmap(row, zip([R]*len(t), range(len(t))))) for R in R_bw_list]



#%% Fig S3: ensemble structure factor validation / estimation

N_ens = 10000
dalpha_ens_fw = [np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_fw_list[0], size=N_ens),
                 np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_fw_list[5], size=N_ens),
                 np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_fw_list[10], size=N_ens)]
dalpha_ens_bw = [np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_bw_list[0], size=N_ens),
                 np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_bw_list[5], size=N_ens),
                 np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_bw_list[10], size=N_ens)]

R_fw, R_bw = [], []
S_cal_fw, S_cal_bw = [], []

for ii in range(3):
    for N_include in [100, 1000, 10000]:
        R = []
        for tau0 in tau:
            idx_diff = round(tau0/dt)
            if idx_diff >=0:
                R.append(np.mean( np.conj(dalpha_ens_fw[ii][:N_include,idx_diff:])*dalpha_ens_fw[ii][:N_include,:len(t)-idx_diff] ))
            else:
                idx_diff = np.abs(idx_diff)
                R.append( np.conj(np.mean( np.conj(dalpha_ens_fw[ii][:N_include,idx_diff:])*dalpha_ens_fw[ii][:N_include,:len(t)-idx_diff] )) )
        R = np.array(R)
        S = np.sum([r*np.exp(1j*omega*tau0) * dt for r, tau0 in zip(R[1:-1], tau[1:-1])], axis=0)
        R_fw.append(R)
        S_cal_fw.append(S)
        
        R = []
        for tau0 in tau:
            idx_diff = round(tau0/dt)
            if idx_diff >=0:
                R.append(np.mean( np.conj(dalpha_ens_bw[ii][:N_include,idx_diff:])*dalpha_ens_bw[ii][:N_include,:len(t)-idx_diff] ))
            else:
                idx_diff = np.abs(idx_diff)
                R.append( np.conj(np.mean( np.conj(dalpha_ens_bw[ii][:N_include,idx_diff:])*dalpha_ens_bw[ii][:N_include,:len(t)-idx_diff] )) )
        R = np.array(R)
        S = np.sum([r*np.exp(1j*omega*tau0) * dt for r, tau0 in zip(R[1:-1], tau[1:-1])], axis=0)
        R_bw.append(R)
        S_cal_bw.append(S)

fig, ax = plt.subplots(3,2, figsize=(6, 3.5), sharex=True)
for ii in range(3):
    ax[ii,0].plot(omega/omega_b, np.real(S_fw_list[5*ii])/delta**2, lw=2, color='crimson', label='Target', zorder=10)
    ax[ii,0].plot(omega/omega_b, np.real(S_cal_fw[3*ii])/delta**2, lw=0.5, color='lightsteelblue', ls='-', zorder=1, label=r'$N_\mathrm{ens}=10^2$')
    ax[ii,0].plot(omega/omega_b, np.real(S_cal_fw[3*ii+1])/delta**2, lw=0.5, color='royalblue', ls='-', zorder=5, label=r'$N_\mathrm{ens}=10^3$')
    ax[ii,0].plot(omega/omega_b, np.real(S_cal_fw[3*ii+2])/delta**2, lw=1, color='midnightblue', ls='-', zorder=100, label=r'$N_\mathrm{ens}=10^4$')
    
    ax[ii,1].plot(omega/omega_b, np.real(S_bw_list[5*ii])/delta**2, lw=2, color='crimson', zorder=10)
    ax[ii,1].plot(omega/omega_b, np.real(S_cal_bw[3*ii])/delta**2, lw=0.5, color='lightsteelblue', ls='-', zorder=1)
    ax[ii,1].plot(omega/omega_b, np.real(S_cal_bw[3*ii+1])/delta**2, lw=0.5, color='royalblue', ls='-', zorder=5)
    ax[ii,1].plot(omega/omega_b, np.real(S_cal_bw[3*ii+2])/delta**2, lw=1, color='midnightblue', ls='-', zorder=100)

ax[2,0].set(xlim=(-3,3), xlabel=r'$\omega/\omega_0$')
ax[1,0].set(ylabel=r'$S_\mathrm{FW}(\omega)/\delta^2$')
ax[2,1].set(xlim=(-3,3), xlabel=r'$\omega/\omega_0$')
ax[1,1].set(ylabel=r'$S_\mathrm{BW}(\omega)/\delta^2$')

ax[0,0].legend(frameon=False, loc=2, fontsize=6.7, ncol=2)
fig.tight_layout()
fig.savefig('FigS2_validation_Sw.pdf', format='pdf', dpi=1200)
fig.savefig('FigS2_validation_Sw.png', format='png', dpi=400)



#%% Scattering ensemable simulation, data generation

N_ens = 10000

def intensity_oscillation(cov_alpha):
    P_forward_TMM, P_backward_TMM = [], []
    np.random.seed(0)
    
    dalpha_ens = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha, size=N_ens)
 
    for dalpha in dalpha_ens:
        alpha = 1+ dalpha
        sys = TMM.time_dep(dt*np.ones_like(t), 1/alpha, np.ones_like(t), 1, 1)
        trans, reflec = sys.evolution(omega_b)[3:5]
        P_forward = np.abs(trans-np.exp(-1j*omega_b*Tp))**2
        P_backward = np.abs(reflec)**2
        P_forward_TMM.append(P_forward)
        P_backward_TMM.append(P_backward)
    return np.array(P_forward_TMM), np.array(P_backward_TMM)


tic = time()

P_ctrlfw_ens = [intensity_oscillation(cov) for cov in cov_alpha_fw_list]
P_ctrlbw_ens = [intensity_oscillation(cov) for cov in cov_alpha_bw_list]


P_ctrlfw_fw_ens = np.array([pens[0] for pens in P_ctrlfw_ens])
P_ctrlfw_bw_ens = np.array([pens[1] for pens in P_ctrlfw_ens])
P_ctrlbw_fw_ens = np.array([pens[0] for pens in P_ctrlbw_ens])
P_ctrlbw_bw_ens = np.array([pens[1] for pens in P_ctrlbw_ens])

toc = time()
print(toc-tic)

np.savetxt('Data_Fig2_ctrlfw_fw_ens.csv', P_ctrlfw_fw_ens, delimiter=' ')
np.savetxt('Data_Fig2_ctrlfw_bw_ens.csv', P_ctrlfw_bw_ens, delimiter=' ')
np.savetxt('Data_Fig2_ctrlbw_fw_ens.csv', P_ctrlbw_fw_ens, delimiter=' ')
np.savetxt('Data_Fig2_ctrlbw_bw_ens.csv', P_ctrlbw_bw_ens, delimiter=' ')




#%% Fig 2

P_ctrlfw_fw_ens = np.loadtxt('Data_Fig2_stat_forward.csv',  delimiter=' ')
P_ctrlfw_bw_ens = np.loadtxt('Data_Fig2_stat_backward.csv', delimiter=' ')

P_ctrlbw_fw_ens = np.loadtxt('Data_Fig3_stat_forward.csv',  delimiter=' ')
P_ctrlbw_bw_ens= np.loadtxt('Data_Fig3_stat_backward.csv', delimiter=' ')




cs = np.array([0.2,0.2,0.2,1]).reshape(1,-1)
ce = np.array(matplotlib.colors.to_rgba('mediumvioletred')).reshape(1,-1)
newcolors = cs +  np.linspace(0,1,256).reshape(-1,1) @ (ce-cs)
cmap1 = ListedColormap(newcolors)

ce = np.array(matplotlib.colors.to_rgba('royalblue')).reshape(1,-1)
newcolors = cs +  np.linspace(0,1,256).reshape(-1,1) @ (ce-cs)
cmap2 = ListedColormap(newcolors)

fig = plt.figure(figsize=(3.375*2, 2))
gs = GridSpec(nrows=125, ncols=200)



ax = np.array([fig.add_subplot(gs[10:58,0:30]), fig.add_subplot(gs[62:110,0:30]),
                fig.add_subplot(gs[10:41,50:95]), 
                fig.add_subplot(gs[44:76,50:95]),
                fig.add_subplot(gs[79:110,50:95]),
                fig.add_subplot(gs[10:58,128:158]), fig.add_subplot(gs[10:58,170:200]),
                fig.add_subplot(gs[62:110,128:158]), fig.add_subplot(gs[62:110,170:200])])



for ii in range(N_list):
    c1 = cmap1(ii/(N_list)*0.999)
    ax[0].plot(omega/omega_b, S_fw_list[ii]*omega_b/delta**2, c=c1, lw=1)
    
    c2 = cmap2(ii/(N_list)*0.999)
    ax[1].plot(omega/omega_b, S_bw_list[ii]*omega_b/delta**2, c=c2, lw=1)
    
ax[0].set_title('Structure factor', fontsize=7)
ax[0].set(xticks=np.linspace(-2,2,3), xticklabels=[], xlim=(-3,3),
          ylabel=r'$S_\mathrm{FW}(\omega)$', ylim=(0,np.max(S_fw_list*omega_b/delta**2)))
ax[1].set(xticks=np.linspace(-2,2,3), xlabel=r'$\omega/\omega_0$', xlim=(-3,3),
          ylabel=r'$S_\mathrm{BW}(\omega)$', ylim=(0,np.max(S_bw_list*omega_b/delta**2)))



t_before = np.arange(-1*a, 0-dt/10, dt)
t_after = np.arange(Tp*a, (Tp+1)*a-dt/10, dt)
t_tot = np.arange(-1*a, (Tp+1)*a-dt/10, dt)

N_rep = 0
np.random.seed(2)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_fw_list[N_rep])
alpha = 1+ dalpha

alpha_tot = np.hstack([np.ones_like(t_before), alpha, np.ones_like(t_after)])
sys = TMM.time_dep(dt*np.ones_like(t), 1/alpha, np.ones_like(t), 1, 1)
psi_sca = sys.evolution(omega_b)[0][:-1] - np.exp(-1j*omega_b * t)
P_sca = np.abs(psi_sca)**2
trans, reflec = sys.evolution(omega_b)[3:5]
psi_sca_after = (trans-np.exp(-1j*omega_b*Tp))*np.exp(-1j*omega_b * (t_after-Tp)) + reflec*np.exp(1j*omega_b * (t_after-Tp))
P_sca_before = np.zeros_like(t_before)
P_sca_after = np.abs(psi_sca_after)**2
P_sca_tot = np.hstack([P_sca_before, P_sca, P_sca_after]) 
depsilon = 1/alpha_tot - 1
ax[2].fill_between(t_tot, np.real(depsilon)/delta, -4, color='grey', edgecolor='k', lw=0.5,  alpha=0.4)
ax[2].set(xlabel=r'$t/t_0$', xlim=(-0.5,Tp+1), xticklabels=[],xticks=(0,5,10,15,20),
          ylim=(-3.5,3.5), yticks=(-2,0,2))
ax[2].set_title('Realizations', fontsize=7)

c = cmap1(N_rep/(N_list-1)*0.999)

axt2 = ax[2].twinx()
axt2.plot(t_tot, P_sca_tot/delta**2, c=c, lw=0.75)
axt2.set(ylim=(0,np.max(P_sca_tot/delta**2)), yticks=(0, 5, 10), yticklabels=[0, 5, r'10  '])
axt2.tick_params('y', colors=c)
axt2.spines["right"].set_edgecolor(c)


N_rep = 10
np.random.seed(2)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_fw_list[N_rep])
alpha = 1+ dalpha

alpha_tot = np.hstack([np.ones_like(t_before), alpha, np.ones_like(t_after)])
sys = TMM.time_dep(dt*np.ones_like(t), 1/alpha, np.ones_like(t), 1, 1)
psi_sca = sys.evolution(omega_b)[0][:-1] - np.exp(-1j*omega_b * t)
P_sca = np.abs(psi_sca)**2
trans, reflec = sys.evolution(omega_b)[3:5]
psi_sca_after = (trans-np.exp(-1j*omega_b*Tp))*np.exp(-1j*omega_b * (t_after-Tp)) + reflec*np.exp(1j*omega_b * (t_after-Tp))
P_sca_before = np.zeros_like(t_before)
P_sca_after = np.abs(psi_sca_after)**2
P_sca_tot = np.hstack([P_sca_before, P_sca, P_sca_after]) 
depsilon = 1/alpha_tot - 1
ax[3].fill_between(t_tot, np.real(depsilon)/delta, -4, color='grey', edgecolor='k', lw=0.5, alpha=0.4)
ax[3].set(xlabel=r'$t/t_0$', xlim=(-0.5,Tp+1), xticklabels=[],xticks=(0,5,10,15,20),
          ylabel=r'$\Delta\epsilon(t)/\delta$ ', ylim=(-3.5,3.5), yticks=(-2,0,2))

c = cmap1(N_rep/(N_list-1)*0.999)

axt3 = ax[3].twinx()
axt3.plot(t_tot, P_sca_tot/delta**2, c=c, lw=0.75)
axt3.set(ylim=(0,np.max(P_sca_tot/delta**2)), yticks=(0,50, 100, 150))
axt3.set_ylabel(r'$|\psi_\mathrm{sca}(t)|^2$', color='k')
axt3.tick_params('y', colors=c)
axt3.spines["right"].set_edgecolor(c)



N_rep = 10
np.random.seed(0)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_bw_list[N_rep])
alpha = 1+ dalpha

alpha_tot = np.hstack([np.ones_like(t_before), alpha, np.ones_like(t_after)])
sys = TMM.time_dep(dt*np.ones_like(t), 1/alpha, np.ones_like(t), 1, 1)
psi_sca = sys.evolution(omega_b)[0][:-1] - np.exp(-1j*omega_b * t)
P_sca = np.abs(psi_sca)**2
trans, reflec = sys.evolution(omega_b)[3:5]
psi_sca_after = (trans-np.exp(-1j*omega_b*Tp))*np.exp(-1j*omega_b * (t_after-Tp)) + reflec*np.exp(1j*omega_b * (t_after-Tp))
P_sca_before = np.zeros_like(t_before)
P_sca_after = np.abs(psi_sca_after)**2
P_sca_tot = np.hstack([P_sca_before, P_sca, P_sca_after]) 
depsilon = 1/alpha_tot - 1
ax[4].fill_between(t_tot, np.real(depsilon)/delta, -4, color='grey', edgecolor='k', lw=0.5,  alpha=0.4)
ax[4].set(xlabel=r'$t/t_0$', xlim=(-0.5,Tp+1), xticks=(0,5,10,15,20),
          ylim=(-3.5,3.5), yticks=(-2,0,2))

c = cmap2(N_rep/(N_list-1)*0.999)

axt4 = ax[4].twinx()
axt4.plot(t_tot, P_sca_tot/delta**2, c=c, lw=0.75)
axt4.set(ylim=(0,np.max(P_sca_tot/delta**2)), yticks=(0, 50, 100))
axt4.tick_params('y', colors=c)
axt4.spines["right"].set_edgecolor(c)






P_ctrlfw_fw_mean = np.average(P_ctrlfw_fw_ens, axis=1)
P_ctrlfw_fw_q1 = np.percentile(P_ctrlfw_fw_ens, 25, axis=1)
P_ctrlfw_fw_q3 = np.percentile(P_ctrlfw_fw_ens, 75, axis=1)

P_ctrlfw_bw_mean = np.average(P_ctrlfw_bw_ens, axis=1)
P_ctrlfw_bw_q1 = np.percentile(P_ctrlfw_bw_ens, 25, axis=1)
P_ctrlfw_bw_q3 = np.percentile(P_ctrlfw_bw_ens, 75, axis=1)

P_ctrlbw_fw_mean = np.average(P_ctrlbw_fw_ens, axis=1)
P_ctrlbw_fw_q1 = np.percentile(P_ctrlbw_fw_ens, 25, axis=1)
P_ctrlbw_fw_q3 = np.percentile(P_ctrlbw_fw_ens, 75, axis=1)

P_ctrlbw_bw_mean = np.average(P_ctrlbw_bw_ens, axis=1)
P_ctrlbw_bw_q1 = np.percentile(P_ctrlbw_bw_ens, 25, axis=1)
P_ctrlbw_bw_q3 = np.percentile(P_ctrlbw_bw_ens, 75, axis=1)



ax[5].plot(S0*omega_b, 0.25*Int_1d_stat_ctrlfw[:,0]*omega_b*Tp/delta**2, 'm', lw=0.75, label=r'Eq.(6)')
ax[7].plot(S0*omega_b, 0.25*Int_1d_stat_ctrlfw[:,1]*omega_b*Tp/delta**2, 'm', lw=0.75, label=r'Eq.(6)')

ax[6].plot(S2w*omega_b, 0.25*Int_1d_stat_ctrlbw[:,0]*omega_b*Tp/delta**2, 'm', lw=0.75, label=r'Eq.(6)')
ax[8].plot(S2w*omega_b, 0.25*Int_1d_stat_ctrlbw[:,1]*omega_b*Tp/delta**2, 'm', lw=0.75, label=r'Eq.(6)')

for ii in range(11):
    c1 = cmap1(ii/(N_list-1)*0.999)
    c1_err = 1-(1-np.array(c1[:3]))*0.5
    c2 = cmap2(ii/(N_list-1)*0.999)
    c2_err = 1-(1-np.array(c2[:3]))*0.5
    
    if ii==0:
        ax[5].errorbar(S0[ii]*omega_b, P_ctrlfw_fw_mean[ii]/delta**2, 
                   yerr=np.array([[P_ctrlfw_fw_mean[ii]-P_ctrlfw_fw_q1[ii]],
                                  [P_ctrlfw_fw_q3[ii]-P_ctrlfw_fw_mean[ii]]])/delta**2,
                   ecolor=c1_err, color=c1, fmt='o', lw=2, ms=2, label='TMM')
    else:
        ax[5].errorbar(S0[ii]*omega_b, P_ctrlfw_fw_mean[ii]/delta**2, 
                       yerr=np.array([[P_ctrlfw_fw_mean[ii]-P_ctrlfw_fw_q1[ii]],
                                      [P_ctrlfw_fw_q3[ii]-P_ctrlfw_fw_mean[ii]]])/delta**2,
                       ecolor=c1_err, color=c1, fmt='o', lw=2, ms=2)
    
    ax[7].errorbar(S0[ii]*omega_b, P_ctrlfw_bw_mean[ii]/delta**2, 
                   yerr=np.array([[P_ctrlfw_bw_mean[ii]-P_ctrlfw_bw_q1[ii]],
                                  [P_ctrlfw_bw_q3[ii]-P_ctrlfw_bw_mean[ii]]])/delta**2,
                   ecolor=c1_err, color=c1, fmt='o', lw=2, ms=2)
    
    ax[6].errorbar(S2w[ii]*omega_b, P_ctrlbw_fw_mean[ii]/delta**2, 
                   yerr=np.array([[P_ctrlbw_fw_mean[ii]-P_ctrlbw_fw_q1[ii]],
                                  [P_ctrlbw_fw_q3[ii]-P_ctrlbw_fw_mean[ii]]])/delta**2,
                   ecolor=c2_err, color=c2, fmt='o', lw=2, ms=2)
    
    ax[8].errorbar(S2w[ii]*omega_b, P_ctrlbw_bw_mean[ii]/delta**2, 
                   yerr=np.array([[P_ctrlbw_bw_mean[ii]-P_ctrlbw_bw_q1[ii]],
                                  [P_ctrlbw_bw_q3[ii]-P_ctrlbw_bw_mean[ii]]])/delta**2,
                   ecolor=c2_err, color=c2, fmt='o', lw=2, ms=2)


ax[5].set_title('Forward control', fontsize=7)
ax[5].set(xlim=np.array([-0.05,1.05])*S0[-1]*omega_b, xticklabels=[], xticks=(0,1,2,3),
          ylim=(0,np.max(P_ctrlfw_fw_q3/delta**2)), ylabel=r'$P_\mathrm{FW}$', yticks=(0,50,100))
ax[7].set(xlim=np.array([-0.05,1.05])*S0[-1]*omega_b, xlabel=r'$S_0$',xticks=(0,1,2,3),
          ylim=(0,np.max(P_ctrlfw_bw_q3/delta**2)), ylabel=r'$P_\mathrm{BW}$', yticks=(0,0.5,1), yticklabels=['0.0',0.5,' 1.0'])

ax[5].legend(frameon=False, fontsize=6, loc=2)

ax[6].set_title('Backward control', fontsize=7)
ax[6].set(xlim=np.array([-0.05,1.05])*S2w[-1]*omega_b, xticklabels=[],xticks=(0,1,2),
          ylim=(0,np.max(P_ctrlbw_fw_q3/delta**2)),  yticks=(0,0.5,1, 1.5))
ax[8].set(xlim=np.array([-0.05,1.05])*S2w[-1]*omega_b, xlabel=r'$S_{2\omega}$', xticks=(0,1,2),
          ylim=(0,np.max(P_ctrlbw_bw_q3/delta**2)), yticks=(0,25,50,75), yticklabels=[0,25,50,' 75'])

fig.savefig('fig2_target_ctrl_modified.pdf', format='pdf', dpi=1200)
fig.savefig('fig2_target_ctrl_modified.png', format='png', dpi=1200)