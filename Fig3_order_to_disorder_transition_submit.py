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





bump = lambda x: 1-(1-(np.abs(x)<1)* (x**2-1)**2)**1

Bragg_peak = lambda x: np.exp(-(4*x)**2/2)


fig, ax = plt.subplots(figsize=(3,2))
ax.plot(np.linspace(-1.5,1.5,3001), bump(np.linspace(-1.5,1.5,3001)), lw=1.5, color='y')
ax.plot(np.linspace(-1.5,1.5,3001), Bragg_peak(np.linspace(-1.5,1.5,3001)), lw=1.5, color='m')
ax.set(xlim=(-1.5,1.5), ylim=(0,1), xlabel=r'$\xi$', ylabel=r'Bump function $\Psi(\xi)$')



#%% Design: disorder to crystal

a = 1
omega_b = 2*np.pi / a
dt = a *0.01 

Tp = 20

tau = np.arange(-(Tp)*a, (Tp)*a+dt/10, dt)

delta = 0.01 / Tp
omega = 4*omega_b*np.linspace(-1,1, 8001)
domega = omega[1]-omega[0]




dod = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
logdod = np.log(dod)

doc = 1 - dod
N_list = len(dod)

bandwidth = 2*omega_b
S_disorder = bump((omega-2*omega_b)/bandwidth) +bump((omega+2*omega_b)/bandwidth)
factor_disorder = delta**2/np.sum(S_disorder*domega/(2*np.pi))

omega_c_list = omega_b*np.hstack([np.arange(0.5,2,0.5), np.arange(2.5,5,0.5)])
S_crystal = np.sum([bump((omega-omega_c)/(bandwidth/40))+bump((omega+omega_c)/(bandwidth/40)) for omega_c in omega_c_list], axis=0)
S_crystal *= 1/np.cosh(omega/(omega_b/2))

factor_crystal = delta**2/np.sum(S_crystal*domega/(2*np.pi))


S_list = (factor_disorder * bump(1/(bandwidth*dod).reshape(-1,1) @ (omega-2*omega_b).reshape(1,-1))
          + factor_disorder * bump(1/(bandwidth*dod).reshape(-1,1) @ (omega+2*omega_b).reshape(1,-1)) 
          + factor_crystal * doc.reshape(-1,1)@S_crystal.reshape(1,-1))

S_Poisson = np.ones_like(omega) * np.average(S_list[-1])

R_list = np.real([np.sum([s*np.exp(-1j*om*tau)*domega/(2*np.pi) for s, om in zip(S, omega)], axis=0) for S in S_list])

order_metric = np.sum((S_list - S_Poisson)**2, axis=1) * domega/a
log_om = np.log(order_metric)


#%%

fig, ax = plt.subplots(2,1)

ax[0].plot(omega/omega_b, S_disorder)
ax[1].plot(omega/omega_b, S_crystal)


cmap1 = matplotlib.cm.get_cmap('cividis')
newcolors = cmap1(np.linspace(0,0.9,256))
cmap = ListedColormap(newcolors)

fig, ax = plt.subplots(1, 2, figsize=(3.3,1.5))
for ii, (S, R) in enumerate(zip(S_list, R_list)):
    c = cmap(order_metric[ii]/order_metric[0])
    ax[0].plot(omega/omega_b, S/delta**2,  lw=0.75, c=c)
    ax[1].plot(tau, R/delta**2,  lw=0.75, c=c)
ax[0].set_title('Structure factor', fontsize=7)
ax[1].set_title('Correlation function', fontsize=7)
ax[0].set(xlabel=r'$\omega/\omega_\mathrm{b}$', xlim=(1.7,2.3), 
          ylabel=r'$S(\omega)/\delta^2$', ylim=(0, np.max(S_list)/delta**2/10))
ax[1].set(xlabel=r'$\tau/t_0$', xlim=(-4,4), 
          ylabel=r'$S(\tau)/\delta^2$', ylim=(-0.6, 1))
fig.tight_layout()


t_idx = np.arange(-round(Tp/dt), 0)
Rcenter_idx = np.argmin(np.abs(tau))


def int_2d_stat(R):
    Integrand_fw = np.array([R[Rcenter_idx+t_idx-t2] for t2 in t_idx])
    Integrand_bw = np.array([R[Rcenter_idx+t_idx-t2]*np.exp(2j*omega_b*(t_idx-t2)*dt) for t2 in t_idx])
    return np.abs( np.sum(Integrand_fw)* (omega_b*dt)**2 ), np.abs( np.sum(Integrand_bw)* (omega_b*dt)**2 )


def int_1d_stat(R):
    S0 = np.sum([r*np.exp(1j*0*tau0) *dt for r, tau0 in zip(R, tau)])
    S2w = np.sum([r*np.exp(1j*2*omega_b*tau0) *dt for r, tau0 in zip(R, tau)])
    return omega_b*S0, omega_b*S2w


with Pool(30) as pool:
    Int_2d_stat = np.array(pool.map(int_2d_stat, R_list))
    Int_1d_stat = np.array(pool.map(int_1d_stat, R_list))


#%% Block covariance matrix

t =  np.arange(0*a, (Tp-dt/10)*a, dt)
tic = time()
tau0_idx = np.argmin(np.abs(tau))

def row(R, i):
    return np.array(R[tau0_idx+(0-i):tau0_idx+(len(t)-i)] )

def complex_block(mat):
    return 0.5 * np.block([[np.real(mat), -np.imag(mat)],[np.imag(mat), np.real(mat)]])

with Pool(30) as pool:
    cov_alpha_list = [np.array(pool.starmap(row, zip([R]*len(t), range(len(t))))) for R in R_list]
    # cov_alpha_block_list = pool.map(complex_block, cov_alpha_list)







#%% SCattering ensemble simulation new

N_ens = 10000

k_list = omega_b * np.linspace(0.85, 1.15, 301)

dalpha_ens_total = []
for ii in range(N_list):
    np.random.seed(0)
    temp = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_list[ii], size=N_ens)
    dalpha_ens_total.append(temp)
dalpha_ens_total = np.vstack(dalpha_ens_total)


def power_flow(dalpha):
    alpha = 1+ dalpha
    sys = TMM.time_dep(dt*np.ones_like(t), 1/alpha, np.ones_like(t), 1, 1)
    
    temp = np.array([sys.evolution(k)[3:5] for k in k_list])
    trans, reflec = temp[:,0], temp[:,1]
    
    P_forward = np.abs(trans-np.exp(-1j*k_list*Tp))**2
    P_backward = np.abs(reflec)**2
    return P_forward, P_backward


tic = time()
with Pool(30) as pool:
    P_ens_total = np.array(pool.map(power_flow, dalpha_ens_total))

P_fw_ens_total, P_bw_ens_total = P_ens_total[:,0,:], P_ens_total[:,1,:]

toc = time()
print(toc-tic)


np.savetxt('Data_Fig3_bandwidth_fw_fine_mod.csv', np.real(P_fw_ens_total), delimiter=' ')
np.savetxt('Data_Fig3_bandwidth_bw_fine_mod.csv', np.real(P_bw_ens_total), delimiter=' ')



#%% Fig 3 

N_ens = 10000
k_list = omega_b * np.linspace(0.9, 1.1, 201)
P_fw_ens_total = np.loadtxt('Data_Fig3_bandwidth_fw_fine.csv',  delimiter=' ')
P_bw_ens_total = np.loadtxt('Data_Fig3_bandwidth_bw_fine.csv',  delimiter=' ')

cmap1 = matplotlib.cm.get_cmap('cividis_r')
newcolors = cmap1(np.linspace(0.15,1,256))
cmap = ListedColormap(newcolors)

fig = plt.figure(figsize=(3.375, 4))
gs = GridSpec(nrows=120, ncols=100)

ax = np.array([fig.add_subplot(gs[2:14,7:45]), fig.add_subplot(gs[2:14,55:93]),
               fig.add_subplot(gs[16:35,7:93]), 
                fig.add_subplot(gs[55:64,7:70]), 
                fig.add_subplot(gs[64:73,7:70]),
                fig.add_subplot(gs[73:82,7:70]),
                fig.add_subplot(gs[102:115,7:42]), fig.add_subplot(gs[102:115,58:93]),
                fig.add_subplot(gs[55:82,75:93]),])



for kk in [-1, 0,1]:
    ax[kk].spines['top'].set_visible(False)
    ax[kk].spines['right'].set_visible(False)
    ax[kk].spines['left'].set_visible(False)


omega_discrete = omega_b * np.array([-4,-3,5, -3, -2.5, -1.5, -1, -0.5, 0.5, 1, 1.5, 2.5, 3, 3.5, 4])
stem = 1/np.cosh(omega_discrete/(omega_b/2))
mks, sts, bsline = ax[0].stem(omega_discrete/omega_b, stem/1.1, linefmt='grey', markerfmt='o', basefmt=None, bottom=0)
plt.setp(mks, color=cmap(0.99), ms=2.5)
plt.setp(sts, color='grey', lw=0.5, ls='--')
plt.setp(bsline, visible=False)


ax[0].set(xlim=(-4,4), xticks=(-2,0,2), xticklabels=[],
          ylim=(0,np.max(stem)), yticks=[])


X, Y = np.meshgrid(0.98*omega[::40]/omega_b, np.linspace(0, 0.6* np.max(S_disorder), 101))
Z = np.random.normal(size=X.shape)

Zfft = np.fft.fft2(Z)
Zfft[10:-9, :]=0
Zfft[:,20:-19]=0
Z_filt = np.real(np.fft.ifft2(Zfft))
cmap2 = matplotlib.cm.get_cmap('cividis_r')
newcolors2 = cmap1(np.linspace(0,0.3,256))
cmap2 = ListedColormap(newcolors2)

ax[1].pcolormesh(X,Y,Z_filt, cmap=cmap2)

ax[1].fill_between(omega/omega_b, np.ones_like(omega)*np.max(S_disorder), S_disorder/2, 
                   color='white',  lw=0.5, edgecolor='none', zorder=100)
ax[1].set(xlim=(-4,4), xticks=(-2,0,2), xticklabels=[],
          ylim=(0,np.max(S_disorder)), yticks=[])



axins = ax[2].inset_axes([0.6, 0.25, 0.37, 0.72])
axins.set(xticks=[], yticks=[])

P_fw_ens_mean = np.array([np.average(P_fw_ens_total[ii*N_ens:(ii+1)*N_ens, :],axis=0) for ii in range(N_list)])
P_bw_ens_mean = np.array([np.average(P_bw_ens_total[ii*N_ens:(ii+1)*N_ens, :],axis=0) for ii in range(N_list)])

P_fw_ens_q1 = np.array([np.percentile(P_fw_ens_total[ii*N_ens:(ii+1)*N_ens, :],25, axis=0) for ii in range(N_list)])
P_fw_ens_q3 = np.array([np.percentile(P_fw_ens_total[ii*N_ens:(ii+1)*N_ens, :],75, axis=0) for ii in range(N_list)])
P_bw_ens_q1 = np.array([np.percentile(P_bw_ens_total[ii*N_ens:(ii+1)*N_ens, :],25, axis=0) for ii in range(N_list)])
P_bw_ens_q3 = np.array([np.percentile(P_bw_ens_total[ii*N_ens:(ii+1)*N_ens, :],75, axis=0) for ii in range(N_list)])

for ii in range(N_list):
    
    c = cmap(order_metric[ii]/order_metric[0])

    ax[2].plot(omega/omega_b, S_list[ii]*omega_b/delta**2, c=c, lw=0.75)
    axins.plot(omega/omega_b, S_list[ii]*omega_b/delta**2, c=c, lw=1.2)

ax[2].set(xlim=(-4,4), ylim=(0,28), xlabel=r'$\omega/\omega_0$', ylabel=r'$S(\omega)$')
axins.set(xlim=(1.6,2.4), ylim=(0,2.1))
ax[2].indicate_inset_zoom(axins, edgecolor="purple")
    

N_rep = 0
np.random.seed(3)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_list[N_rep])
depsilon = 1/(1+dalpha)-1
ax[3].fill_between(t, depsilon/delta, -3.2, color=cmap(order_metric[N_rep]/order_metric[0]), alpha=0.5, lw=0.5, edgecolor='k')

N_rep = 4
np.random.seed(3)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_list[N_rep])
depsilon = 1/(1+dalpha)-1
ax[4].fill_between(t, depsilon/delta, -3.2, color=cmap(order_metric[N_rep]/order_metric[0]), alpha=0.5, lw=0.5, edgecolor='k')

N_rep = 11
np.random.seed(3)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha_list[N_rep])
depsilon = 1/(1+dalpha)-1
ax[5].fill_between(t, depsilon/delta, -3.2, color=cmap(order_metric[N_rep]/order_metric[0]), alpha=0.5, lw=0.5, edgecolor='k')

ax[3].set_title('Realizations', fontsize=7)
ax[3].set(xlim=(0, 20), ylim=(-3.2,3.2),  xticklabels=[], yticks=[-2,0,2])
ax[4].set(xlim=(0, 20), ylim=(-3.2,3.2), ylabel=r'$\Delta\epsilon(t)/\delta$', xticklabels=[], yticks=[-2,0,2])
ax[5].set(xlim=(0, 20), ylim=(-3.2,3.2), xlabel=r'$t/t_0$', yticks=[-2,0,2])



for ii in [0, 2,  11]:
    c = cmap(order_metric[ii]/order_metric[0])
    ax[6].plot(k_list/omega_b, P_fw_ens_mean[ii]/delta**2, c=c, lw=1, zorder=100)
    ax[7].plot(k_list/omega_b, P_bw_ens_mean[ii]/delta**2, c=c, lw=1, zorder=100)
    ax[6].fill_between(k_list/omega_b, P_fw_ens_q3[ii]/delta**2, P_fw_ens_q1[ii]/delta**2, color=c, lw=0, alpha=0.4)
    ax[7].fill_between(k_list/omega_b, P_bw_ens_q3[ii]/delta**2, P_bw_ens_q1[ii]/delta**2, color=c, lw=0, alpha=0.4)
 
ax[6].set_title('Forward power', fontsize=7)
ax[6].set(xlim=(0.9, 1.1), xlabel=r'$kc/\omega_0$',
          ylim=(0,75), ylabel=r'$ P_\mathrm{FW}(k)$')
ax[7].set_title('Backward power', fontsize=7)
ax[7].set(xlim=(0.9, 1.1), xlabel=r'$kc/\omega_0$', 
          ylim=(0,75), ylabel=r'$ P_\mathrm{BW}(k)$')



axt8 = ax[8].twinx()
ax[8].set(yticks=[], xlabel=r'$P_\mathrm{BW}(\omega_0/c)$')
for ii in range(N_list):
    c = cmap(order_metric[ii]/order_metric[0])
    c_err = 1-(1-np.array(c[:3]))*0.5
    axt8.errorbar(P_bw_ens_mean[ii,100]/delta**2, order_metric[ii]/delta**4, 
           xerr=np.array([[P_bw_ens_mean[ii,100]-P_bw_ens_q1[ii,100]],
                          [P_bw_ens_q3[ii,100]-P_bw_ens_mean[ii,100]]])/delta**2,
           ecolor=c_err, color=c, fmt='o', lw=2, ms=2)



axt8.plot(0.25*Int_2d_stat[:,1]/delta**2, order_metric/delta**4, color='indigo', lw=0.75, ls='-',  label='Eq. (5)')
axt8.axvline(x=0.25*Int_1d_stat[0,1]*omega_b*Tp/delta**2, color='m', lw=0.75, ls='--', label='Eq. (6)', zorder=300)

axt8.set(xlim=(0, 65),
         ylim=(-0.5,20.5), ylabel=r'$\tau$')
axt8.legend(fontsize=6, ncol=1, frameon=False, bbox_to_anchor=(0.45,1.4) , loc='upper center')




fig.tight_layout()


fig.savefig('fig3_bandwidth.pdf', format='pdf', dpi=1200)
fig.savefig('fig3_bandwidth.png', format='png', dpi=1200)