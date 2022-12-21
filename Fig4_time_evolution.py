import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation

import math
from multiprocessing import Pool


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




#%% Initial wave

dt = 0.01
Xp = 200

ct = np.arange(-Xp/2, Xp/2-dt/10, dt)
Nx = len(ct)

dk = 2*np.pi/Xp
k = (np.arange(Nx)-Nx/2)*dk 
k0 = 2*np.pi/1

sigma_x = 1

D_init = np.exp(-0.5* (ct/sigma_x)**2) 
D_init_F = np.fft.fftshift( np.fft.fft(np.fft.ifftshift(D_init))*dt  )


fig, ax = plt.subplots(2,1, figsize=(3.3,2))
ax[0].plot(ct, np.real(D_init), lw=0.75)
ax[0].set(xlim=(-Xp/2,Xp/2), xlabel=r'$ct/a$', ylabel=r'$D(x, t=0)$')

ax[1].plot(k/k0, D_init_F, lw=0.75)
ax[1].set(xlim=(-2,2), xlabel=r'$k/k_0$', ylabel=r'$\tilde{D}(k_x, t=0)$')

fig.tight_layout()

#%% Structure factor

a = 1
Tp = 100
tau = np.arange(-(Tp)*a, (Tp)*a+dt/10, dt)


delta = 0.0001
omega = np.linspace(-1.2*k0, 1.2*k0, 2401)
domega = omega[1]-omega[0]

omega1 = 0.5*k0
omega2 = 1*k0
omega_center = (omega1 + omega2)/2
sigma = 2/np.abs(omega2-omega1)

rect = lambda x: (np.abs(x)<0.5) + (np.abs(x)==0.5)*0.5


S = rect( (np.abs(omega)-0.75*k0)/(0.5*k0)) * delta**2 * np.pi * k0/(omega+k0*1e-10)**2 #sq
R = np.sum([s*np.exp(-1j*om*tau)*domega/(2*np.pi) for s, om in zip(S, omega)], axis=0)


fig, ax = plt.subplots(2,1, figsize=(3.3,2))
ax[0].plot(omega/k0, S/delta**2)
ax[0].set(xlim=(-2.5,2.5))
ax[1].plot(tau, R/delta**2)
ax[1].set(xlim=(-5,5))

fig.tight_layout()


#%% realization generation

t =  np.arange(0*a, (Tp-dt/10)*a, dt)
tau0_idx = np.argmin(np.abs(tau))

def row(R, i):
    return np.array(R[tau0_idx+(0-i):tau0_idx+(len(t)-i)] )

with Pool(30) as pool:
    cov_alpha = np.array(pool.starmap(row, zip([R]*len(t), range(len(t))))) 



np.random.seed(2)
dalpha = np.random.multivariate_normal(np.zeros_like(np.hstack([t])), cov_alpha)

smooth = 1/(1 + np.exp(-(t-Tp*0.1)/(0.25*a)))  + 1/(1 + np.exp((t-Tp*0.9)/(0.25*a))) - 1 
alpha = 1+ smooth* dalpha 



fig, ax = plt.subplots(3,1, figsize=(3.3,3))
ax[0].plot(omega/k0, S/delta**2, lw=0.75)
ax[0].set(xlabel=r'$\omega/ck_0$', xlim=(-1,4), ylabel=r'$S(\omega)/\delta^2$')

ax[1].plot(tau, np.real(R)/delta**2, lw=0.75)
ax[1].plot(tau, np.imag(R)/delta**2, lw=0.75)
ax[1].set(xlabel=r'$\tau/t_0$', xlim=(-4,4), ylabel=r'$S(\tau)/\delta^2$')

ax[2].plot(t, (np.real(alpha)-1)/delta, lw=0.75)
ax[2].plot(t, np.imag(alpha)/delta, lw=0.75)
ax[2].set(xlabel=r'$t/t_0$', xlim=(0, 100), ylabel=r'$\Delta\alpha(t)/\delta$')

fig.tight_layout()


#%% Time evolution

t_ext = dt*np.arange(len(t)*2)
alpha_ext = np.hstack([alpha, np.ones_like(alpha)])

sys = TMM.time_dep(dt*np.ones_like(t_ext), 1/alpha_ext, np.ones_like(t_ext), 1, 1)
sys_inc = TMM.time_dep(dt*np.ones_like(t_ext), np.ones_like(t_ext), np.ones_like(t_ext), 1, 1)

def evol_function(k, D):
    if k>=0:
        return D*sys.evolution(k, 'f')[0]
    else:
        return D*sys.evolution(k, 'b')[0]

def evol_function_inc(k, D):
    if k>=0:
        return D*sys_inc.evolution(k, 'f')[0]
    else:
        return D*sys_inc.evolution(k, 'b')[0]

with Pool(30) as pool:
    D_evol = pool.starmap(evol_function, zip(k, D_init_F))
    D_evol_inc = pool.starmap(evol_function_inc, zip(k, D_init_F))

D_evol = np.column_stack(D_evol)
D_evol_inc = np.column_stack(D_evol_inc)

D_evol_sca = D_evol - D_evol_inc
  
D = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(D_evol, axes=1), axis=1)/dt,axes=1)
D_inc = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(D_evol_inc, axes=1), axis=1)/dt,axes=1)

D_sca = D - D_inc
P_sca = np.abs(D_sca)**2


psi_sca_z0 = D_sca[:-1,10000]
psi_sca_z0_F = np.sum([psi0*np.exp(1j*omega*t0)*dt for psi0, t0 in zip(psi_sca_z0, t_ext)], axis=0)



#%% Fig 4


cmap1 = matplotlib.cm.get_cmap('inferno')
newcolors = cmap1(np.linspace(0,0.85,256))
cmap = matplotlib.colors.ListedColormap(newcolors)

fig = plt.figure(figsize=(3.375*2, 2.7))
gs = GridSpec(nrows=105, ncols=200)
ax = np.array([[fig.add_subplot(gs[5:15,57:95]), fig.add_subplot(gs[5:15,110:148]), fig.add_subplot(gs[5:25,2:33]), fig.add_subplot(gs[5:42,168:198])],
               [fig.add_subplot(gs[15:25,57:95]), fig.add_subplot(gs[15:25,110:148]), fig.add_subplot(gs[40:60,2:33]), fig.add_subplot(gs[58:95,168:198])],
               [fig.add_subplot(gs[25:35,57:95]), fig.add_subplot(gs[25:35,110:148]), fig.add_subplot(gs[75:95,2:33]), np.nan],
               [fig.add_subplot(gs[35:45,57:95]), fig.add_subplot(gs[35:45,110:148]), np.nan, np.nan],
               [fig.add_subplot(gs[45:55,57:95]), fig.add_subplot(gs[45:55,110:148]), np.nan, np.nan],
               [fig.add_subplot(gs[55:65,57:95]), fig.add_subplot(gs[55:65,110:148]), np.nan, np.nan],
               [fig.add_subplot(gs[65:75,57:95]), fig.add_subplot(gs[65:75,110:148]), np.nan, np.nan],
               [fig.add_subplot(gs[75:85,57:95]), fig.add_subplot(gs[75:85,110:148]), np.nan, np.nan],
               [fig.add_subplot(gs[85:95,57:95]), fig.add_subplot(gs[85:95,110:148]), np.nan, np.nan],])


ax[0,0].set_title(r'Total field', fontsize=7)
ax[0,1].set_title(r'Scattering power', fontsize=7)
for i, ti in enumerate(1250*np.arange(9)):
    c = cmap(0.999*ti/len(t))
    ax[i,0].plot(ct-ct[0], np.abs(np.fft.fftshift(np.real(D[ti]))), lw=0.75, color=c)
    ax[i,0].set_yscale('log')
    ax[i,0].set(xlim=(0, Tp), ylim=(1e-6,np.max(np.abs(D))), yticks=[1e-5, 1e-2])
    ax[i,1].plot(ct-ct[0], np.fft.ifftshift(P_sca[ti])/delta**2, lw=0.75, color=c)
    

    if i==8:
        ax[i,1].set(xlim=(0, Tp), ylim=(0,np.max(P_sca)/delta**2), yticks=(0,0.2))
    else:
        ax[i,1].set(xlim=(0, Tp), ylim=(0,np.max(P_sca)/delta**2), yticks=(0,0.2))
    
    if i<8:
        ax[i,0].set(xticklabels=[])
        ax[i,1].set(xticklabels=[])
    else:
        ax[i,0].set(xlabel=r'$z/ct_0$')
        ax[i,1].set(xlabel=r'$z/ct_0$')  

ax[4,0].set(ylabel=r'$|D_\mathrm{tot}(z)|$')
ax[4,1].set(ylabel=r'$|D_\mathrm{sca}(z)|^2$')

    
ax[0,2].set_title(r'Design', fontsize=7)
ax[0,2].plot(omega/k0, S*k0/delta**2, 'k', lw=0.75)
ax[0,2].set(xlim=(-1.2, 1.2), xlabel=r'$\omega /\omega_0$', 
            ylim=(0,12), ylabel=r'$S(\omega)$')

axt2 = ax[0,2].twinx()
axt2.fill_between(omega/k0, S>0, 0, color='midnightblue', alpha=0.2, edgecolor=None)
axt2.set(xlim=(-1.2, 1.2), ylim=(0,1), yticks=[])

ax[1,2].plot(tau, R/delta**2, 'k', lw=0.75)
ax[1,2].set(xlim=(-10, 10), xlabel=r'$\Delta t / t_0$', 
            ylim=(np.min(np.real(R/delta**2)),1.05), ylabel=r'$C(\Delta t) $', 
            yticks=[0,1], yticklabels=[0, ' $\delta^2$'])

ax[2,2].fill_between(t, np.real(1/alpha-1)/delta, -2.5, color='grey', alpha=0.5, edgecolor='k', lw=0.5)

ax[2,2].set(xlim=(5,30), ylim=(-2.5,2.5), yticks=(-2,0,2), 
            xlabel=r'$t/t_0$', ylabel=r'$\Delta\epsilon(t)/\delta$')



axins = ax[0,3].inset_axes([0.4,0.15,0.55,0.8])
axins.set_facecolor('white')
axt = ax[0,3].twinx()
axinst = axins.twinx()


scale_factor = np.exp(-(sigma_x*0.25*k0)**2/2)

for ii, t_snapshot in enumerate(np.arange(9)*12.5):
    c = cmap(0.999*t_snapshot/Tp) 
    ax[0,3].plot(k/k0, np.abs(D_evol_sca[int(t_snapshot/dt)])*k0/delta, lw=0.75, color=c)
    axins.plot(k/k0, np.abs(D_evol_sca[int(t_snapshot/dt)])*k0/delta, lw=0.75, color=c)

ratio_conv = np.max(D_init_F*k0/delta)/(np.max(np.abs(D_evol_sca)*k0/delta)/scale_factor)
ax[0,3].plot(k/k0, np.abs(D_init_F)*k0/delta/ratio_conv, 'dodgerblue', ls='--', lw=0.75)
axinst.plot(k/k0, np.abs(D_init_F)*k0/delta, 'dodgerblue', ls='--', lw=0.75)

axt.set_title('Scattering field', fontsize=7)
ax[0,3].set(xlim=(-0.6,0.6), xlabel=r'$kc/\omega_0$',
       ylim=(0,np.max(np.abs(D_evol_sca)*k0/delta)/scale_factor), ylabel=r'$D_\mathrm{sca}(k)$')

axt.set(xlim=(-0.6,0.6), 
        ylim=(0,np.max(D_init_F*k0)))
axt.set_ylabel(r'$D_\mathrm{inc}(k)$  ($\times 10^4$)', color='dodgerblue')
axt.tick_params('y', colors='dodgerblue')
axt.spines["right"].set_edgecolor('dodgerblue')


axins.set(xlim=(-0.55,-0.2), xticks=[],
          ylim=(0,np.max(np.abs(D_evol_sca)*k0/delta)), yticks=[])


axinst.set(xlim=(-0.55,-0.2), xticks=[],
           ylim=(0,np.max(D_init_F*k0/delta)*scale_factor), yticks=[])
ax[0,3].indicate_inset_zoom(axins, edgecolor='k', lw=0.75)



ax[1,3].plot(omega/k0, np.abs(psi_sca_z0_F)*k0/delta, color='navy', lw=0.75)
axt = ax[1,3].twinx()
S2omega = rect( (np.abs(omega)-0.375*k0)/(0.25*k0) ) 
axt.fill_between(omega/k0, S2omega, 0, color='midnightblue', alpha=0.2, edgecolor=None)

ax[1,3].set(xlabel=r'$\omega / \omega_0 $', xlim=(0,0.6), ylim=(0,np.max(np.abs(psi_sca_z0_F)*k0/delta)),
       ylabel=r'$|D_\mathrm{sca}(\omega)|$')
axt.set(xlim=(0,0.6), ylim=(0,1), yticks=[])



fig.tight_layout()
fig.savefig('fig4.pdf', dpi=1200, format='pdf')


fig_cb, cax = plt.subplots(figsize=(3.375*2, 2.7))
norm = matplotlib.colors.Normalize(vmin=0,vmax=100)
cb = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=cax,  
                  orientation='horizontal', shrink=0.07)
cb.set_label(r'$t/t_0$', fontsize=7)
fig_cb.savefig('fig4_colorbar.pdf', dpi=1200, format='pdf')







#%% Supp Movie: Animation


fig = plt.figure(figsize=(5, 2.5))

gs = GridSpec(nrows=100, ncols=100)
ax = np.array([fig.add_subplot(gs[10:20, 10:52]),
               fig.add_subplot(gs[20:47, 10:52]), 
               fig.add_subplot(gs[53:63, 10:52]),
               fig.add_subplot(gs[63:90, 10:52]), 
               fig.add_subplot(gs[10:90,68:95]),])

axt = ax[4].twinx()


ax[0].set(xlim=(0,Tp),  xticks=[], yticks=[])
ax[1].set(xlim=(0,Tp),  xticklabels=[],
          ylim=(1e-6,np.max(np.abs(D))), ylabel=r'$|D_\mathrm{tot}(z,t)|$')
ax[1].set_yscale('log')

ax[2].set(xlim=(0,Tp),  xticks=[], yticks=[])
ax[3].set(xlim=(0,Tp), xlabel=r'$z/ct_0$',
          ylim=(0,np.max(P_sca)/delta**2), ylabel=r'$|D_\mathrm{sca}(z,t)|^2$')



ax[4].set(xlim=(-0.6, 0.6), xlabel=r'$k/c\omega_0$',
          ylim=(0, 1.5*np.max(np.abs(D_evol_sca)/delta)), ylabel=r'$|D_\mathrm{sca}(k,t)|$',)
axt.set(xlim=(-0.6,0.6), ylim=(0,np.max(np.abs(D_init_F)*k0)))
axt.set_ylabel(r'$D_\mathrm{inc}(k) $', color='dodgerblue')
axt.tick_params('y', colors='dodgerblue')
axt.spines["right"].set_edgecolor('dodgerblue')


scale_factor = np.exp(-(sigma_x*0.25*k0)**2/2)
scale_factor2 = np.max(np.abs(D_evol_sca)/(delta))/scale_factor /np.max(np.abs(D_init_F))
ax[4].plot(k/k0, np.abs(D_init_F)*k0*scale_factor2, 'dodgerblue', ls='--', lw=0.75)
ax[4].set(xlim=(-0.6,0.6), xlabel=r'$kc/\omega_0$',
            ylim=(0,np.max(np.abs(D_evol_sca)*k0/delta)/scale_factor), ylabel=r'$|D_\mathrm{sca}(k)|$')


line0, = ax[1].plot([], [], lw=0.5)
line1, = ax[3].plot([], [], lw=0.5) 
line2, = ax[4].plot([], [], lw=1)

Z, X = np.meshgrid(ct[::10], np.linspace(0,1,2))
pcm0 = ax[0].pcolormesh(Z, X, Z*0, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
pcm1 = ax[2].pcolormesh(Z, X, Z*0, cmap=plt.cm.RdBu_r, vmin=-0.62, vmax=0.62)


title = ax[4].text(-0.2, 3.2, "")

def init():
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    return (line0, line1, line2,)

def animate(i):
    c = cmap(0.999*i/len(t))
    line0.set_data(ct, np.abs(D[i]))
    line1.set_data(ct, P_sca[i]/delta**2)
    line2.set_data(k/k0,  np.abs(D_evol_sca[i])*k0/delta)

    
    line0.set_color(c)
    line1.set_color(c)
    line2.set_color(c)
    title.set_text(r'$t/t_0 ='+str(math.floor(t[i]))+'$')
    
    pcm0.set_array( np.array([np.real(D[i, ::10])]*2) )
    pcm1.set_array( np.array([np.real(D_sca[i, ::10]/delta)]*2) )
    
    return (line0, line1, line2,)


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=range(0,len(t),15), interval=1, blit=True)
anim.save('Supp_Movie.gif', fps=40, dpi=300)



