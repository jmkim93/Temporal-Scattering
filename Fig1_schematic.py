import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
from scipy.integrate import odeint, solve_ivp, DOP853
from scipy.interpolate import interp1d


from multiprocessing import Pool

import TMM

matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams.update({'axes.linewidth': 0.5})
matplotlib.rcParams.update({'ytick.major.width': 0.5})
matplotlib.rcParams.update({'xtick.major.width': 0.5})
matplotlib.rcParams.update({'ytick.minor.width': 0.5})
matplotlib.rcParams.update({'xtick.minor.width': 0.5})
matplotlib.rcParams.update({'ytick.direction': 'in'})
matplotlib.rcParams.update({'xtick.direction': 'in'})
matplotlib.rcParams.update({'figure.figsize': [3.375 *2, 2.5 * 1.8]})
matplotlib.rcParams.update({'savefig.pad_inches': 0.02}) #default 0.1
matplotlib.rcParams.update({'savefig.format': 'png'}) #default 0.1
matplotlib.rcParams.update({'figure.subplot.bottom': 0.05,
                            'figure.subplot.hspace': 0,
                            'figure.subplot.left': 0.05,
                            'figure.subplot.right': 0.95,
                            'figure.subplot.top': 0.95,
                            'figure.subplot.wspace': 0}) #default 0.125, 0.2, 0.125, 0.9, 0.88, 0.2




#%% Example disorder calculation

a = 1
dt = 0.004 * a
t = np.arange(-1*a, 11.5*a, dt)
k0 = 2*np.pi/a

alpha_b = 1
delta = 0.05
sigma = 0.15*a

i = np.arange(len(t))
def column(j):
    # return np.sinc(np.abs(i-j)*cdt/sigma)   # sinc
    return np.exp(-0.5* ((i-j)*dt/sigma)**2)   # Gaussian
pool = Pool(31)
cov_alpha = (delta**2) * np.array(pool.map(column, np.arange(len(t))))
pool.close()

np.random.seed(24)
mu_alpha = np.ones(len(t)) * alpha_b
alpha = np.random.multivariate_normal(mu_alpha, cov_alpha)

smooth = 1/(1 + np.exp(-(t-0*a)/(0.02*a))) + 1/(1 + np.exp((t-10*a+0*a)/(0.02*a)))-1
alpha_s = alpha_b + (alpha - alpha_b)*smooth
epsilon_s = 1/alpha_s

def diff(x, dt):
    diffx_init = (x[1]-x[0])/dt
    diffx_final = (x[-1]-x[-2])/dt
    diffx_med = (x[2:] - x[:-2])/(2*dt)
    return np.hstack([diffx_init, diffx_med, diffx_final])
    

sys = TMM.time_dep(dt*np.ones_like(t), epsilon_s, np.ones_like(t), epsilon_s[0], 1)
psi = sys.evolution(k0)[0][:-1]
diff_alpha = diff(alpha_s, dt)
diff_psi = diff(psi, dt)

P1 = np.abs(psi)**2
P2 = np.abs(diff_psi / k0)**2

u_M = P2/4
u_E = (alpha_s*P1 - np.cumsum(diff_alpha*P1*dt) )/4
u_E0 = (alpha_s*P1  )/4

u_EM = u_E + u_M

u_EM0 = (alpha_s*P1 + P2)/4
power = diff(u_EM0, dt)

power_max = np.max(np.abs(power[2:-2]))

power_in = ma.array(power, mask=(power<+0.002*power_max))
power_out = ma.array(power, mask=(power>-0.002*power_max))



#%% Fig 1 

fig = plt.figure(figsize=(3.375, 2.8))
gs = GridSpec(nrows=30, ncols=20)

ax = np.array([fig.add_subplot(gs[0:7, 2:9]), fig.add_subplot(gs[0:7, 11:18]),
               fig.add_subplot(gs[10:17, 2:18]), 
               fig.add_subplot(gs[18:28, 2:18])])

cmap = matplotlib.cm.get_cmap('Purples')
wt = np.linspace(-2*np.pi, 2*np.pi, 201)
G = (wt<0) * 0 + (wt>=0) * np.sin(wt)
for kx in np.arange(0, np.pi, np.pi/10):
    c = cmap(kx/(np.pi)*0.99)
    ax[0].plot(wt, np.real(np.exp(1j*kx)*G ), color=c, lw=1)

ax[0].arrow(-2.1*np.pi, 0, 4.2*np.pi, 0, width=0.0001, head_width=0.1)
ax[0].axvline(x=0, color='k', lw=0.5, ls='--')
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False) 
ax[0].set(xticks=[], yticks=[], xlabel=r'$t=t^\prime$')  

    
cmap = matplotlib.cm.get_cmap('Greys')
kx = np.linspace(-2*np.pi, 2*np.pi, 201)
G = (kx<0) * np.exp(-1j * kx) + (kx>=0) * np.exp(1j * kx)
for wt in np.arange(0, np.pi, np.pi/10):
    c = cmap(wt/(np.pi)*0.99)
    ax[1].plot(kx, np.real(np.exp(-1j*wt)*G ), color=c, alpha=0.6,  lw=1)

ax[1].arrow(-2.1*np.pi, 0, 4.2*np.pi, 0, width=0.0001, head_width=0.1)
ax[1].axvline(x=0, color='k', lw=0.5, ls='--')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['bottom'].set_visible(False) 
ax[1].set(xticks=[], yticks=[], xlabel=r'$z = z^\prime$') 


axt = ax[3].twinx()

ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].spines['bottom'].set_visible(False)
ax[2].axvline(x=11, lw=0.75, color='k', ls='--')
ax[2].axhline(y=power_max*0.85, lw=0.75, color='k', ls='--')
ax[2].axhline(y=-power_max*0.85, lw=0.75, color='k', ls='--')

c_face = np.array(matplotlib.colors.to_rgba('crimson'))
c_face[3] = 0.2
ax[2].set_facecolor([0.5,0.5,0.5,0.1])

ax[3].spines['top'].set_visible(False)
ax[3].spines['right'].set_visible(False)
axt.spines['top'].set_visible(False)
axt.spines['left'].set_visible(False)


ax[2].fill_between(t[2:-2], power_in[2:-2],0, color='orange', lw=0.5, alpha=0.7, edgecolor='k')
ax[2].fill_between(t[2:-2], power_out[2:-2],0, color='lightseagreen', lw=0.5, alpha=0.7, edgecolor='k')
ax[2].axhline(y=0, lw=0.5, color='k', ls='-')


ax[2].set(xlim=(-1,11), xticks=[], 
          ylabel=r'$P_\mathrm{in}(t)$', yticks=[0], yticklabels=['   0'],
          ylim=np.array([-1,1])*power_max*0.85)

ax[3].fill_between(t, epsilon_s,1-3*delta, color='grey', label=r'$\epsilon(t)$', edgecolor='k', alpha=0.6, lw=0.3)
ax[3].set(xlim=(-1,11), 
          ylim=(1-3*delta, 1+3*delta), ylabel=r'$\epsilon(t)$')
ax[3].set(xticks=(0, 10), xticklabels=[r'$0$', '$T$'], xlabel=r'Time $t$')


twincolor = 'indigo'
axt.plot(t[1:-1], u_EM0[1:-1], color=twincolor, lw=0.75, label=r'$u_\mathrm{E}^\mathrm{(0)}(t)$')
axt.set(xlim=(-1,11), ylim=(0.462,0.538), yticks=np.linspace(0.48,0.52,3))
axt.tick_params('y', colors=twincolor)
axt.set_ylabel(r'$u_\mathrm{EM}^0(t)$', fontsize=7, color=twincolor)
axt.spines["right"].set_edgecolor(twincolor)


fig.tight_layout()
fig.savefig('Fig1.pdf', format='pdf', dpi=1200)


