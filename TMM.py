import numpy as np
from numpy.linalg import inv

class time_dep():
    def __init__(self, T_list, eps_list, mu_list, eps_amb, mu_amb):
        self.num_layer = len(T_list)
        self.interval = np.array(T_list)
        self.eps = np.array(eps_list)
        self.mu = np.array(mu_list)
        self.eps_amb = eps_amb
        self.mu_amb = mu_amb
        
        self.n = np.sqrt(0j + self.eps * self.mu)
        self.Z = np.sqrt(0j + self.mu / self.eps)
        self.n_amb = np.sqrt(0j + self.eps_amb * self.mu_amb)
        self.Z_amb = np.sqrt(0j + self.mu_amb / self.eps_amb)
        
        self.t = np.array([ sum(self.interval[:i]) for i in range(self.num_layer+1)])
        
    # Time-domain transfer matrix method (TD-TMM)    
    def T_matrix(self, k):
        omega_0 = k
        omega = omega_0 / self.n
        
        T = np.eye(2)
        A_amb = np.array([[1,1], [self.Z_amb, -self.Z_amb]])
        T = A_amb @ T
        
        for i in range(self.num_layer):
            A = np.array([[1,1], [self.Z[i], -self.Z[i]]])    # Transmission matrix at interface
            B = np.array([[np.exp(-1j*omega[i]*self.interval[i]), 0],[0, np.exp(1j*omega[i]*self.interval[i])]])    # Propagation matrix
            T = B @ inv(A) @ T
            T = A @ T
        T = inv(A_amb) @ T
        return T
    
    # Transfer matrix to Scattering matrix
    def S_matrix(self, k):
        omega_0 = k
        omega = omega_0 / self.n
        
        T = np.eye(2)
        A_amb = np.array([[1,1], [self.Z_amb, -self.Z_amb]])
        T = A_amb @ T
        
        for i in range(self.num_layer):
            A = np.array([[1,1], [self.Z[i], -self.Z[i]]])
            B = np.array([[np.exp(-1j*omega[i]*self.interval[i]), 0],[0, np.exp(1j*omega[i]*self.interval[i])]])
            T = B @ inv(A) @ T
            T = A @ T
        T = inv(A_amb) @ T
        
        S11 = -T[1,0]/T[1,1]
        S12 = 1/T[1,1]
        S21 = T[0,0]-T[0,1]*T[1,0]/T[1,1]
        S22 = T[0,1]/T[1,0]
        
        S = np.array([[S11, S12], 
                      [S21, S22]])
        return S
    
    
    # Wave evolution with TD-TMM 
    def evolution(self, k, direction='f'):
        
        omega_0 = np.abs(k) 
        omega = omega_0 / self.n
        
        if direction=='f':
            D = np.array([[1], [0]])
        elif direction=='b':
            D = np.array([[0], [1]])
        else:
            D = np.array([[0.5], [0.5]])/np.sqrt(2)
        # D = np.array([[1], [0]]) if direction=='f' else np.array([[0], [1]])
        D_list = [D]
        
        A_amb = np.array([[1,1], [self.Z_amb, -self.Z_amb]])
        D = A_amb @ D
        
        for i in range(self.num_layer):
            A = np.array([[1,1], [self.Z[i], -self.Z[i]]])
            B = np.array([[np.exp(-1j*omega[i]*self.interval[i]), 0],[0, np.exp(1j*omega[i]*self.interval[i])]])
            D = B @ inv(A) @ D
            D_list.append(D)
            D = A @ D
            
        D = inv(A_amb) @ D
        trans, reflec = D[0,0], D[1,0]
        T, R = np.abs(trans)**2, np.abs(reflec)**2
        return np.sum(np.column_stack(D_list), axis=0), T, R, trans, reflec
    
    def evolution_Born(self, omega_b):
        alpha = 1/self.eps
        alpha_b = 1/self.eps_amb
        dalpha= alpha - alpha_b
        
        integrand = lambda i: self.interval[:i]*dalpha[:i] * (1 - 
                                                              np.exp(-2j*omega_b * 
                                                                     (self.t[1:i+1]-self.interval[:i]/2 
                                                                      - self.t[i] + self.interval[i-1]/2 )))
        psi_ratio = (omega_b/2j) * np.array( [np.sum(integrand(i)) for i in range(1, self.num_layer+1)] )
        psi_ratio = np.hstack([0, psi_ratio])
        psi_inc = np.exp(-1j * omega_b * self.t)
        psi_sca = psi_ratio * psi_inc
        psi_tot = psi_inc + psi_sca
        
        psi_Tp = psi_tot[-1]
        dpsi_Tp = (psi_tot[-1] - psi_tot[-2]) / self.interval[-1]
        
        reflec = (psi_Tp + dpsi_Tp/(1j*omega_b))/2
        trans = (psi_Tp - dpsi_Tp/(1j*omega_b))/2
        
        return psi_tot, np.abs(trans)**2, np.abs(reflec)**2, trans, reflec
    
    

    def grid(self, dt):
        total_interval = np.sum(self.interval)
        num_new_interval = int(total_interval/dt)
        
        if dt * num_new_interval + dt/2 < total_interval:
            num_new_interval += 1
        
        t_center = np.arange(num_new_interval) * dt + dt/2
        
        temp = 0
        eps_new = np.zeros_like(t_center)
        mu_new = np.zeros_like(t_center)
        interval_new = dt* np.ones_like(t_center)
    
        for i in range(self.num_layer):
            eps_new = eps_new + self.eps[i] * (t_center > temp) * (t_center <= temp + self.interval[i] )
            mu_new = mu_new + self.mu[i] * (t_center > temp) * (t_center <= temp + self.interval[i] )
            temp += self.interval[i]
            
        return time_dep(interval_new, eps_new, mu_new, self.eps_amb, self.mu_amb)
        
