# %% [markdown]
# # **Finite difference method for solving Black-Scholes PDE**
# %%
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.sparse import diags
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

# Market and option parameters
K, r, T = 100, 0.01, 1

# %%
# Grid parameters
s_min, s_max = 10, 300
N, M = 1000, 290  # we want the grid of x to be 5 

# Setup of grids
dtau = T/N
dx = (s_max - s_min)/M
s = np.linspace(s_min, s_max, M+1)
tau = np.linspace(0, T, N+1)

# %%
Volatility_function = lambda x, y: (1 + (T - x) / 30) * (0.1 + 0.4 * np.exp(- y/50)) # x and y are indeed tau and s 

#compute volatility at tau = 0, ie. t = T 
volatility = np.array(Volatility_function(0, s))

def getmatrix(volatility):
    A =  0.5 * volatility**2 * dtau / dx**2 * s**2 - 0.5 * r * dtau/dx * s
    B =  - volatility**2 * dtau / dx**2 * s**2 - r * dtau
    C =  0.5 * volatility**2 * dtau / dx**2 * s**2 + 0.5 * r * dtau/dx * s
    a_diag = np.concatenate([A[1:-1],[0]])
    b_diag = np.concatenate([[0],B[1:-1],[0]])
    c_diag = np.concatenate([[0],C[1:-1]])
    L = diags([a_diag, b_diag, c_diag], [-1, 0, 1]).toarray()
    I = np.identity(M+1)
    return a_diag, b_diag, c_diag, L, I

# Running the explicit scheme
# Initial condition
a_diag, b_diag, c_diag, L, I = getmatrix(volatility)
v_ex = np.maximum(s - K, 0) # call option payoff

# Iteration of explicit scheme
for n in range(1,N+1):
    v_ex = np.matmul(I+L, v_ex)           # Matrix product V^{n+1} = (I+L)V^n
    v_ex[0] = 0                           # Boundary operation at s_min
    v_ex[M] = s_max - K*np.exp(-r*tau[n])   # Boundary operation at s_max
   
    volatility =  np.array(Volatility_function(n/N, s))
    a_diag, b_diag, c_diag, L, I = getmatrix(volatility)


# %%
v_ex[70:111:5] #S_0 = {80, 85, ..., 120}}

# %%
# Tridiagional matrix solver
def TDMAsolver(a, b, c, d):

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

# %%
# Running the implicit scheme
volatility = np.array(Volatility_function(tau[1], s))

a_diag, b_diag, c_diag, L, I = getmatrix(volatility)

# Running the implicit scheme
v_im = np.maximum(s - K, 0)

# Iteration of implicit scheme
for n in range(1,N+1):
    d = v_im   # The RHS of the system of equations is V^{n-1}
    d[0] = 0   # Boundary operator at s_min
    d[M] = s_max - K*np.exp(-r*tau[n])  # Boundary operator at s_min
    v_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   # Run the Thomas algorithm to solve for V^n

    volatility =  np.array(Volatility_function((n+1)/N, s))
    a_diag, b_diag, c_diag, L, I = getmatrix(volatility)
# %%
v_im[70:111:5]
# %%
# Running the Crank-Nicolson scheme
volatility1 = np.array(Volatility_function(0, s))
volatility2 = np.array(Volatility_function(tau[1], s))

a_diag1, b_diag1, c_diag1, L1, I = getmatrix(volatility1)
a_diag2, b_diag2, c_diag2, L2, I = getmatrix(volatility2)

# Initial condition
v_crank = np.maximum(s - K, 0)

for n in range(1,N+1):
    d = np.matmul(I + 0.5*L1, v_crank)     # The RHS of the system of equations is V^{n-1}
    d[0] = 0   # Boundary operator at s_min
    d[M] = s_max - K*np.exp(-r*tau[n])  # Boundary operator at s_min
    v_crank = TDMAsolver(-0.5*a_diag2, 1-0.5*b_diag2, -0.5*c_diag2, d)   # Run the Thomas algorithm to solve for V^n

    volatility1 =  np.array(Volatility_function(n/N, s))
    volatility2 = np.array(Volatility_function((n+1)/N, s))

    a_diag1, b_diag1, c_diag1, L1, I = getmatrix(volatility1)
    a_diag2, b_diag2, c_diag2, L2, I = getmatrix(volatility2)
# %%
v_crank[70:111:5]
# %%
callprice = pd.DataFrame(
    {'Initial': [i for i in range(80, 125, 5)],
     'Explicit': v_ex[70:111:5],
     'Implicit': v_im[70:111:5],
     'Crank-Nicolson': v_crank[70:111:5],
    })
callprice.to_csv('call.csv', index = False, float_format="%.5f")  

# %%
callprice

# %%
plt.figure(figsize = (15,10))
plt.plot(callprice.Initial, callprice.Explicit, label = 'Explicit Scheme')
plt.plot(callprice.Initial, callprice.Implicit, label = 'Implicit Scheme')
plt.plot(callprice.Initial, callprice.iloc[:, 3], label = 'Explicit Scheme')
plt.xlabel(r'$S_0$', fontsize=15)
plt.ylabel("European Call Option Price", fontsize=15)
plt.legend(loc='upper left', fontsize = 15)
plt.savefig('european_call.eps',format = 'eps')
# %%
import numpy as np
from scipy.stats import norm
from scipy import optimize

def impliedvol(S,K,T,r,marketoptionPrice):
  
    def bs_call(vol):
        d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1-(vol * np.sqrt(T))
        BSprice_call =  S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2) 
        fx = BSprice_call - marketoptionPrice
        return fx

    return optimize.brentq(bs_call,0.0001,100,maxiter=1000) #If convergence is not achieved in maxiter iterations, an error is raised.
# %%
# Use (1) to compute European call option price 
# Say we use explicit scheme
K_list = [i for i in range(70, 131)]
Call_price = []

for K in K_list:
    volatility = np.array(Volatility_function(0, s))
    a_diag, b_diag, c_diag, L, I = getmatrix(volatility)
    v_ex = np.maximum(s - K, 0) # call option payoff

    for n in range(1,N+1):
        v_ex = np.matmul(I+L, v_ex)           # Matrix product V^{n+1} = (I+L)V^n
        v_ex[0] = 0       
        v_ex[M] = s_max - K*np.exp(-r*tau[n])                        
   
        volatility =  np.array(Volatility_function(n/N, s))
        a_diag, b_diag, c_diag, L, I = getmatrix(volatility)

    Call_price.append(v_ex[90])
# %%
impliedsigma = []

for i in range(61):
    j = impliedvol(S = 100, K = K_list[i], T = 1, r = 0.01, marketoptionPrice = Call_price[i])
    impliedsigma.append(j)

# %%
local = pd.DataFrame(
    {'K': K_list,
     'Call Option Price':Call_price,
     'Implied Volatility': impliedsigma,
    })
local.to_csv('local.csv', index = False, float_format="%.4f")  

# %%
local

# %%
plt.figure(figsize = (15,10))
plt.plot(K_list, impliedsigma)
plt.xlabel('Strike K', fontsize=15)
plt.ylabel("Implied Volatility", fontsize=15)
plt.savefig('implied_volatility1.eps',format = 'eps')

# %%
# Now we fix constant local volatility
# Using implicit scheme
sigma = 0.157
a_diag, b_diag, c_diag, L, I = getmatrix(sigma)

newCall_price = []

for K in K_list:
    v_im = np.maximum(s - K, 0) # call option payoff
    
    for n in range(1,N+1):
        d = v_im           
        d[0] = 0       # Boundary operation at s_min
        d[M] = s_max - K*np.exp(-r*tau[n])          
        v_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)     

    newCall_price.append(v_im[90])
# %%
new_impliedsigma = []
for i in range(61):
    j = impliedvol(S = 100, K = K_list[i], T = 1, r = 0.01, marketoptionPrice = newCall_price[i])
    new_impliedsigma.append(j)
# %%
new_local = pd.DataFrame(
    {'K': K_list,
     'Call Option Price':newCall_price,
     'Implied Volatility': new_impliedsigma,
    })
new_local.to_csv('new_local.csv', index = False, float_format="%.4f")  

# %%
plt.figure(figsize = (15,10))
plt.plot(K_list, new_impliedsigma)
plt.xlabel('Strike K', fontsize=15)
plt.ylabel("Implied Volatility", fontsize=15)
plt.savefig('implied_volatility2.eps',format = 'eps')

# %%
#Local Volatility Model 
# Put Option Price, implicit scheme 
B_list = [60, 70, 80, 90]
K, r, T = 100, 0.01, 1

volatility = np.array(Volatility_function(tau[1], s))
a_diag, b_diag, c_diag, L, I = getmatrix(volatility)
# Running the implicit scheme
v_im = np.maximum(K - s, 0)

# Iteration of implicit scheme
for n in range(1,N+1):
    d = v_im   
    d[0] = K*np.exp(-r*tau[n]) 
    d[M] = 0
    v_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   

    volatility =  np.array(Volatility_function((n+1)/N, s))
    a_diag, b_diag, c_diag, L, I = getmatrix(volatility)

Put_price = v_im[100 - s_min]
# %%
Put_price

# %%
# Down and out barrier put 
B_list = [60, 70, 80, 90]
outbarrier = []

for B in B_list:
    s_min = B 
    s_max = 300
    N, M = 1000, s_max - s_min  # we want the grid of x to be 5 

    # Setup of grids
    dtau = T/N
    dx = 1
    s = np.linspace(s_min, s_max, M+1)
    tau = np.linspace(0, T, N+1)

    volatility = np.array(Volatility_function(tau[1], s))
    a_diag, b_diag, c_diag, L, I = getmatrix(volatility)

    indicator = lambda x: 1 if x > B else 0
    indicator = np.vectorize(indicator)
    v_im = np.maximum(K - s, 0)* indicator(s)

    for n in range(1, N+1):
        d = v_im     
        d[0] = 0
        d[M] = 0
        v_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d) 

        volatility =  np.array(Volatility_function((n+1)/N, s))
        a_diag, b_diag, c_diag, L, I = getmatrix(volatility)
    
    outbarrier.append(v_im[100-B])

# %%
localbarrier = pd.DataFrame(
    {
        'B': B_list,
        'Put Option Pirce': [Put_price]*4,
        'Down and out Barrier': outbarrier,
        'Down and in Barrier': [Put_price - x for x in outbarrier],
    }
)
localbarrier.to_csv('localbarrier.csv', index = False, float_format="%.5f")  
localbarrier

# %%
# Constant volatility case
B_list = [60, 70, 80, 90]
K, r, T = 100, 0.01, 1
s_min, s_max = 10, 300
N, M = 1000, 290  # we want the grid of x to be 5 

# Setup of grids
dtau = T/N
dx = (s_max - s_min)/M
s = np.linspace(s_min, s_max, M+1)
tau = np.linspace(0, T, N+1)

sigma = 0.157
a_diag, b_diag, c_diag, L, I = getmatrix(sigma)

v_im = np.maximum(K - s, 0)

for n in range(1,N+1):
    d = v_im   
    d[0] = K*np.exp(-r*tau[n]) 
    d[M] = 0
    v_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d)   

Put_const = v_im[100 -10]
# %%
Put_const

# %%
# Down and out barrier put 
outbarrier_new = []
for B in B_list:
    s_min = B 
    s_max = 300
    N, M = 1000, s_max - s_min  # we want the grid of x to be 5 

    # Setup of grids
    dtau = T/N
    dx = 1
    s = np.linspace(s_min, s_max, M+1)
    tau = np.linspace(0, T, N+1)

    volatility = 0.157
    a_diag, b_diag, c_diag, L, I = getmatrix(volatility)

    indicator = lambda x: 1 if x > B else 0
    indicator = np.vectorize(indicator)
    v_im = np.maximum(K - s, 0)* indicator(s)

    for n in range(1, N+1):
        d = v_im     
        d[0] = 0
        d[M] = 0
        v_im = TDMAsolver(-a_diag, 1-b_diag, -c_diag, d) 
    
    outbarrier_new.append(v_im[100-B])
# %%
constbarrier = pd.DataFrame(
    {
        'B': B_list,
        'Put Option Pirce': [Put_const]*4,
        'Down and out Barrier': outbarrier_new,
        'Down and in Barrier': [Put_const - x for x in outbarrier_new],
    }
)
constbarrier.to_csv('constbarrier.csv', index = False, float_format="%.5f")  
# %%
plt.figure(figsize = (15,10))
plt.plot(B_list, localbarrier.iloc[:, 3], label = 'local volatility function')
plt.plot(B_list, constbarrier.iloc[:, 3], label = 'constant volatility')
plt.xlabel('Barrier level', fontsize=15)
plt.ylabel("Down-and-in Barrier", fontsize=15)
plt.legend(loc='upper left', fontsize = 15)
plt.savefig('Barrier.eps',format = 'eps')
# %%
local_price = localbarrier.iloc[:,3]
const_price = constbarrier.iloc[:,3]
diff = [100* (local_price[i] - const_price[i]) / local_price[i] for i in range(0,4)]
difftable = pd.DataFrame(
    {
        'B': B_list,
        'Local volatility function': local_price,
        'Constant volatility': const_price,
        'Percentage difference': diff,
    }
)
difftable.to_csv('diff.csv', index = False, float_format="%.5f")  


