# %%
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# Implementation of the above pricing algorithm:

# %%
def TrinomailTree(s0, u, d, qu, qm, qd, R, N, payoff, american=False):
    # Kamrad-Ritchken parametrisation #
        
    # Create empty arrays to store the stock price and option value at each node
    V = np.zeros(((N+1)*2 - 1, N+1))
    V[:] = np.nan
    S = np.zeros(((N+1)*2 - 1, N+1))
    S[:] = np.nan
    
    # Set up S[j,k,n] = s0 * u^(n-j-k) * d^k 
    for t in range(N+1):
      a = [0] * t
      b = [i for i in range(t, -1, -1)]
      c = [i for i in range(0,t+1)]
      S[0:(t+1)*2 -1, t] = s0 * u**np.array(b+a) * d**np.array(a+c)
    
    # Compute the option price at terminal time
    V[:,N] = payoff(S[:,N])
    
    # Loop backward in time
    for t in reversed(range(N)):
        if american == True:
          V[0:(t+1)*2 -1, t] = np.maximum(payoff(S[0:(t+1)*2 -1, t]), (qu * V[0:(t+1)*2 -1,t+1] + qm * V[1:(t+1)*2, t+1] + qd * V[2:(t+1)*2 +1, t+1]) / R)  # The R is going to be taken as R=exp(r dt), which represents the interest rate factor
        else:
          V[0:(t+1)*2 -1, t] = (qu * V[0:(t+1)*2 -1,t+1] + qm * V[1:(t+1)*2, t+1] + qd * V[2:(t+1)*2 +1, t+1]) / R   
        
    return V, S     # get the function to return the whole tree of stock prices and option values

# %%
def GetTrinomialCRRPara(r, lam,  dt, sigma):
  u = np.exp(lam * sigma * np.sqrt(dt))
  d= 1/u
  R = np.exp(r * dt)
  qu = 1/(2* lam**2) + (r - sigma**2 /2) * np.sqrt(dt) / (2 * lam * sigma)
  qm = 1 - 1/(lam ** 2)
  qd = 1/(2* lam**2) - (r - sigma**2 /2) * np.sqrt(dt) / (2 * lam * sigma)
  return u, d, qu, qm, qd, R

# %%
list = [1, 1.25, 1.5, 1.75]
for lam in list:
    print(GetTrinomialCRRPara(0.01, lam, 0.5, 0.2))

# %%
GetTrinomialCRRPara(r = 0.01, lam = 1.25, dt = 1/3, sigma=0.2)

# %% [markdown]
# Now we try to use a 2-period tree to price a European call option which payoff function is $(S_T - K)^{+}$. Parameters are: initial stock price $S_0=100$, strike price $K=100$, maturity $T=1$ year, interest rate $r=1\%$ and volatility $\sigma=20\%$.

# %% [markdown]
# ### Black Scholes for European Call Option: 
# %%
from scipy.stats import norm

def d1(S,K,T,r,sigma):
    return(np.log(S/K) + (r + sigma**2/2.)*T)/(sigma * np.sqrt(T))

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma) - sigma * np.sqrt(T)


def bs_call(S,K,T,r,sigma):
    return S * norm.cdf(d1(S,K,T,r,sigma)) - K * np.exp(-r*T) * norm.cdf(d2(S,K,T,r,sigma))

# %%
def price_plot(s0, r, T, sigma, K):
    CallPayoff = lambda x : np.maximum(x - K, 0)    

    Lam = [1, 1.25, 1.5, 1.75]
    prices = np.zeros((100, 4))

    for lam in Lam:
        for N in range(2, 102):
            u, d, qu, qm, qd, R = GetTrinomialCRRPara(r, lam, T/N, sigma)
            V, S = TrinomailTree(s0, u, d, qu, qm, qd, R, N, CallPayoff)
            prices[N-2][Lam.index(lam)] = V[0,0]
    
    df = pd.DataFrame(prices, columns=['lambda = 1', 'lambda = 1.25', 'lambda = 1.5', 'lambda = 1.75'])
    bs = bs_call(s0, K, T, r, sigma)

    plt.figure(figsize = (20,15))
    for i in df.columns:
        plt.plot(df[i], label = str(i))
    plt.axhline(y=bs, color='k', linestyle='-', label = 'BS = ' + str(round(bs,3)))
    plt.legend(loc="upper right")
    plt.xlabel("N", fontsize=15)
    plt.ylabel("Price at initial time", fontsize=15)
    

# %%
r, T, sigma, K = 0.01, 1, 0.2, 100

price_plot(60, r, T, sigma, K)
plt.savefig('price60.eps',format = 'eps')

price_plot(80, r, T, sigma, K)
plt.savefig('price80.eps',format = 'eps')

price_plot(100, r, T, sigma, K)
plt.savefig('price100.eps',format = 'eps')

price_plot(120, r, T, sigma, K)
plt.savefig('price120.eps',format = 'eps')
    

# %% [markdown]
# ### Eurpean Option (c)
# Different $S_0$ and $\lambda$

# %%
r, T, sigma, K = 0.01, 1, 0.2, 100     
N = 500
CallPayoff = lambda x : np.maximum(x - K, 0)    

# Exact Black Scholes
BS = []
for i in range(70, 131, 5):  # take 12 values in [70, 130]
    BS.append(bs_call(i, 100, 1, 0.01, 0.2))

# Using Trinomial Tree
table = []
Lam = [1, 1.25, 1.5, 1.75]

for j in range(70, 131, 5):
    for lam in Lam:
        u, d, qu, qm, qd, R = GetTrinomialCRRPara(r, lam, T/N, sigma)
        V, S = TrinomailTree(j, u, d, qu, qm, qd, R, N, CallPayoff)
        table.append(V[0,0])


# %%
table = np.array(table).reshape(13,4)
table = pd.DataFrame(table, columns=['lambda = 1', 'lambda = 1.25', 'lambda = 1.5', 'lambda = 1.75'])
table['BS'] = BS
table = table.round(4)
table['Initial'] = [x for x in range(70, 131, 5)]
table

# %%
table.to_csv('table.csv', index = False, header=False)

# %%
# compute absolute difference 
for column in table.columns[:4]:
    print(sum(abs(table['BS'] - table[column])))
# so we should probably choose lambda = 1.5

# %% [markdown]
# ### American Option
# Now let's try to price an American version of the put instead.

# %%
# 2.(b)
r, T, sigma, K = 0.01, 1, 0.2, 100
lam = 1.5     
N = 500                                         
CallPayoff = lambda x : np.maximum(x - K, 0)   

American = []
u, d, qu, qm, qd, R = GetTrinomialCRRPara(r, lam, T/N, sigma)
Intrinsic = CallPayoff(np.arange(70, 130))
for i in range(70, 130):
    V, S = TrinomailTree(i, u, d, qu, qm, qd, R, N, CallPayoff, True)
    American.append(V[0,0])


# %%
plt.figure(figsize = (15,10))
plt.plot(np.arange(70, 130), American, label = r'American Call Option (r = 0.01, $\sigma$ = 0.2)')
plt.plot(np.arange(70, 130), Intrinsic, color = 'black',label = 'Intrinsic Value')
plt.xlabel(r'$S_0$', fontsize=15)
plt.ylabel("Price at time zero", fontsize=15)
plt.legend(loc='upper left', fontsize = 15)
plt.savefig('american_call.eps',format = 'eps')

# %%
# 2.(c)

PutPayoff = lambda x: np.maximum(K - x, 0)
T, K, lam = 1, 100, 1.5
N = 500   

Intrinsic_put = PutPayoff(np.arange(70, 130))
American_put = []
par = [(0.01, 0.2), (0.05, 0.2), (0.01, 0.45),(0.05, 0.45)]

for (i, j) in par:
    u, d, qu, qm, qd, R = GetTrinomialCRRPara(i, lam, T/N, j)
    for i in range(70, 130):
        V, S = TrinomailTree(i, u, d, qu, qm, qd, R, N, PutPayoff, True)
        American_put.append(V[0,0])


# %%
list(set(American_put[60:120]).intersection(Intrinsic_put))
# min = 19, i.e. S0 = 100 - 19 = 81

# %%
plt.figure(figsize = (15,10))
plt.plot(np.arange(70, 130), American_put[:60], label = r'American Put Option (r = 0.01, $\sigma$ = 0.2)')
plt.plot(np.arange(70, 130), American_put[60:120], label = r'American Put Option (r = 0.05, $\sigma$ = 0.2)')
plt.plot(np.arange(70, 130), American_put[120:180], label = r'American Put Option (r = 0.01, $\sigma$ = 0.45)')
plt.plot(np.arange(70, 130), American_put[180:240], label = r'American Put Option (r = 0.05, $\sigma$ = 0.45)')
plt.plot(np.arange(70, 130), Intrinsic_put, color = 'black', label = 'Intrinsic Value')
plt.plot((81,81), (0, 23), color = 'orange', linestyle='--')
plt.xlabel(r'$S_0$', fontsize=15)
plt.ylabel("Price at time zero", fontsize=15)
plt.legend(loc='upper right', fontsize = 15)
plt.savefig('american_put.eps',format = 'eps')


# %% [markdown]
# ### Floatint strike lookback put option
# Forward shooting grid method

# %%
V, S = TrinomailTree(100, 2, 1/2, 0.4, 0.3, 0.4, 1, 2, PutPayoff, american=False)

# %%
def gridM(s0, u, N):

    M = np.zeros((N+1, N+1))

    for t in range(N+1):
        M[0:t+1, t] = s0* u**np.arange(t, -1, -1)
    return M

# %%
def floatingstrike(N, qu, qm, qd, R, s0, u, d, american =False):
    V = dict()

    #compute trinomial tree price S
    S = np.zeros(((N+1)*2 - 1, N+1))
    S[:] = np.nan
    for t in range(N+1):
        a = [0] * t
        b = [i for i in range(t, -1, -1)]
        c = [i for i in range(0,t+1)]
        S[0:(t+1)*2 -1, t] = s0 * u**np.array(b+a) * d**np.array(a+c)
    ######################
    M = gridM(s0, u, N) 
    M_value = M[0:N+1, N]
    S_value = S[0: 2*N + 1, N]

    a = [0] * N
    b = [i for i in range(t, -1, -1)]
    c = [i for i in range(0,t+1)]
    k_list = a+c
    j_list = [N-(b+a)[i] - k_list[i] for i in range(2*N+1)]
    list_jk = [(j_list[i], k_list[i]) for i in range(2*N + 1)]


    #terminal case
    V[N] = np.zeros((N+1, 2*N + 1))
    for l in range(N+1):
        for (j,k) in list_jk:
            if l <= j+2*k:
                V[N][l, (j+2*k)] = M_value[l] - S_value[j+2*k]  
    #####################

    Shooting = lambda j,k,l: min(j + 2*k, l+1) #shooting function 

    for t in reversed(range(N)):
        V[t] = np.zeros((t+1, 2*t + 1))
        a = [0] * t
        b = [x for x in range(t, -1, -1)]
        c = [x for x in range(0, t+1)]

        M = gridM(s0, u, t) 
        M_value = M[0:t+1, t]
        S_value = S[0: 2*t + 1, t]

        k_list = a+c
        j_list = [t-(b+a)[x] - k_list[x] for x in range(2*t+1)]
        list_jk = [(j_list[x], k_list[x]) for x in range(2*t + 1)] #construct (j,k) coordinate

        for l in range(t+1):
            for (j,k) in list_jk:
                if l <= j+2*k:
                    if american == True:
                        V[t][l, (j+2*k)] = max(M_value[l] - S_value[j+2*k], (qu * V[t+1][Shooting(j,k,l), (j+2*k)] + qm * V[t+1][Shooting(j+1,k,l), j+1+2*k] + qd * V[t+1][Shooting(j,k+1,l),j+2*(k+1)]) / R)
                    else:
                        V[t][l, (j+2*k)] = (qu * V[t+1][Shooting(j,k,l), (j+2*k)] + qm * V[t+1][Shooting(j+1,k,l), j+1+2*k] + qd * V[t+1][Shooting(j,k+1,l),j+2*(k+1)]) / R

    return V[0][0,0]

# %%
# example
lam = 1.25
sigma = 0.2
T = 1
N = 2
u, d, qu, qm, qd, R = GetTrinomialCRRPara(r, lam, T/N, sigma)
V_Euro = floatingstrike(2, qu, qm, qd, R, 100, u, d, False)
V_Ameri = floatingstrike(2, qu, qm, qd, R, 100, u, d, True)
V_Euro, V_Ameri

# %%
# 3.(b)
import timeit
Euro = []
Ameri = []
time_euro = []
time_ameri=[]

lam = 1.5
sigma = 0.2
T = 1
s0 = 100 

List = [2, 10, 50, 100, 500]

for N in List:
    u, d, qu, qm, qd, R = GetTrinomialCRRPara(r, lam, T/N, sigma)
    start = timeit.default_timer()
    Euro.append(floatingstrike(N, qu, qm, qd, R, s0, u, d, False))
    stop = timeit.default_timer()
    time_euro.append(stop-start)

    start = timeit.default_timer()
    Ameri.append(floatingstrike(N, qu, qm, qd, R, s0, u, d, True))
    stop = timeit.default_timer()
    time_ameri.append(stop-start)

# %%
Euro, Ameri

# %%
time_euro, time_ameri

# %%
floatingstrike = pd.DataFrame(list(zip(List, Euro, time_euro, Ameri, time_ameri)), columns=['N','European','Time for European', 'American', 'Time for American'])
floatingstrike

# %%
floatingstrike.to_csv('floatingstrike.csv', index = False, header=False)


