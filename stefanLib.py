import numpy as np
import pandas as pd 
from numpy import linalg as ln 
import numpy.random as rdm
import matplotlib.pyplot as plt; plt.style.use('dark_background')
from   ttictoc import tic,toc
from   scipy.stats import norm
import tensorflow as tf
import tensorflow.math as tfm
from   tensorflow import keras
from keras import losses 
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant, RandomUniform
import matplotlib.colors as colors

from matplotlib.patches import Patch
from matplotlib import colors
cmap = colors.LinearSegmentedColormap.from_list('my_cmap', (
       (0., (0., 0., 0.)),
       (0.5, (1., 1., 1.)),
       (1., (.0, .25, .6))))

tf.config.list_physical_devices()
tf.config.set_visible_devices([], 'GPU')

# Volume of unit ball
vBall = {1:2, 2: np.pi, 3: 4/3*np.pi}
_n,_t = np.newaxis,tf.newaxis
#============================================================#

def trainParams(probParams):
    """Training parameters used throughout the experiments"""
    d,T = probParams["d"],probParams["T"]
    trnParams = {}
    trnParams["B"]  = 2**(7 + d)                 # Batch size
    trnParams["M"]  = 3000                       # Number of training iterations
    trnParams["K"]  = int(50*d)                  # Number of test functions
    trnParams["dt"] = 0.01                       # Step size
    trnParams["N"]  = int(T/trnParams["dt"])     # Number of time steps 
    trnParams["ts"] = np.linspace(0,T,trnParams["N"]+1)[:,_n] # Time grid
    # Mushy region width
    eps = {i: np.sqrt(probParams["alpha"][i]*d*trnParams["dt"]) for i in [1,2]} 
    eps[0] = probParams["R"]/10 # Mushy region width for integrals
    trnParams["eps"]     = eps
    trnParams["delta"]   = {i: 2*eps[i] for i in [1,2]}
    trnParams["bdrySim"] = 1000
    trnParams["lbda0"]   = 0.1 # Lagrange multiplier
    return trnParams 

#============================================================#

def BM(x0,d,nSim,T,N):
    """Brownian """
    dW = np.sqrt(T/N) * tf.random.normal(shape=(nSim,N,d))
    X  = tfm.cumsum(dW, axis=1)
    X  = x0 + tf.concat((np.zeros((nSim,1,d)), X),axis=1)
    return X

mod = lambda x,R: (x - np.floor((x+R)/(2*R))*2*R).numpy()

def simBall(nSim,r,R,d):
    """Simulate points in {x : r < |x| < R} uniformly."""
    if   d == 1:
        return r + (R - r)*rdm.rand(nSim,1,1)
    elif d == 2: 
        phi = 2*np.pi*rdm.rand(nSim,1,1)
        r2 = r**2 + (R**2 - r**2)*rdm.rand(nSim,1,1)
        return np.sqrt(r2)*np.concatenate([np.cos(phi),np.sin(phi)],axis=2)
    else: 
        phi1,phi2 = 2*np.pi*rdm.rand(nSim,1,1), np.pi*rdm.rand(nSim,1,1)
        r3        = r**3 + (R**3 - r**3)*rdm.rand(nSim,1,1)
        return (r3)**(1/3)*np.concatenate([np.cos(phi1)*np.sin(phi2),
                                           np.sin(phi1)*np.sin(phi2),
                                           np.cos(phi2)],axis=2)
    
def simX0(nSim,R,d,Phi0,vGamma0 = None):
    """Simulate points in Omega with density v0. """
    if vGamma0 is not None: 
        ratio = {1: 1 - vGamma0/(vBall[d]*R**d), 2: vGamma0/(vBall[d]*R**d)}
        X0 = {}
        for i in [1,2]: 
            x0    = simBall(min(int(nSim/ratio[i]),100*nSim),0.,R,d)
            X0[i] = x0[np.where(Phi0(x0)*(-1)**i < 0.)][:,_n]
    else: 
        x0 = simBall(nSim,0.,R,d)
        X0 = {i: x0[np.where(Phi0(x0)*(-1)**i < 0.)][:,_n] for i in [1,2]}
    return X0

def RBM(x0,T,N,R,alpha = 1.,antithetic = True):
    """Reflected Brownian Motion in a Ball, with/without antithetic sampling."""
    nSim,_,d = np.shape(x0)
    # Brownian Increments
    dW = np.sqrt(T/N) * tf.random.normal(shape=(nSim,N,d))
    # Antithetic sampling
    if antithetic: x0,dW = np.tile(x0,(2,1,1)), np.concatenate([dW,-dW],axis = 0)
    # Reflected Brownian motions
    X  = x0 + np.concatenate((np.zeros_like(x0), np.zeros_like(dW)),axis=1)
    for n in range(N):
        X[:,n+1] = X[:,n] + np.sqrt(alpha) * dW[:,n]
        ids      = np.where(ln.norm(X[:,n+1],axis=1) >  R)[0]
        for j in ids: X[j,n+1] = X[j,n+1] * max(2*R/ln.norm(X[j,n+1]) - 1., 0.)    
    return X

#========================================#
#================== NN ==================#
#========================================#

def leakyReLU(x): return tf.nn.leaky_relu(x,alpha = 1e-2)

def normalize(X,mu = None,sig = None,ax=0): 
    """Input normalization"""
    try: 
        return  (X - mu)/(sig + 1e-12)
    except:
        mu,sig = tf.reduce_mean(X,axis=ax), tf.math.reduce_std(X,axis=ax)
        return  (X - mu)/(sig + 1e-12)
    
def newNN(dX): 
    """Neural Network Architecture"""
    # Input dimension
    d_ = 1 + max(dX,0) 
    # Number of nodes in hidden layers 
    qH = 20 + d_
    return   Sequential([ 
             Dense(qH,activation = leakyReLU,input_dim = d_), 
             Dense(qH,activation = leakyReLU),
             Dense(qH,activation = leakyReLU),
             Dense(1, activation = None)])

def nnOut(NN,ts,X,R,T):
    X_   = X/R
    init = (ts[0] == 0.) & (len(ts) > 1)
    if init: X_ = X[:,1:]; ts = ts[1:] 
    nSim,N,d = np.shape(X_)   
    Y = np.tile(ts - T/2,[nSim,1,1])
    Y = tf.concat([Y,X_],axis=2)
    Y = tf.reshape(Y,(nSim * N,d+1))
    g = tf.reshape(NN(Y),(nSim,N))
    if init: g = tf.concat([tf.zeros_like(g[:,:1]),g],axis=1)
    if (ts == 0.).all(): g = 0. * g
    return g

opt = keras.optimizers.Adam(learning_rate = 1e-3)

def h(dist):  
    """Stopping factors for each time point."""
    return tf.clip_by_value((1-dist)/2,0.,1.)

#====================================================#
#================== Test functions ==================#
#====================================================#

Psi  = lambda k  : lambda x: np.exp(- a * ln.norm(x-z[k,:],axis=2)**2)
PsiZ = lambda z,a: lambda x: np.exp(- a * ln.norm(x-z,axis=2)**2)

#=========================================#
#================== Viz ==================#
#=========================================#

def showLosses(losses,burnIn=10,what = "Loss",save = False,figName = None):
    """Plot loss vs training iteration."""
    losses = losses[burnIn:] # Remove burn-in period (optional)
    plt.figure(figsize=(6,3))
    plt.plot(burnIn + np.arange(1,len(losses)+ 1),losses,color ="steelblue")
    plt.xlabel("Training Step"); plt.title("%s vs Training Step"%what)
    if save: plt.savefig(figName,dpi = 600,bbox_inches = "tight")
    plt.show()

nGrid = 501
m2V   = lambda s: np.reshape(s,(-1,1))  # Matrix to (column) Vector
v2M   = lambda s: np.reshape(s,(nGrid,nGrid)) # Vector to Matrix
    
def viz(NN,probParams,trnParams,sharp = False,save = False,figName = None):  
    """Heatmap of ice and water regions over time"""
    d,T,R,r,L,eta,gam,alpha,c,phase,tension,Phi0,vOmega,vGamma0 = probParams.values()
    B,M,K,dt,N,ts,eps,delta,bdrySim,lbda0,C                     = trnParams.values()
    section = {"": [0,1]} if d == 2 else {"x,y": [0,1,2], "y,z": [2,0,1], "x,z": [0,2,1]}
    nT     = int(N/4); ts_ = ts[::nT]; N_ = len(ts_) - 1
    # Grids
    nGrid  = 501
    grid1D = np.linspace(-R,R,nGrid)    # one-dimensional grid
    x1,x2  = np.meshgrid(grid1D,grid1D) # mesh (Square)
    X      = np.hstack([m2V(x1),m2V(x2)]).astype("float32")
    if d == 3: X = np.hstack([X,np.zeros_like(X[:,[0]])])
    X      = np.reshape(np.tile(X,(1,1,N_ + 1)),(nGrid**2,N_ + 1,d))
    
    for key in section.keys():
        x_ = X[...,section[key]]
        u  = tf.where(Phi0(x_) + nnOut(NN,ts_,x_,R,T) >= 0, 1.,0.)
        u  = tf.where(ln.norm(x_,axis = 2) > R, -1.,u)
        u  = {ts_[n,0]: v2M(u[:,n]) for n in range(N_+1)}
        # Plot
        fig,ax = plt.subplots(1,N_+1,figsize=(17,5))
        for n in range(N_+1):
            contPlt = ax[n].contourf(x1, x2, u[ts_[n,0]],cmap=cmap,alpha=0.9) 
            ax[n].axis('scaled')
            ax[n].locator_params(axis='both', nbins=4)
            ax[n].set_title("t = %2.3f"%(n*T/N_))#ts_[n,0])
            ax[n].set_xlabel(r"$x_1$",fontsize=13)
            if n==0: ax[n].set_ylabel(r"$x_2$",rotation = 0,labelpad=8,fontsize=13)
            ax[n].plot(X[0,:nT*n,0],X[0,:nT*n,1],color="k")
            ax[n].scatter(X[0,0,0],X[0,0,1],s=10,color="darkred") 
            ax[n].grid(alpha = 0.2); 
            ax[n].set_xticks(np.linspace(-R,R,5))
            ax[n].set_yticks(np.linspace(-R,R,5))
        if save: plt.savefig("%s, %s.pdf"%(figName,key),dpi = 600,bbox_inches = "tight")
        plt.suptitle(key,y = 0.9,fontsize = 15)
        plt.show()
        
def showRadial(m,NN,probParams,trnParams,nR = 50,nA = 100):   
    """Plot Neural network values in random directions"""
    d,T,R,r,L,eta,gam,alpha,c,phase,tension,Phi0,vOmega,vGamma0 = probParams.values()
    B,M,K,dt,N,ts,eps,delta,bdrySim,lbda0,C = trnParams.values()
    # Radii and angles
    rGrid,angles = np.linspace(R/100,R,nR)[:,_n,_n], simBall(nA,1.,1.,d) 
    G_,Phi_,Q_,dist_ = np.zeros((nA,nR,N+1)),np.zeros((nA,nR,N+1)),np.zeros((nA,nR,N+1)),np.zeros((nA,nR,N+1))
    for i in range(nA):
        X       = rGrid @ angles[i]
        X       = tf.Variable(tf.reshape(np.tile(X,(1,1,N+1)),(nR,N+1,d)))
        with tf.GradientTape(persistent=True) as tape:
            G   = nnOut(NN,ts,X,R,T)
            Phi = Phi0(X) + G 
        nab = tape.gradient(Phi, X)
        G_[i],Phi_[i]  = G.numpy(), Phi.numpy() 
        dist_[i] = Phi_[i] / np.maximum(1e-10,ln.norm(nab,axis = 2))
        Q_[i]    = h(dist_[i] / eps[0])
    G, Phi, Q, dist = np.mean(G_,axis = 0),np.mean(Phi_,axis = 0),np.mean(Q_,axis = 0), np.mean(dist_,axis=0)
    stdG,stdQ = np.std(G_,axis = 0), np.std(Q_,axis = 0)#/np.sqrt(nA)/np.sqrt(nA)
    # PLOT
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    for n in np.arange(0,N,2): #[0,int(N/4),int(N/2),int(3*N/4),N]:
        plt.plot(rGrid.flatten(),G[:,n],label = r"$t=%2.2f$"%ts[n])
        ax.fill_between(rGrid.flatten(),G[:,n] - stdG[:,n],G[:,n] + stdG[:,n],color = "w",alpha = 0.1)
    plt.grid(alpha = .2); plt.title(r"$G$"); plt.xlabel(r"$r$"); plt.show()
    if m >= M-1:
        fig,ax = plt.subplots(1,1,figsize=(5,3))
        for n in np.arange(0,int(N),2): #[0,int(N/4),int(N/2),int(3*N/4),N]:
            plt.plot(rGrid.flatten(),Phi[:,n],label = r"$t=%2.2f$"%ts[n])
        plt.hlines(0,0,R,lw = 1/2)
        plt.grid(alpha = .2); plt.title(r"Level set $(\Phi)$") ; plt.show()
        # Signed distance
        fig,ax = plt.subplots(1,1,figsize=(5,3))
        for n in np.arange(0,int(N),2): 
            plt.plot(rGrid.flatten(),dist[:,n],label = r"$t=%2.2f$"%ts[n])
        plt.hlines(0,0,R,lw = 1/2)
        plt.grid(alpha = .2); plt.title(r"Signed distance") ; plt.show()
        
        
#===================================================#
#============= Curvature Computation ===============#
#===================================================#    
    
def length(y1,y2): 
    """length of segment"""
    return ln.norm(y2 - y1, axis = 2,keepdims = 1)


def areaTri(y1,y2,y3): 
    """Area of Triangle"""
    return ln.norm(np.cross(y2 - y1,y3 - y1),axis = y1.ndim-1,keepdims = 1)/2

def area(y,y0):
    """Area of Diamond"""
    A = [areaTri(y0,y[i],y[(i+1) % 4]) for i in range(4)]
    return np.sum(A,axis =0)

def simBdry0(bdrySim,NN,probParams,trnParams):
    """Simulation of boundary particles"""
    d,T,R,r,L,eta,gam,alpha,c,phase,tension,Phi0,vOmega,vGamma0 = probParams.values()
    B,M,K,dt,N,ts,eps,delta,bdrySim,lbda0,C = trnParams.values()
    Y0  = simBall(bdrySim,0.,R,d)
    Y0_ = tf.Variable(np.reshape(np.tile(Y0,(1,1,N+1)),(bdrySim,N+1,d)))
    with tf.GradientTape() as tape:
        Phi = Phi0(Y0_) + nnOut(NN,ts,Y0_,R,T)
    nab  = tape.gradient(Phi, Y0_)[:,1:]
    Phi  = Phi[:,1:]/(1e-12+ln.norm(nab,axis = 2))
    id_  = {}
    id_[1]  = np.where((Phi <= delta[1])  & (Phi > 0.))  # water particles
    id_[2]  = np.where((Phi >= -delta[2]) & (Phi <= 0.)) # ice particles
    return {i: {n: Y0[id_[i][0][np.where(id_[i][1] == n)]] for n in range(N)} for i in [1,2]}

def curvature2D(y0,t,delta,Phi0,NN,R,T):
    """Curvature approximation by dilation (d = 2)"""
    l = delta/10; iota = l
    # Normal vector
    y0_ = tf.Variable(y0)
    with tf.GradientTape() as tape:
        Phi = Phi0(y0_) + nnOut(NN,t,y0_,R,T)
    nab = tape.gradient(Phi, y0_)
    nu0 = nab / np.maximum(1e-10,ln.norm(nab,axis = 2,keepdims = True))  
    mu  = tf.concat([-nu0[...,1:],nu0[...,:1]],axis = 2)
    y, yShft = {},{} # Segment endpoints (close to boundary and shifted)
    for i in range(2):
        y[i] = y0 + (-1)**i * l * mu
        y_   = tf.Variable(y[i])
        # Normal vector of enpoints
        with tf.GradientTape() as tape:
            Phi = Phi0(y_) + nnOut(NN,t,y_,R,T)
        nab = tape.gradient(Phi,y_) 
        nu  = nab / np.maximum(1e-10,ln.norm(nab,axis = 2,keepdims = True)) 
        # Shifted Points
        yShft[i] = y_ + iota * nu 
    # Curvature
    kap = (length(yShft[0],yShft[1])/length(y[0],y[1]) - 1)/iota
    return tf.cast(tf.reshape(kap,np.shape(kap)[:-1]),dtype = tf.float32) 

def curvature3D(y0,t,delt,Phi0,NN,R,T):
    l = delt/2; iota = l
    # Normal vector
    y0_ = tf.Variable(y0)
    with tf.GradientTape() as tape:
        Phi = Phi0(y0_) + nnOut(NN,t,y0_,R,T)
    nab = tape.gradient(Phi, y0_)
    nu0 = nab / np.maximum(ln.norm(nab,axis = 2,keepdims = True),1e-12)                   
    # Orthogonal directions
    mu1 = np.cross(nu0,nu0 + rdm.randn(1,3)); mu1 /= np.maximum(ln.norm(mu1,axis = 1,keepdims = True),1e-10)
    mu2 = np.cross(nu0,mu1)
    # Vertices of quadrangle
    y  = y0 + l * np.stack([mu1,mu2,-mu1,-mu2],axis = 0); yh = {}
    for i in range(4):
        # Normal vector of vertices
        y_ = tf.Variable(y[i])
        with tf.GradientTape() as tape:
            Phi = Phi0(y_) + nnOut(NN,t,y_,R,T)
        nab = tape.gradient(Phi,y_) 
        nu  = nab / np.maximum(1e-10,ln.norm(nab,axis = 2,keepdims = True))
        # Shifted Points
        yh[i] = y[i] + iota * nu 
    yh0 = y0 + iota * nu0
    # Curvature
    kap = (area(yh,yh0)/np.maximum(1e-10,area(y,y0)) - 1)/(2*iota)
    return tf.cast(tf.reshape(kap,np.shape(kap)[:-1]),dtype = tf.float32) 

#===============================================================#

def train(NN,probParams,newParams = {},show = True):
    """Deep Level-set Method, Training Loop"""
    for key in set(probParams.keys()).intersection(newParams.keys()):
        probParams[key] = newParams[key]
    # Problem Parameters
    d,T,R,r,L,eta,gam,alpha,c,phase,tension,Phi0,vOmega,vGamma0 = probParams.values()
    # Training Parameters
    trnParams = trainParams(probParams); trnParams["C"] = vOmega/2
    # Overwrite default training parameters (if needed)
    for key in set(trnParams.keys()).intersection(newParams.keys()):
        trnParams[key] = newParams[key]      
    B,M,K,dt,N,ts,eps,delta,bdrySim,lbda0,C = trnParams.values()
    # Display parameters
    print("Problem parameters")
    display(pd.DataFrame(probParams.values(),index = probParams.keys()))
    print("Training parameters")
    display(pd.DataFrame(trnParams.values(),index = trnParams.keys()))
    # Neural network
    if show: viz(NN,probParams,trnParams)
    # Dispersion of test functions
    aMin,aMax = (2/R)**2 * np.log(1e2), (8/R)**2 * np.log(1e2) 
    # Training
    lbdaEW, lbda = 0.1, 1 # Learning rate annealing parameters
    theta, losses, pens,lbdas = NN.trainable_weights, [],[],[]; tic()
    for m in range(M):
        if (m+1) % int(M/5) == 0 and show: 
            print("", end = f"\rIteration: {m+1}/{M}")
            showRadial(m,NN,probParams,trnParams); viz(NN,probParams,trnParams)
        z = np.reshape(simBall(K,0.,R,d),(K,d))
        a = aMin + (aMax - aMin)*rdm.rand(K)
        # Simulations
        X = {}; X_ = simBall(B,0.,R,d); X[0] = np.tile(X_,[1,N+1,1]) 
        X0 = simX0(B,R,d,Phi0,vGamma0 = vGamma0) 
        for i in np.arange(1,phase+1): X[i] = RBM(X0[i],T,N,R,alpha[i])
        if tension: 
            Y0 = simBdry0(bdrySim,NN,probParams,trnParams)
            bdryPsi, K_ = {i : np.zeros((K,N+1,N+1)) for i in [1,2]}, {}  
        iPsi = {} 
        with tf.GradientTape(persistent = True) as tape: 
            for i in np.arange(phase + 1): 
                if normalize:
                    with tf.GradientTape() as tape2:
                        x_ = tf.Variable(X[i])
                        phi = Phi0(x_) + nnOut(NN,ts,x_,R,T) 
                    nab  = tape2.gradient(phi, x_); nabNorm = ln.norm(nab,axis = 2)
                    dist = phi/np.maximum(1e-12, nabNorm)
                else: 
                    dist = Phi0(X[i]) + nnOut(NN,ts,X[i],R,T)
                # q: probability of being in the same phase
                # p: probability of entering the other phase 
                q = 1 - h(dist/eps[i]) if i == 1 else h(dist/eps[i]); p = 1 - q
                if i == 0: 
                    # Integrated test functions (left-hand side of growth condition)
                    psi_ = [PsiZ(z[k,:],a[k])(X_) * q * vOmega for k in range(K)]
                    # Displaced volume for penalty term
                    dVol = tf.reduce_mean(tfm.abs(q[:,1:] - q[:,:-1]) * vOmega,axis=0)[_t,:] 
                else:
                    V    = tfm.cumprod(tf.concat([tf.ones_like(q[:,:1]),q[:,:-1]+1e-10],axis=1),axis=1)
                    psi_ = [tf.cumsum(PsiZ(z[k,:],a[k])(X[i]) * p * V,axis=1) for k in range(K)] 
                iPsi[i]  = tf.concat([tf.reduce_mean(psi,axis=0)[_t,:] for psi in psi_],axis = 0) 
                if i > 0 and tension:
                    # Curvature terms
                    for m in range(N):
                        if len(Y0[i][m]) > 0:
                            Y = RBM(Y0[i][m],T-ts[m],N-m,R,alpha[i]) 
                            with tf.GradientTape() as tape3:
                                y_  = tf.Variable(Y)
                                phi = Phi0(y_) + nnOut(NN,ts[m:],y_,R,T) 
                            nab  = tape3.gradient(phi, y_)
                            dist = phi/np.maximum(1e-8, ln.norm(nab,axis = 2))
                            q    = 1 - h(dist/eps[i]) if i == 1 else h(dist/eps[i]); p = 1 - q
                            V    = tfm.cumprod(tf.concat([tf.ones_like(q[:,:1]),q[:,:-1]+1e-10],axis=1),axis=1)
                            psi_ = [tf.cumsum(PsiZ(z[k,:],a[k])(Y) * p * V,axis=1) \
                                    - PsiZ(z[k,:],a[k])(Y[:,:1]) for k in range(K)] 
                            if d ==2: curv = np.sign(curvature2D(Y[:,:1],ts[m],delta[i],Phi0,NN,R,T))
                            else:     curv = curvature3D(Y[:,:1],ts[m],delta[i],Phi0,NN,R,T)
                            bdryPsi[i][:,m:,m] = tf.concat([tf.reduce_mean(curv * psi,axis=0)[_t,:] for psi in psi_],axis = 0)
                    K_[i] = (gam * dt)/delta[i] * tf.cast(tf.reduce_sum(bdryPsi[i],axis=2),dtype = tf.float32)
            # Loss 
            LHS  = iPsi[0][:,:1] - iPsi[0][:,1:] 
            RHS  = eta * c[1] * iPsi[1][:,1:]/L           
            if phase == 2: RHS -= c[2] * iPsi[2][:,1:]/L       # if two-phase
            if tension:    RHS += (eta *  K_[1] - K_[2])[:,1:] # if surface tension
            loss = tf.reduce_mean((LHS - RHS)**2)     # loss
            pen  = leakyReLU(tf.reduce_max(dVol) - C) # penalty
        # Gradient update
        gradLoss = tape.gradient(loss, theta)
        gradPen  = tape.gradient(pen, theta)
        lbda     = np.mean([ln.norm(g) for g in gradLoss]) / np.maximum(np.mean([ln.norm(g) for g in gradPen]),1e-12)
        lbda     = lbda * (1-lbdaEW) + lbdaEW * lbda; lbdas.append(lbda)
        gradTot  = [gradLoss[i] + lbda0 * lbda * gradPen[i] for i in range(len(gradLoss))]
        opt.apply_gradients(zip(gradTot,theta))
        losses.append(loss.numpy()); pens.append(pen.numpy())
    runTime = toc()        
    if show: 
        print("\nTraining: %2.f seconds"%runTime)
        showLosses(losses); showLosses(pens,what = "penalty")  
    return NN,losses,pens,lbdas

