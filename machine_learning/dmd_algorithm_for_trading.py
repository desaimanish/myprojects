import numpy as np
from scipy.io import FortranFile
from os import listdir
#%% 
g = '/home/manish/dmd/dataset' #dataset directory path
size = len(listdir(g))   # finding out how many files are present
m= np.zeros((256*256*256,size)) #initializing the variable m which stores the domain in the form  of arrays
n = 0 # counter

#%% Loop to read the data from the binary data
for i in listdir(g):    
    f = FortranFile(g+"/" + i)
    data = f.read_record(dtype=np.int32) 
    data1 = f.read_record(dtype = np.int32)
    data2 =f.read_record(dtype = np.int32)
    data3 = f.read_record(dtype = np.float64)
    data4  =f.read_record(dtype = np.float64)
    data5  =f.read_record(dtype = np.float64)
    data6  =f.read_record(dtype = '<f')
    print(len(data1),len(data2),len(data3),len(data4),len(data5),len(data6))
    m[:,n]=data6
    n = n+1
    break
#%% executing the SVD
x = m[:,:-1]  
xd = m[:,1:]
u,s,v = np.linalg.svd(x, full_matrices=False)
#%%
s= np.diag(s)
#%%
r=30
su = u[:,:r]
sv = v[:r,:]
ss = s[:r,:r]
#%%
suh = su.T
svh = sv.T
ssh = np.linalg.inv(ss)
atilda = suh@xd@svh@ssh
l,w = np.linalg.eig(atilda)
phi = xd@svh@ssh@w
#%%
rphi = phi.real
iphi = phi.imag
#%%
del(data6,u,v)
#%%


c1 = np.zeros(((256,256,256)))
c2 = np.zeros(((256,256,256)))
c3 = np.zeros(((256,256,256)))
for k in range(256):
  for j in range(256):
    for i in range(256):
      c1[i,j,k] = data3[i]
      c2[i,j,k] = data4[j]
      c3[i,j,k] = data5[k]

n= 0
d= np.zeros((256*256*256,3))
for k in range(256):
  for j in range(256):
    for i in range(256):
      d[n,0] =  c1[i,j,k]
      n = n+1
n= 0

for k in range(256):
  for j in range(256):
    for i in range(256):
      d[n,1] =  c2[i,j,k]
      n = n+1
n= 0

for k in range(256):
  for j in range(256):
    for i in range(256):
      d[n,2] =  c3[i,j,k]
      n = n+1

#%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
for i in range(r):
    fig = plt.figure()
    ax = fig.add_subplot( projection='2d')
    
    
    img = ax.scatter(d[:,0], d[:,1] c=rphi[:,i], cmap=plt.jet())
    fig.colorbar(img)
    plt.show()