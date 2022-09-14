
# *********************************************************************************************************************************************************************
#%% ***IMPORTANT NOTICE***
#%% This is just a basic implimentation of DMD. The developed sparsiity promoted DMD and iterative DMD are more efficient to use for stoct market prediction.
# as the code is confidential it is not shown in public, but if you need the code how it works mail me Mail ID: desai.manish736@gmail.com
# *********************************************************************************************************************************************************************



import numpy as np
from scipy.io import FortranFile
from os import listdir
#%% 
g = '/home/manish/dmd/dataset' #dataset directory path
size = len(listdir(g))   # finding out how many files are present
m= np.zeros((256*256*256,size)) #initializing the variable m which stores the domain in the form  of arrays
n = 0 # counter

#%% Loop to read the data from the binary data(this may change based on the dataset you are reading)
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


