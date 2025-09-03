from scipy.special import legendre 
import numpy as np  
import matplotlib.pyplot as plt
from scipy.integrate import quad

a = 1000

#Generating stiffness matrix

x = np.linspace(0,1,100)

def integrateMatrix(handle,a,b,dim):
    '''Integrates function over the range for a matrix [a,b] based on matrix square dimension of dim'''
    mOut = np.zeros((dim,dim))
    for i in range(dim):
        for ii in range(dim):
            (mOut[i,ii],error) = quad(lambda x: handle(x)[i,ii],a,b)
    return mOut 
    
def integrateVector(handle,a,b,dim):
    '''Integrates function over the range for a matrix [a,b] based on matrix square dimension of dim'''
    mOut = np.zeros((dim,))
    for i in range(dim):
        (mOut[i],error) = quad(lambda x: handle(x)[i],a,b)
    return mOut
    

class modL():
    def __init__(self,order):
        self.lHandle = legendre(order)
        self.dBool = False 
    def deriv(self,order):
        self.dBool = True
        self.dLHandle = self.lHandle.deriv(1)
        return self 
    def __call__(self,x):
        if self.dBool:
            return self.dLHandle(x)*(x-x**2) + (1-2*x)*self.lHandle(x)
        else:
            return self.lHandle(x)*(x-x**2)
            
class modL2():
    def __init__(self,order):
        self.lHandle = legendre(order)
        self.dBool = False 
    def deriv(self,order):
        self.dBool = True
        self.dLHandle = self.lHandle.deriv(1)
        return self 
    def __call__(self,x):
        if self.dBool:
            return self.dLHandle(x)
        else:
            return self.lHandle(x)
        


class basisFuns():
    #This can be updated to include a function to map to domain of 0 to 1
    def __init__(self,orderStart,orderEnd):
        self.orderStart = orderStart
        self.orderEnd = orderEnd 
        self.funCount = orderEnd - orderStart
        self.N = []
        self.dN = [] 
        index = 0

        for i in range(orderStart,orderEnd):
            self.N.append(modL(i))
            self.dN.append(modL(i).deriv(1))
    
            
    def __call__(self,x):
        if type(x) == np.ndarray:
            rows = len(x)
        elif type(x) == list:
            rows = len(x) 
        else:
            rows = 1
        Nval = np.zeros((rows,self.funCount))
        dNval = np.zeros((rows,self.funCount))
        for i in range(self.funCount):
            Nval[:,i] = self.N[i](x)
            dNval[:,i] = self.dN[i](x)
        return Nval, dNval
    def plot(self,x):
        (y,dy) = self.__call__(x)
        plt.figure()
        for i in range(self.funCount):
            plt.plot(x,y[:,i], label = f'Fun Number: {i}')
        plt.grid("on")
        plt.legend()
        plt.title("Basis Functions")
        plt.show()
        
        plt.figure()
        for i in range(self.funCount):
            plt.plot(x,dy[:,i], label = f'Fun Number: {i}')
        plt.grid("on")
        plt.legend()
        plt.title("Derivative of basis functions")
        plt.show()
    def NVals(self,x):
        if type(x) == np.ndarray:
            rows = len(x)
        elif type(x) == list:
            rows = len(x) 
        else:
            rows = 1
        Nval = np.zeros((rows,self.funCount))
        for i in range(self.funCount):
            Nval[:,i] = self.N[i](x)
        return Nval 
    def dNVals(self,x):
        if type(x) == np.ndarray:
            rows = len(x)
        elif type(x) == list:
            rows = len(x) 
        else:
            rows = 1
        dNval = np.zeros((rows,self.funCount))
        for i in range(self.funCount):
            dNval[:,i] = self.dN[i](x)
        return dNval 
  
class basisFuns2():
    #This can be updated to include a function to map to domain of 0 to 1
    def __init__(self,orderStart,orderEnd):
        self.orderStart = orderStart
        self.orderEnd = orderEnd 
        self.funCount = orderEnd - orderStart
        self.N = []
        self.dN = [] 
        index = 0

        for i in range(orderStart,orderEnd):
            self.N.append(modL2(i))
            self.dN.append(modL2(i).deriv(1))
    
            
    def __call__(self,x):
        if type(x) == np.ndarray:
            rows = len(x)
        elif type(x) == list:
            rows = len(x) 
        else:
            rows = 1
        Nval = np.zeros((rows,self.funCount))
        dNval = np.zeros((rows,self.funCount))
        for i in range(self.funCount):
            Nval[:,i] = self.N[i](x)
            dNval[:,i] = self.dN[i](x)
        return Nval, dNval
    def plot(self,x):
        (y,dy) = self.__call__(x)
        plt.figure()
        for i in range(self.funCount):
            plt.plot(x,y[:,i], label = f'Fun Number: {i}')
        plt.grid("on")
        plt.legend()
        plt.title("Basis Functions")
        plt.show()
        
        plt.figure()
        for i in range(self.funCount):
            plt.plot(x,dy[:,i], label = f'Fun Number: {i}')
        plt.grid("on")
        plt.legend()
        plt.title("Derivative of basis functions")
        plt.show()
    def NVals(self,x):
        if type(x) == np.ndarray:
            rows = len(x)
        elif type(x) == list:
            rows = len(x) 
        else:
            rows = 1
        Nval = np.zeros((rows,self.funCount))
        for i in range(self.funCount):
            Nval[:,i] = self.N[i](x)
        return Nval 
    def dNVals(self,x):
        if type(x) == np.ndarray:
            rows = len(x)
        elif type(x) == list:
            rows = len(x) 
        else:
            rows = 1
        dNval = np.zeros((rows,self.funCount))
        for i in range(self.funCount):
            dNval[:,i] = self.dN[i](x)
        return dNval       
        
# EXAMPLE USE CASE FOLLOWS -----------------:



# a = 1000

#Generating stiffness matrix

# x = np.linspace(0,1,100)

# BFun = basisFuns(1,8)
# BFun.plot(x)
#      
# def f(x):
    # (y,dy) = BFun(x)
    # return np.matmul(dy.transpose(),dy)*a
    


# K = integrateMatrix(f,0,1,BFun.funCount)
# KInv = np.linalg.inv(K)

# B = BFun.NVals(np.array([0.5,0.2]))
# print(B.shape)




