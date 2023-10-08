import numpy as np
from sklearn.preprocessing import OneHotEncoder


# clip=True can also serve as a regularization technique as it prevents a rapid decrease in the sum of the Hessian.

class RFLBinary():
    def __init__(self, r, q, sklearn=False, clip=False):
        self.r = r
        self.q = q
        self.clip = clip  
        self.sklearn = sklearn
        self.epsilon = 1e-16

    def __call__(self, labels, preds):
        
        pt = self.sigmoid(preds)
        pt[labels==0] = 1-pt[labels==0]
        grad = np.zeros(preds.shape[0])
        hess = np.zeros(preds.shape[0])
        
        grad = (2*labels-1)*self.grad(pt)
        hess = self.hess1(pt)+self.hess2(pt)

        if self.clip:
            return grad, np.maximum(hess, self.epsilon)
        else:
            return grad, hess
    
    def grad(self, p):
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)] 
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = (self.r*p1*((1-p1)**self.r)*(p1**self.q-1)-\
                self.q*(p1**self.q)*(1-p1)**(self.r+1))/self.q  # p>=0.5
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = (p1**self.q)*(p1-1)
        elif self.r > 0 and self.q == 0:
            p1 = np.clip(p1, 1e-9, 1)
            result[(0<p)&(p<1)] = ((1-p1)**self.r)*(self.r*p1*np.log(p1)+p1-1)
        else:
            result[(0<p)&(p<1)] = p1-1
        return result

    def hess1(self, p):  # dldp2*dp**2
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)]
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = ((1-p1)**self.r)*(-self.r*(self.r-1)*(p1**self.q-1)*p1**2-\
                self.q*(self.q-1)*(p1**self.q)*(p1-1)**2-2*self.r*self.q*p1**(self.q+1)*(p1-1))/self.q
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = (1-self.q)*(p1**self.q)*(p1-1)**2
        elif self.r > 0 and self.q == 0:
            p1 = np.clip(p1, 1e-12, 1)
            result[(0<p)&(p<1)] = ((1-p1)**self.r)*(-self.r*(self.r-1)*p1**2*np.log(p1)-\
                2*self.r*p1*(p1-1)+(p1-1)**2)
        else:
            result[(0<p)&(p<1)] = (p1-1)**2
        return result
    
    def hess2(self, p):  # dldp*dp2
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)] 
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = (1-2*p1)*(self.r*((1-p1)**self.r)*(p1**self.q-1)*p1-\
                self.q*(p1**self.q)*((1-p1)**(self.r+1)))/self.q  # p>=0.5
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = -(p1**self.q)*(p1-1)*(2*p1-1)
        elif self.r > 0 and self.q == 0:
            p1 = np.clip(p1, 1e-12, 1)
            result[(0<p)&(p<1)] = ((1-p1)**self.r)*(1-2*p1)*(self.r*p1*np.log(p1)+p1-1)
        else:
            result[(0<p)&(p<1)] = (1-p1)*(2*p1-1)
        return result 

    def sigmoid(self, x):
        kEps = 1e-16 #  avoid 0 div
        x = np.minimum(-x, 88.7)  # avoid exp overflow
        return 1 / (1 + np.exp(x)+kEps)



class XGBRFLMulti():
    def __init__(self, r, q, clip=False):
        self.r = r
        self.q = q
        self.clip = clip
        self.epsilon = 1e-16

    def __call__(self, labels, preds):
        p = self.softmax(preds)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
        
        pt = np.sum(p*encoded_label, axis=1, keepdims=True)
        
        grad = self.dldpXp(pt)*(encoded_label-p)
        hess = self.dldp2Xp2(pt)*(encoded_label-p)**2+\
            self.dldpXp(pt)*(encoded_label-p)*(1-2*p)

        if self.clip:
            hess = np.maximum(hess, self.epsilon)
        return grad.reshape(grad.shape[0]*grad.shape[1]), hess.reshape(hess.shape[0]*hess.shape[1])
    
    def dldpXp(self, p):  # dldp*p
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)] 
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = (self.r*(1-p1)**(self.r-1)*(p1**self.q)-\
                self.q*(p1**self.q)*(1-p1)**self.r)/self.q  # p>=0.5
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = -p1**self.q
        elif self.r > 0 and self.q == 0:
            result[(0<p)&(p<1)] = (1-p1)**(self.r-1)*(self.r*p1*np.log(p1)+p1-1)
        else:
            result[(0<p)&(p<1)] = -1
        return result
        
    def dldp2Xp2(self, p):  # dldp2*p**2
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)]
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = -(1-p1)**(self.r-2)*(self.r*(self.r-1)*(p1**self.q-1)*p1**2+\
                self.q*(self.q-1)*(p1**self.q)*(p1-1)**2+2*self.r*self.q*(p1**(self.q+1))*(p1-1))/self.q
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = (1-self.q)*(p1**self.q)
        elif self.r > 0 and self.q == 0:
            result[(0<p)&(p<1)] = -(1-p1)**(self.r-2)*(self.r*(self.r-1)*p1**2*np.log(p1)+\
                2*self.r*p1*(p1-1)-(p1-1)**2)
        else:
            result[(0<p)&(p<1)] = 1
        return result

    def softmax(self, x):
        kEps = 1e-16 #  avoid 0 div
        x = np.minimum(x, 88.7)  # avoid exp overflow
        e = np.exp(x)
        return e / np.expand_dims(np.sum(e, axis=1)+kEps, axis=1)


class LGBRFLMulti():
    def __init__(self, r, q, clip=False):
        self.r = r
        self.q = q
        self.clip = clip
        self.epsilon = 1e-16

    def __call__(self, labels, preds):
        preds = preds.reshape((labels.shape[0], -1), order='F')
        p = self.softmax(preds)
        grad = np.zeros(preds.shape)
        hess = np.zeros(preds.shape)
        encoder = OneHotEncoder()
        encoded_label = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
        
        pt = np.sum(p*encoded_label, axis=1, keepdims=True)
        
        grad = self.dldpXp(pt)*(encoded_label-p)
        hess = self.dldp2Xp2(pt)*(encoded_label-p)**2+\
            self.dldpXp(pt)*(encoded_label-p)*(1-2*p)
    
        if self.clip:
            hess = np.maximum(hess, self.epsilon)
        return grad.reshape(grad.shape[0]*grad.shape[1], order='F'), hess.reshape(hess.shape[0]*hess.shape[1], order='F')
    
    def dldpXp(self, p):  # dldp*p
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)] 
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = (self.r*(1-p1)**(self.r-1)*(p1**self.q-1)*p1-\
                self.q*(p1**self.q)*(1-p1)**self.r)/self.q  # p>=0.5
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = -p1**self.q
        elif self.r > 0 and self.q == 0:
            result[(0<p)&(p<1)] = (1-p1)**(self.r-1)*(self.r*p1*np.log(p1)+p1-1)
        else:
            result[(0<p)&(p<1)] = -1
        return result
        
    def dldp2Xp2(self, p):  # dldp2*p**2
        result = np.zeros(p.shape)
        p1 = p[(0<p)&(p<1)]
        if self.r > 0 and self.q > 0:
            result[(0<p)&(p<1)] = -(1-p1)**(self.r-2)*(self.r*(self.r-1)*(p1**self.q-1)*p1**2+\
                self.q*(self.q-1)*(p1**self.q)*(p1-1)**2+2*self.r*self.q*(p1**(self.q+1))*(p1-1))/self.q
        elif self.r == 0 and self.q > 0:
            result[(0<p)&(p<1)] = (1-self.q)*(p1**self.q)
        elif self.r > 0 and self.q == 0:
            result[(0<p)&(p<1)] = -(1-p1)**(self.r-2)*(self.r*(self.r-1)*p1**2*np.log(p1)+\
                2*self.r*p1*(p1-1)-(p1-1)**2)
        else:
            result[(0<p)&(p<1)] = 1
        return result

    def softmax(self, x):
        kEps = 1e-16 #  avoid 0 div
        x = np.minimum(x, 88.7)  # avoid exp overflow
        e = np.exp(x)
        return e / np.expand_dims(np.sum(e, axis=1)+kEps, axis=1)
    
    