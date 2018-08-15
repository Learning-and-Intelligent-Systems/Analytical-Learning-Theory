'''
Created on 31 Dec 2017

@author: vermav1
'''
import numpy as np

x=np.asarray((0.2,0.3,0.5))
y=np.asarray((0.9,0.01,0.1))

ce=0
for i in xrange(x.shape[0]):
    ce+=-(x[i]*np.log(y[i]))

print ce