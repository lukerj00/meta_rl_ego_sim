# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:53:45 2022

@author: lukej
"""
# 1 design model (input, output, forward pass)
# 2 construct loss and optimizer
# 3 training loop
# -forward pass (compute prediction)
# -backward pass (gradients)
# -update weights


import numpy as np
import torch
import torch.nn as nn

X = torch.tensor([[3],[6],[9],[12]], dtype=torch.float32)
Y = torch.tensor([[10],[19],[28],[37]], dtype=torch.float32)
# 4x1 tensor, each row is a sample

n_samples, n_features = X.shape

Xtest = torch.tensor([5], dtype=torch.float32)

#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #must be torch tensor and must require gradients as want dl/dw (updating w) 

### training params
learning_rate = 0.005
n_iters = 2000

### prediction (manual)

#def forward(x):
#    return w * x # yhat

in_size = n_features
out_size = n_features

# model = nn.Linear(in_size, out_size) # requires x and y have individual rows as samples; use []
# pytorch model implicitly contains parameters w

class LinearRegression(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(in_size, out_size)

### loss (manual)

#def loss(y,yhat):
#    return ((yhat-y)**2).mean()

loss = nn.MSELoss()

### grad (manual)

#def gradient(x,y,yhat):
#    return np.dot(3*x,yhat-y).mean()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer now takes in parameters attribute of model (w)

print(f'prediction pre-training: f(5) = {model(Xtest).item():.3f}')

for epoch in range(n_iters):
    # pred = forward pass
    yhat_ = model(X)
    # now call 'model' for forward pass instead of forward()
    
    # loss
    l = loss(Y, yhat_) # yhat_ is fnc of w
    # now call automatically defined 'loss' to calculate objective
    
    # grad = backward pass
    l.backward() #dl/dw (as l is fnc of w)
    # accumulates grad in w.grad
    
    # gradient (manual)
    #dw = gradient(X,Y,yhat_)
    
    # update weights
    #with torch.no_grad(): # 'not part of gradient tracking graph' ie this computation doesnt need gradients
    #    w -= learning_rate * w.grad
    
    # update weights
    optimizer.step() # automatic optimisation step
    
    #w.grad.zero_() # must zero grad after each iteration
    optimizer.zero_grad()
    
    if epoch % 50 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(Xtest).item():.3f}')

