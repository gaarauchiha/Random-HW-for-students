#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

bdata = pd.read_csv("./boston.csv")

X = np.array(bdata.loc[:, ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", 
                            "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", 
                            "LSTAT"]], dtype='float')
y = np.array(bdata["PRICE"], dtype='float')

# Preprocessing and normalizing the data for better fit
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) # This is bias
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def compute_loss(W, b, X, y):
    predictions = X[:, 1:] @ W + b  
    return (1 / len(y)) * np.sum((y - predictions) ** 2)

def stochastic_gradient_descent(X, y, learning_rate=0.001, epochs=1000, batch_size=1):
    m, n = X.shape
    W = np.random.randn(n - 1) * 0.01  
    b = np.random.randn() * 0.01       
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)  # We shuffle indices for SGD
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            predictions = X_batch[:, 1:] @ W + b

            error = predictions - y_batch
            W_grad = (2 / batch_size) * (X_batch[:, 1:].T @ error)
            b_grad = (2 / batch_size) * np.sum(error)

            W -= learning_rate * W_grad
            b -= learning_rate * b_grad

        loss_history.append(compute_loss(W, b, X, y))

    return W, b, loss_history

batch_sizes = [1, 3, 5,10,30]
plt.figure(figsize=(12, 6))

for batch_size in batch_sizes:
    W_optimal, b_optimal, loss_history = stochastic_gradient_descent(X_train, y_train,
                                                                    learning_rate=0.001,
                                                                    epochs=100,
                                                                    batch_size=batch_size)
    
    plt.plot(loss_history, label=f'Batch Size: {batch_size}')

plt.title('Convergence of Loss Function with Different Batch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[3]:


def stochastic_gradient_descent_with_momentum(X, y, learning_rate=0.001, epochs=1000, batch_size=1, beta=0.9):
    m, n = X.shape
    W = np.random.randn(n - 1) * 0.01  
    b = np.random.randn() * 0.01        
    loss_history = []
    
    v_W = np.zeros_like(W)
    v_b = 0                  

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            predictions = X_batch[:, 1:] @ W + b

            error = predictions - y_batch
            W_grad = (2 / batch_size) * (X_batch[:, 1:].T @ error)
            b_grad = (2 / batch_size) * np.sum(error)

            v_W = beta * v_W + (1 - beta) * W_grad
            v_b = beta * v_b + (1 - beta) * b_grad

            W -= learning_rate * v_W
            b -= learning_rate * v_b

        loss_history.append(compute_loss(W, b, X, y))

    return W, b, loss_history

plt.figure(figsize=(12, 6))

W_optimal_sgd, b_optimal_sgd, loss_history_sgd = stochastic_gradient_descent(X_train, y_train,
                                                                          learning_rate=0.001,
                                                                          epochs=100,
                                                                          batch_size=5)

# SGD with Momentum
W_optimal_momentum, b_optimal_momentum, loss_history_momentum = stochastic_gradient_descent_with_momentum(X_train, y_train,
                                                                                                      learning_rate=0.001,
                                                                                                      epochs=100,
                                                                                                      batch_size=5)

plt.plot(loss_history_sgd, label='SGD without Momentum')
plt.plot(loss_history_momentum, label='SGD with Momentum')
plt.title('Convergence of Loss Function: SGD vs SGD with Momentum')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[5]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train[:, 1:], y_train)  

# Coeff and intercept
W_sklearn = model.coef_
b_sklearn = model.intercept_

predictions_sklearn = model.predict(X_test[:, 1:])

print("SGD Weights:", W_optimal_momentum)
print("SGD Bias:", b_optimal_momentum)
print("Sklearn Weights:", W_sklearn)
print("Sklearn Bias:", b_sklearn)

loss_sklearn = compute_loss(W_sklearn, b_sklearn, X_test, y_test)
print("Loss from Sklearn Linear Regression:", loss_sklearn)


# In[17]:


# Question 2

import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-1, 2, 400)
x2_1 = (1 - x1) / 2   # x1 + 2x2 <= 1
x2_2 = (1 - 2*x1)     # 2x1 + x2 <= 1

plt.figure(figsize=(8, 6))
plt.plot(x1, x2_1, label=r'$x_1 + 2x_2 \leq 1$')
plt.plot(x1, x2_2, label=r'$2x_1 + x_2 \leq 1$')

plt.fill_between(x1, np.minimum(x2_1, x2_2), -1, where=(np.maximum(x2_1, x2_2) > -1), color='green', alpha=0.2)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.title("Feasible Region")
plt.legend()
plt.grid(True)
plt.show()


# In[24]:


# Question 3

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 48*x + 96*y - x**2 - 2*x*y - 9*y**2

x = np.linspace(0, 30, 400)
y = np.linspace(0, 60, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 20x + 4y = 216
constraint_x = np.linspace(0, 10.8, 400)
constraint_y = (216 - 20*constraint_x) / 4

plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.5)
plt.colorbar(contour)

plt.plot(constraint_x, constraint_y, color='red', label='Constraint: 20x + 4y = 216')

plt.fill_between(constraint_x, 0, constraint_y, color='blue', alpha=0.2, label='Feasible Domain')

plt.title('Contour Plot of f(x, y) with Feasible Region')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, 30)
plt.ylim(0, 60)
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




