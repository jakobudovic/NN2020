#!/usr/bin/env python
# coding: utf-8

# ##### Author contributions
# Please fill out for each of the following parts who contributed to what:
# - Conceived ideas: 
# - Performed math exercises: 
# - Performed programming exercises:
# - Contributed to the overall final assignment: 

# # Chapter 1
# ## Introduction
# 
#     Hand-in bug-free (try "Kernel" > "Restart & Run All") and including all (textual as well as figural) output via Brightspace before the deadline (see Brightspace).
#     
# Learning goals:
# 1. Get familiar with jupyter notebooks
# 2. Brush up basics of vectors and matrices
# 3. Get familiar with python
# 4. Get familiar with activation functions

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# ### Notes
# 
# For exercises 1-5 you are required to write down derivations explicitly in $\LaTeX$. Thus, you do not need to write any Python code, but you might find calculating the exercises with Python useful for practice. We denote such exercises with "$\LaTeX$ here."
# 
# For exercises 6-8, you do have to write Python code. We denote these with "## Code here ##". Make sure that your plots are shown *in* the notebook when we open it; which you can achieve by saving the notebook when all plots are open and handing in this version. 
# 
# As mentioned in the course reader, in every assignment we will check whether your notebook code runs through without errors with the Cell->Run All command. You risk loosing many points if this fails. We will not debug your code. Be sure to restart the notebook kernel from time to time (Kernel->Restart) and before submitting to notice it if you use old variable or function names that are still defined in the notebook kernel, but not anymore in the code. 
# 
# A useful reference for linear algebra is [**The Matrix Cookbook**](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274). A good overview for partial derivatives is [Khan Academy: Partial Derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives-and-the-gradient/a/introduction-to-partial-derivatives).

# ### Exercise 1: Vector operations (1 point)
# Let's look at vectors. Work out this assignment by hand and write down your solution in markdown.
# 
# Let $\mathbf{x} = (1,2)^T$ and $\mathbf{y} = (-1,1)^T$
# 
# 1. How much is $10\mathbf{x}$?
# 1. What is the length (norm) of the vector $\mathbf{x}$? Briefly show how to calculate the solution. 
# 1. How much is $\mathbf{x}^T\mathbf{y}$?
# 1. What is the angle between $\mathbf{x}$ and $\mathbf{y}$ in degrees? Briefly show how to calculate the solution. 
# 

# ### Solution 1
# 1. $\mathbf{10x} = (10,20)^T$
# 1. 2            ``
# 1. $\mathbf{x}^T\mathbf{y} = 1$ 
# 1. 0.9504152802551828

# In[65]:


import torch
import numpy as np
import math


# In[66]:


x = np.array([1,2])
y = np.array([-1,1])

# 2.
print("----------------")
print(np.linalg.norm(x))              # using library
print(math.sqrt(x[0]**2 + x[1]**2))   # "by hand"

# 3. 
print("----------------")

r = np.dot(x.transpose(), y)          # using library
print(r)


result = [0]

print("lenx:", len(x))

for i in range(len(x)):
    result[0] += x[i] * y[i] 
    print("h:", x[i] * y[i] )
print(x, "*", y, "=", result)
    

# 4.
# arccos(alpha) = (x*y)/(|x|*|y|)
print("----------------")

lenx = np.linalg.norm(x)
leny = np.linalg.norm(y)
print(lenx, leny)
arccosAlpha = result[0]/(lenx * leny)
print(arccosAlpha)
print("angle is: ", np.cos(arccosAlpha))


# ### Exercise 2: Vectors and matrices (1 point)
# Let's look at vectors and matrices. Work out this assignment by hand and write down your solution in markdown. 
# 
# Let $\mathbf{x} = (1,2)^T$ and $\mathbf{A} = 
# \left(
# \begin{array}{cc}
# 1 & 2 \\
# 3 & 4
# \end{array}
# \right)
# $.
# 
# 1. Can we compute $\mathbf{x}\mathbf{A}$? Why can, or why can't we?
# 1. How much is $\mathbf{A}\mathbf{x}$?

# ### Solution 2
# 1. We can't, because the dimensions of the matrices don't match.
# 1. $\mathbf{Ax} = (5,11)^T$

# ### Exercise 3: Matrices (1 point)
# Let's look at matrices. Work out this assignment by hand and write down your solution in markdown. 
# 
# Let $\mathbf{A} = 
# \left(
# \begin{array}{cc}
# 1 & 2 \\
# 3 & 4
# \end{array}
# \right)
# $ and $\mathbf{B} = 
# \left(
# \begin{array}{cc}
# 5 & 6 \\
# 7 & 8
# \end{array}
# \right)
# $.
# 
# 1. How much is $AB$?
# 1. How much is $BA$?

# ### Solution 3
# 1.  $\mathbf{AB} = 
# \left(
# \begin{array}{cc}
# 19 & 24 \\
# 43 & 50 
# \end{array}
# \right)
# $
# 1. $\mathbf{BA} = 
# \left(
# \begin{array}{cc}
# 23 & 34 \\
# 31 & 46 
# \end{array}
# \right)
# $

# ### Exercise 4: Partial derivatives (1 point)
# 
# Let's brush up on partial derivatives. 
# 
# Let $\mathbf{x} = (x_1,\ldots,x_i,\ldots,x_n)^T$ (a vector) and $f(\mathbf{x}) = \mathbf{x}^T\mathbf{x}$. Write down the expression for the partial derivative $\frac{\partial f}{\partial x_i}$. Briefly explain how you arrived at the result. For the mathematicions, all $x_i$ are i.i.d.
# 
# Hint: How would the function $f(\mathbf{x})$ look like if it was written with the vector scalars $x_i$ instead of the vector $\mathbf{x}$?

# ### Solution 4
# 
# Since we're multiplying 2 martrices of size $1\times n$ and $n\times 1$ (1 row with one column), we're getting just 1 value out of it. The value is the sum of the all elements squared (we can see that by bultiplying matrices above), so we get something like $x_1^2 + ... + x_i^2 + ... + x_n^2$.
# 
# Now we just make a partial derivation with respect to $x_i$ and we treat other elements as constants. As we know, constants go to zero or "disappear" after derivation, so we are left with only element $x_i^2$. 
# 
# After the derivation of $x_i^2$ we get $2x_i$ which is the answer to our problem.
# 
# 
# Answer: $2x_i$

# ### Exercise 5: Gradients (1 point)
# Often, we need to compute the gradient of a particular function. Given a function $f(x_1,\ldots,x_n)$, the gradient is just a collection of partial derivatives:
# \begin{equation*}
# \nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots,\frac{\partial f}{\partial x_n}\right) \,.
# \end{equation*}
# 
# Let $f(x,y) = - (\cos^2 x + \cos^2 y)^2$. 
# 
# Derive the gradient $\nabla f = \left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\right)$.

# ### Solution 5
# 
# $\nabla f = 
# \left(
# 4\sin x \cos x (\cos^2 x + \cos^2 y),
# 4\sin y \cos y (\cos^2 x + \cos^2 y)
# \right)
# $
# 
# 

# ### Exercise 6: Linear activation function (1 point)
# Define a python function that computes the *linear activation function* (trivial identity) for any given input. Plot it over the input range $x \in [-10,10]$, and don't forget to add sensible labels to the axes. 
# 
# Hint: use `np.arange()` with a sensible stepsize

# In[6]:


import matplotlib.pyplot as plt

def plot(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid(True, which='both')

    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')


# ### Solution 6

# In[7]:


# Linear activation function: 
import numpy as np 

const = 1;
linact = lambda x: const * x

# Plot activation over given range: 
arr = np.arange(-10, 10, 1)
plot(arr, linact(arr))


# ### Exercise 7: Linear threshold activation function (1 point)
# Define a python function that computes the *linear threshold activation function* (also known as step activation function) for any given input. This activation function has a parameter $\theta$, which you can, by default, set to $\theta=0$. Plot it over the input range $x \in [-10,10]$, and don't forget to add sensible labels to the axes. 
# 
# Hint: use `np.arange()` with a sensible stepsize

# ### Solution 7

# In[14]:


# Linear threshold activation function: 

const = 1
thresh = 0
linactthresh = np.vectorize(lambda x: 1.0 if(const*x >= thresh) else 0)

# Plot activation over given range: 
X = np.arange(-10, 11, 1)
plot(X, linactthresh(X))


# ### Exercise 8: Sigmoid activation function (1 point)
# Define a python function that computes the *sigmoid activation function* for any given input. Plot it over the input range $x \in [-10,10]$, and don't forget to add sensible labels to the axes. 
# 
# Hint: use `np.arange()` with a sensible stepsize

# ### Solution 8

# In[17]:


# Linear sigmoid activation function: 
sigmoid = lambda x: 1 / (1 + np.exp(1)**-x)


# Plot activation over given range: 
X = np.arange(-10, 11, 1)
plot(X, sigmoid(X))


# ### Exercise 9 (1 point)
# 
# **1.** The input of the activation function in a simple perceptron (or any regular neural network neuron) is calculated as a weighted sum between each input value $x_i$ and each corresponding weight $w_i$, that is: $\sum_{i=1}^m w_i x_i $.
# 
# Calculate the input of the activation function for the given input values ```x_inputs``` and weight values ```weights``` **in a for-loop**. 

# ### Solution 9.1

# In[26]:


x_inputs = np.array([4.0,2.0,3.0])
weights  = np.array([0.7,0.3,0.2])

print("Shape of inputs: {}.".format(x_inputs.shape))
print("Shape of weights: {}.".format(weights.shape))

activation_input = 0.0

# Write a for-loop
for x, weight in zip(x_inputs, weights):
    activation_input += x * weight

print("The input of the activation function is: {}.".format(activation_input))


# **2.** For-loops tend to be slow. There is a direct mathematical operation that expresses the same as our weighted sum above. This operation is also efficiently implemented as a ```numpy``` function. 
# 
# How is the operation called? Use the corresponding ```numpy``` function **once** to calculate ```activation_input``` in one line without any for-loop. (this also excludes list comprehensions)
# 
# Hint: $\sum_{i=1}^m w_i x_i = \mathbf{w}^\top \mathbf{x}$ 
# 
# Note: this one operation is the prefered operation over using for loops, so use it in all upcoming assignments!

# ### Solution 9.2

# In[27]:


x_inputs = np.array([4.0,2.0,3.0])
weights  = np.array([0.7,0.3,0.2])

print("Shape of inputs: {}.".format(x_inputs.shape))
print("Shape of weights: {}.".format(weights.shape))

# Write a one-liner
np.inner(x_inputs, weights)

print("The input of the activation function is: {}.".format(activation_input))


# ### Exercise 10 (1 point)

# **1.** When implementing a full neural network we will have multiple $h_n$ hidden units (think $h_n$ individual perceptrons). In a multi-layer perceptron (a simple fully connected neural network), every hidden unit $h_i$ is connected to all of the $m$ input units, leading to $m \times h_n$ weights in total. Again, first implement this with for-loops only, and with none of numpy's special mathematical functions. In the example below `weights` represents the weights for 4 hidden units. 

# ### Solution 10.1

# In[32]:


x_inputs = np.array([4.0,2.0,3.0])
weights  = np.array([[0.7,0.3,0.2], 
                     [-0.23,0.42,-0.1], 
                     [-1.5,-2.3,0.4], 
                     [0.83,-0.12,-0.7]])

print("Shape of inputs: {}.".format(x_inputs.shape))
print("Shape of weights: {}.".format(weights.shape))

activation_inputs = np.zeros([weights.shape[0],])

# Write a for-loop
for i in range(weights.shape[0]):
    activation_inputs[i] += sum([x * w for x, w in zip(x_inputs, weights[i])])

print("The inputs of the activation functions for the hidden units are: {}.".format(activation_inputs))


# **2.** Now implement the same with the operation you found before. You should only use this function once and should not use *any* for loops.  This also excludes list comprehensions. 
# 
# Note: this one operation is the prefered operation over using for loops, so use it in all upcoming assignments!

# ### Solution 10.2

# In[34]:


x_inputs = np.array([4.0,2.0,3.0])
weights  = np.array([[0.7,0.3,0.2], 
                     [-0.23,0.42,-0.1], 
                     [-1.5,-2.3,0.4], 
                     [0.83,-0.12,-0.7]])

print("Shape of inputs: {}.".format(x_inputs.shape))
print("Shape of weights: {}.".format(weights.shape))

activation_inputs = np.zeros([weights.shape[0],])

# Write a one-liner
activation_inputs = np.inner(x_inputs, weights)

print("The inputs of the activation functions for the hidden units are: {}.".format(activation_inputs))


# **3.** Usually you would process multiple examples at once (*in a batch*), generating a unit activation individually for every example. `x_inputs` now carries two examples. Now - using only the one-line operation you found before one time - again gather the activations. You should not use *any* for-loops. This also excludes list comprehensions. 
# 
# Note: this one operation is the prefered operation over using for loops, so use it in all upcoming assignments!

# ### Solution 10.3

# In[36]:


x_inputs = np.array([[4.0,2.0,3.0], 
                     [3.0,0.5,4.0]])

weights  = np.array([[0.7,0.3,0.2], 
                     [-0.23,0.42,-0.1], 
                     [-1.5,-2.3,0.4], 
                     [0.83,-0.12,-0.7]])

print("Shape of inputs: {}.".format(x_inputs.shape))
print("Shape of weights: {}.".format(weights.shape))

activation_inputs = np.zeros([weights.shape[0], 2])

# Write a one-liner
activation_inputs = np.inner(x_inputs, weights)

print("The 2 sets of inputs of the activation functions for the hidden units are: {}.".format(activation_inputs))


# In[ ]:




