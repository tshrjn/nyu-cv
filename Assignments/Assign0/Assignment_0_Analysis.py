
# coding: utf-8

# # Computer Vision CSCI-GA.2271-001 - Assignment 0
# Due date: Monday September 18th 2017
# Given on: September 7, 2017
# 
# ## 1 Introduction
# The purpose of this assignment is two-fold: (i) to ensure that you have the pre-requisite knowledge necessary for the rest of the course and (ii) to help you get experience with PyTorch, the language we will be using in the course.
# 
# To install PyTorch, follow the directions at http://pytorch.org. This also contains a number of tutorials that it might be helpful to work through.
# To turn in the assignment, please zip your code into a file named lastname firstname assign1.zip and email it to the grader Utku Evci (ue225@nyu.edu), cc’ing me (fergus@cs.nyu.edu).
# 

# In[1]:

import torch
import matplotlib.pyplot as plt
# import seaborn


# In[2]:

data_path = "assign0_data.py"


# ## 2 Whitening Data
# Pre-processing is an important part of computer vision and machine learning algorithms. One common approach is known as whitening in which the data is transformed so that it has zero mean and is decorrelated.
# 
# You should implement a PyTorch function that:
# * Load up the 2D dataset from the file assign1 data.py.
# * Visualize it by making a 2D scatter plot (e.g. using matplotlib).
# * Translates the data so that it has zero mean (i.e. is centered at the origin).
# * Decorrelates the data so that the data covariance is the identity matrix.
# * Plot the whitened data.
# * As a comment in your code, discuss the dependencies present in the whitened data.

# ### Helper Functions for plotting

# In[3]:

def plot_2d(tensor2d, title, limit, color=None, clubbed=False):
    '''
    A helper function to plot a 2-D Tensor with title and axes limits.
    '''

    plt.scatter(tensor2d[:,0].numpy(),tensor2d[:,1].numpy(), c=color)
    plt.title(title, fontsize=18)
    plt.xlim(-limit,limit)
    plt.ylim(-limit,limit)
    if not clubbed:
        plt.show()

def clubbed_plot(limit=None, *tensors_with_head):
    '''
    Function for plotting a single figure with multiple graphs
    '''
    
    legends = []
    p = []
    plt.xlim(-limit,limit)
    plt.ylim(-limit,limit)

    for tensor_with_head in tensors_with_head:
        x = tensor_with_head[0]
        legends.append(tensor_with_head[1])
        c = tensor_with_head[2]
        p.append(plt.scatter(x[:,0].numpy() , x[:,1].numpy(), c=c))
    plt.legend(p, legends, fontsize=12)
    plt.show()


# ### Required Function for Whitening

# In[4]:

def whitening_data(data_path):
    '''
    Input: data address
    
    Prints: Orginal 
    
    '''
    # 1. loading the data
    x = torch.load(data_path) 
    
    # using same limit for all plots to make visual comparisons easy
    limit = max(abs(x.min()), x.max())
    limit = int(limit) + int(limit**(0.5))
    
    # 2. Visualise data
    plot_2d(x, "Original Data", limit, 'brown') 
    
    # 3. zero-center the data
    X = x - x.mean(0).expand_as(x) 
    cov = torch.mm(X.t(), X) / X.size()[0] # get the data covariance matrix
    
    U, S, V = torch.Tensor(cov).svd()

    # 4. decorrelate the data
    Xrot = torch.mm(X, U)
    plot_2d(Xrot, "Decorrelated Data", limit, 'green')
    
    # 5. Whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    Xwhite = Xrot / (torch.sqrt(S + 1e-5).expand_as(Xrot))    
    # Note that we’re adding 1e-5 (or a small constant) to prevent division by zero
    plot_2d(Xwhite, "Whitened Data", limit, )   
        
    clubbed_plot(limit,(x, "Original Data", "brown"), (Xrot, "Decorrelated Data", "green"), (Xwhite, "Whitened Data", "blue"))

    print("Covaraince of whitened data:")
    print(torch.mm(Xwhite.t(), Xwhite) / Xwhite.size()[0])

    # 6. Dependencies present in the whitened data:
    '''
    Second-order dependencies have been removed by the whitening process. 
    However, higher order dependencies might still exist.
    The eigenvalues represent the variance in the data.
    '''


# In[5]:

whitening_data(data_path)


# ## 3 Fitting a 1D function with a simple neural net
# In PyTorch, generate the function y = cos(x) over the interval −π ≤ x ≤ π, at discrete intervals of 0.01. Adapting the examples from http://pytorch.org/tutorials/beginner/pytorch_with_examples.html, implement a neural net that regresses this function. I.e. takes as input x and produces an estimate yˆ, where ∥y − yˆ∥2 is minimized. The network should have a single hidden layer with 10 units and a single tanh() non-linearity. Make a plot of the true function y, with the network output yˆ before and after training overlaid (please use different colors for each).

# In[6]:

import numpy as np


# Create input and output data
step = 0.01
x = torch.arange(-np.pi, np.pi, step)
y = torch.cos(x)


# In[7]:

from torch.autograd import Variable

# N is batch size
N = 17*37

# Reshape input and output data according to the batch_size
x = Variable(x.view(N, -1))
y = Variable(y.view(N, -1), requires_grad=False)

print(x.size())

#  D_in is input dimension; H is hidden dimension; D_out is output dimension.
D_in, H, D_out = x.size()[1], 10, x.size()[1]

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)
loss_values = []

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    loss_values.append(loss.data.tolist()[0])
    # print(t, loss.data[t])

    if t ==0:
        y_before_training = y_pred

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
    
print("Final Loss", loss[-1])
plt.plot(loss_values, c='red')
plt.title("Loss", fontsize=18)
plt.ylabel('Loss Value')
plt.xlabel('Number of iterations')
plt.show()


# ### Plotting true, untrained & trained  value

# In[8]:

# Reshape to 1D
X = x.view(-1,)
Y_pred = y_pred.view(-1,)
Y_before_training = y_before_training.view(-1,)
Y = y.view(-1,)

s = [8]* len(X)

p_true = plt.scatter(X.data.tolist() , Y.data.tolist(), s=s)
p_pred = plt.scatter(X.data.tolist() , Y_pred.data.tolist(), s=s)
p_org = plt.scatter(X.data.tolist() , Y_before_training.data.tolist(), s=s)

plt.legend((p_true,p_pred, p_org),
           ('True Value', 'After Training', 'Before Training'), fontsize=12)
plt.show()

