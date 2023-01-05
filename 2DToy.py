import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

def f1(theta1,theta2,theta_star):
    return torch.sqrt((theta1 - theta_star[0])**2 + (theta2-theta_star[1])**2)

def f(theta,theta_star):
    return torch.linalg.norm(theta-theta_star,ord=2)

def g(theta,a,b):
    return torch.linalg.norm(torch.dot(a,theta)+b,ord=2)
def Toy2DExperiment(theta_init,theta_star,a,b,c,gamma=.1):
    alphas = [.1,1,10]
    betas = [.01,1,10]
    
    def dynamic_barrier_gradient_descent(alpha,beta):
        thetas = [theta_init]
        theta = thetas[0]
        for i in range(100):
            grad_f = grad(f(theta,theta_star),theta)[0]
            grad_g = grad(g(theta,a,b),theta)[0]

            phi = min(alpha * (g(theta,a,b)-c),beta * torch.linalg.norm(grad_g,ord=2))

            lam = max(0,(phi-torch.dot(grad_f,grad_g))/torch.linalg.norm(grad_g,ord=2))

            delta = grad_f+lam*grad_g

            theta.grad = None

            theta = theta - gamma * delta
            thetas.append(theta)
        return thetas

    # Experiment 1a
    x = torch.linspace(-3, 3, 100)
    X1, X2 = torch.meshgrid(x, x)
    Z1 = f1(X1,X2,theta_star)

    # Compute the halfspace
    Z2 = (c - a[0]*X1 - a[1]*X2 - b)
    
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Add the halfspace to the plot
    cs = ax.contour(X1, X2, Z2, levels=[0],colors=['b'])
    
    # Add the contour plot of f
    contours = plt.contour(X1, X2, Z1, 20)

    # Add labels and title
    plt.clabel(contours, inline=True, fontsize=10)
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.title(f'2D constrained optimization varying alpha with beta=1')

    for alpha in alphas:
        thetas = dynamic_barrier_gradient_descent(alpha,1)
        plt.plot([theta[0].item() for theta in thetas], [theta[1].item() for theta in thetas],'--',label=f"alpha={alpha}")
    plt.plot(theta_star[0],theta_star[1],'*')
    plt.colorbar()
    plt.legend(loc="lower left")
    plt.savefig('experiment_1a')

    ############################
    # Experiment 1b
    x = torch.linspace(-3, 3, 100)
    X1, X2 = torch.meshgrid(x, x)
    Z1 = f1(X1,X2,theta_star)

    # Compute the halfspace
    Z2 = (c - a[0]*X1 - a[1]*X2 - b)
    
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Add the halfspace to the plot
    cs = ax.contour(X1, X2, Z2, levels=[0],colors=['b'])
    
    # Add the contour plot of f
    contours = plt.contour(X1, X2, Z1, 20)

    # Add labels and title
    plt.clabel(contours, inline=True, fontsize=10)
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.title(f'2D constrained optimization varying beta with alpha=20')

    for beta in betas:
        thetas = dynamic_barrier_gradient_descent(20,beta)
        plt.plot([theta[0].item() for theta in thetas], [theta[1].item() for theta in thetas],'--',label=f"beta={beta}")
    plt.plot(theta_star[0],theta_star[1],'*')
    plt.colorbar()
    plt.legend(loc="lower left")
    plt.savefig('experiment_1b')
    #######################
    # Experiment 1d
    def dynamic_barrier_gradient_descent_with_optimizer():
        thetas = [theta_init]
        theta = thetas[0]
        for i in range(100):
            grad_f = grad(f(theta,theta_star),theta)[0]
            grad_g = grad(g(theta,a,b),theta)[0]

            phi = min(alpha * (g(theta,a,b)-c),beta * torch.linalg.norm(grad_g,ord=2))

            lam = max(0,(phi-torch.dot(grad_f,grad_g))/torch.linalg.norm(grad_g,ord=2))

            delta = grad_f+lam*grad_g

            theta = theta - gamma * delta
            thetas.append(theta)
        return thetas





torch.manual_seed(993)

theta = torch.randn(2,requires_grad=True)-1
theta_star = torch.randn(2)

a = torch.randn(2)
b = torch.randn(1)

Toy2DExperiment(theta,theta_star,a,b,c=.01)
