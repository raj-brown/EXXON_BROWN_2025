import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



class PINN(nn.Module):
    def __init__(self, input_size, hideen_size, output_size):
        super(PINN, self).__init__()
        c = nn.Parameters(torch.tensor(1.0))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation_1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation_2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_1(x)
        x = self.fc_2(x)
        x = self.activation_2(x)
        x = self.fc3(x)
        return x

def residual_loss(model, x_f):
    pass


def data_loss(model, x_d):
    pass


def generate_data():
    pass


def generate_collocation_points(radius, num_samples):
    r = radius * np.sqrt(np.random.rand(num_samples))
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def generate_boundary_points(radius, num_samples):
    theta = np.linspace(0, 2 * np.pi, num_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def generate_validation_data(radius, num_samples):
    pass


def plot_solution(u, x, y):
    pass



def prediction(model, x, y):
    pass


def load_model():
    pass

def save_model():
    pass


if __name__=="__main__":
    r =1.0
    n_col = 1000
    x_f, y_f = generate_collocation_points(r, n_col)
    plt.scatter(x_f, y_f)
    x_b, y_b = generate_boundary_points(r, n_col)
    plt.plot(x_b, y_b, "-r", lw=1.5)
    plt.tight_layout()
    plt.axis("equal")
    plt.show()
    print(f"In main function :PINN\n")
    nIters = 100

    for it in range(0, nIters):
        pass

    
