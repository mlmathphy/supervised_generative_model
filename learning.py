# train density estimators on various datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import os
import datasets
from time import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# set paths
root_output = 'output/'   # where to save trained models
root_orbit = 'orbit/'      # where to save orbits
root_data = 'data/'       # where the datasets are
savedir = 'train_result/'

# holders for the datasets
data = None
data_name = None


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)



def load_data(name):
    """
    Loads the dataset. Has to be called before anything else.
    :param name: string, the dataset's name
    """

    assert isinstance(name, str), 'Name must be a string'
    datasets.root = root_data
    global data, data_name

    if data_name == name:
        return

    if name == 'power':
        data = datasets.POWER()
        data_name = name

    elif name == 'gas':
        data = datasets.GAS()
        data_name = name

    elif name == 'hepmass':
        data = datasets.HEPMASS()
        data_name = name

    elif name == 'miniboone':
        data = datasets.MINIBOONE()
        data_name = name

    elif name == 'bsds300':
        data = datasets.BSDS300()
        data_name = name

    else:
        raise ValueError('Unknown dataset')


def is_data_loaded():
    """
    Checks whether a dataset has been loaded.
    :return: boolean
    """
    return data_name is not None


def create_model_id(model_name):
    """
    Creates an identifier for the provided model description.
    """

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    delim = '_'
    id = data_name + delim + model_name
    return id



class NF_Net(nn.Module):

    def __init__(self, n_hiddens, dim):

        super(NF_Net, self).__init__()

        self.hid_size = n_hiddens
        self.dim = dim  # dimension of input parameter
     
        self.input = nn.Linear(self.dim, self.hid_size)
        self.fc1= nn.Linear(self.hid_size,self.hid_size)
        # self.fc2= nn.Linear(self.hid_size,self.hid_size)
        # self.fc3= nn.Linear(self.hid_size,self.hid_size)
        self.output = nn.Linear(self.hid_size,self.dim)

    def forward(self, x):

        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        x = self.output(x)
        return x
    
    def loss(self, x, y):

        loss = torch.mean(torch.sum((x - y)**2,dim=1))
        return loss



scalar = 0.99
def cond_alpha(t):
    return 1 - scalar*t

def cond_beta(t):
    return (1-scalar) + t*scalar

def b(t):
    # f=d_(log_alpha)/dt
    alpha_t = cond_alpha(t)
    f_t = -scalar/(alpha_t)
    return f_t

def sigma_sq(t):
    dsigma2_dt = scalar
    b_t = b(t)
    beta_t = cond_beta(t)
    sigma2 = dsigma2_dt - 2*b_t*beta_t
    return sigma2

def sigma(t):
    return torch.sqrt(sigma_sq(t))

T = 1
def inverse_SDE(X_T, x0, T, time_steps):
    # Generate the time mesh
    dt = T/time_steps
    t_vec_inv = torch.linspace(T, 0, steps=time_steps+1)

    # Initialization
    xt=X_T
    path_all=xt

    batchsize_ode = 1000
    row_x0 = torch.randint(0, x0.size(dim=0), (batchsize_ode,))

    for i in range(time_steps):
        t = t_vec_inv[i]
        # print(xt[:,None,:].size())
        # print(x0[None,row_x0,:].size())
        difference = -1.0 * (xt[:,None,:] - cond_alpha(t)*x0[None,row_x0,:])/cond_beta(t)
        # print(difference.size())

        # Compute the weights
        tmp_diff = (xt[:,None,:] - cond_alpha(t)*x0[None,row_x0,:])**2/cond_beta(t)
        P_xt_x0 = torch.exp(-0.5*torch.sum(tmp_diff,axis=2)) 
        weight = P_xt_x0 / torch.sum(P_xt_x0, axis=1,keepdims=True)
        score = torch.sum(torch.permute(torch.permute(difference,(2,0,1)) * weight,(1,2,0)), axis=1)

        # Evaluate the drift term
        drift = b(t)*xt - 0.5 * sigma_sq(t) * score
        xt = xt - dt*drift #+ np.sqrt(dt)*diffuse*Z
        path_all=xt
        # print(i)
    return path_all, t_vec_inv



def train_DiffLabel(n_hiddens,learning_rate,batch_path,n_iter):
    """
    Train Diffusion Labeling generative model
    """
    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    fig, axs = plt.subplots(3,6, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(0, 6):
        axs[i].hist(data.trn.x[:,i], bins=50)
        axs[i].set_title(f'Dataset {i+1}')

    # define inverse ODE
    time_steps = 500
    print(data.trn.N)
    print(data.n_dims)
    X_T=torch.randn(data.trn.N,data.n_dims).to(device)
    x_sample = torch.tensor(data.trn.x, dtype=torch.float32).to(device)  
    # path_inv,_ = inverse_SDE(X_T, x_sample, T, time_steps)

    path_inv = torch.zeros_like(X_T)
    num_batch = np.ceil(data.trn.N/batch_path).astype(int)
    for jj in range(num_batch):
        print(jj)
        if jj<num_batch-1:
            path_inv[(jj-1)*batch_path:jj*batch_path,:],_ = inverse_SDE(X_T[(jj-1)*batch_path:jj*batch_path,:], x_sample, T, time_steps)
        else:
            path_inv[(jj-1)*batch_path:,:],_ = inverse_SDE(X_T[(jj-1)*batch_path:,:], x_sample, T, time_steps)


    X_0 = path_inv.cpu().detach().numpy()
    y_train = path_inv

    for i in range(6, 12):
        axs[i].hist(X_0[:,i-6], bins=50)
        axs[i].set_title(f'ODE {i+1-6}')

    # define NN
    NF = NF_Net(n_hiddens, data.n_dims).to(device)
    NF.zero_grad()
    optimizer = optim.Adam(NF.parameters(), lr=learning_rate)

    for i in range(n_iter):

        if i > 200:
            optimizer.param_groups[0]["lr"] = learning_rate/2.0
        elif i > 400:
            optimizer.param_groups[0]["lr"] = learning_rate/4.0

        optimizer.zero_grad()
        y_pred = NF(X_T)
        loss = NF.loss(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if i%50==0:
            print(i, loss)

    ntest = 20000
    test_sample = torch.randn(ntest,data.n_dims).to(device)
    test_pred = NF(test_sample).cpu().detach().numpy()
    for i in range(12, 18):
        axs[i].hist(test_pred[:,i-12], bins=50)
        axs[i].set_title(f'pred {i+1-12}')

    plt.tight_layout()
    plt.savefig('result.png')


def train_DiffLabel_orbit(batch_path):
    """
    Train Diffusion Labeling generative model
    """
    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    print(np.mean(data.trn.x,axis=0))
    print(np.std(data.trn.x,axis=0))
    print(data.trn.x.shape)


    fig, axs = plt.subplots(2,6, figsize=(10, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i in range(0, 6):
        axs[i].hist(data.trn.x[:,i], bins=50)
        axs[i].set_title(f'Dataset {i+1}')

    # define inverse ODE
    time_steps = 500
    print(data.trn.N)
    print(data.n_dims)

    if data.trn.N >=100000:
        num_XT = data.trn.N
    else:
        num_XT = 200000
        
    X_T=torch.randn(num_XT,data.n_dims).to(device)
    x_sample = torch.tensor(data.trn.x, dtype=torch.float32).to(device)  


    path_inv = torch.zeros_like(X_T)
    num_batch = np.ceil(num_XT/batch_path).astype(int)
    for jj in range(num_batch):
        print(jj)
        if jj<num_batch-2:
            path_inv[jj*batch_path:(jj+1)*batch_path,:],_ = inverse_SDE(X_T[jj*batch_path:(jj+1)*batch_path,:], x_sample, T, time_steps)
        else:
            path_inv[jj*batch_path:,:],_ = inverse_SDE(X_T[jj*batch_path:,:], x_sample, T, time_steps)

    X_0 = path_inv.cpu().detach().numpy()

    for i in range(6, 12):
        axs[i].hist(X_0[:,i-6], bins=50)
        axs[i].set_title(f'ODE {i+1-6}')

    plt.tight_layout()
    plt.savefig('result' + data_name+ '.png')

    X_T_save = X_T.cpu().detach().numpy()
    make_folder(root_orbit)
    with open(root_orbit + data_name+'.npy', 'wb') as f:
        np.save(f, data.trn.x)
        np.save(f, X_T_save)
        np.save(f, X_0)




def train_DiffLabel_NN(n_hiddens,learning_rate,n_iter):
    """
    Train Diffusion Labeling generative model
    """
    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    fig, axs = plt.subplots(2,6, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

  
    with open(root_orbit + data_name+'.npy', 'rb') as f:
        x_sample = np.load(f)
        X_T = np.load(f)
        X_0 = np.load(f)

    # x_sample = torch.from_numpy(x_sample).to(device)
    x_train = torch.from_numpy(X_T).to(device)
    y_train = torch.from_numpy(X_0).to(device)

    print(x_train.size())
    print(y_train.size())

    print(torch.min(torch.std(y_train, dim=0)))
    print(torch.max(torch.std(y_train, dim=0)))

    for i in range(0, 6):
        axs[i].hist(X_0[:,i], bins=50)
        axs[i].set_title(f'ODE {i+1}')

    # define NN
    NF = NF_Net(n_hiddens, data.n_dims).to(device)
    NF.zero_grad()
    optimizer = optim.Adam(NF.parameters(), lr=learning_rate)

    for i in range(n_iter):

        if i > 5000:
            optimizer.param_groups[0]["lr"] = learning_rate/2.0
        if i > 10000:
            optimizer.param_groups[0]["lr"] = learning_rate/4.0
        if i > 12000:
            optimizer.param_groups[0]["lr"] = learning_rate/8.0

        optimizer.zero_grad()
        y_pred = NF(x_train)
        loss = NF.loss(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if i%50==0:
            print(i, loss)

            ntest = 10000
            test_sample = torch.randn(ntest,data.n_dims).to(device)
            test_pred = NF(test_sample).cpu().detach().numpy()

            print(np.min(np.mean(test_pred,axis=0) ))
            print(np.max(np.mean(test_pred,axis=0) ))
            print(np.min(np.std(test_pred,axis=0) ))
            print(np.max(np.std(test_pred,axis=0) ))


    t1=time()
    ntest = 100000
    test_sample = torch.randn(ntest,data.n_dims).to(device)
    test_pred = NF(test_sample).cpu().detach().numpy()
    t2=time()

    for i in range(6, 12):
        axs[i].hist(test_pred[:,i-6], bins=50)
        axs[i].set_title(f'pred {i+1-6}')
    print('sampling time')
    print(t2-t1)
    plt.tight_layout()
    plt.savefig('result_nn_' + data_name+ '.png')

    with open(root_orbit + data_name+'_nn.npy', 'wb') as f:
        np.save(f, test_pred)



