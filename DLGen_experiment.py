# runs pseudoRev method for four UCI datasets (POWER, GAS, HEPMASS, MINIBOONE);
# USAGE:
#   python run_learning.py dataset1 dataset2 ...

from time import time
import sys
import learning as ln
import numpy as np
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device',device)

seed = 12345
torch.manual_seed(seed)
np.random.seed(seed)



# power dataset (6d)
def learn_model_power():
    # parameter for Neural network; 
    n_hiddens = 256  # neurons in each hidden layer
    learning_rate = 0.002  # learning rate
    batch_size = 5000   # size of batch
    n_iter = 20000   # number of iteration (epoch) to run

    ln.load_data('power') # load data
    # ln.train_DiffLabel(n_hiddens,learning_rate,batch_size,n_iter) # training
   
    t1=time()
    ln.train_DiffLabel_orbit(batch_size)
    t2=time()
    ln.train_DiffLabel_NN(n_hiddens,learning_rate,n_iter)
    t3=time()
    print('power')
    print('ODE time')
    print(t2-t1)
    print('NN training time')
    print(t3-t2)

# gas dataset (8d)
def learn_model_gas():
    n_hiddens = 512
    learning_rate = 0.002 
    batch_size = 5000   
    n_iter = 20000

    ln.load_data('gas')
    # ln.train_DiffLabel(n_hiddens,learning_rate,batch_size,n_iter)
   
    t1=time()
    ln.train_DiffLabel_orbit(batch_size)
    t2=time()
    ln.train_DiffLabel_NN(n_hiddens,learning_rate,n_iter)
    t3=time()
    print('gas')
    print('ODE time')
    print(t2-t1)
    print('NN training time')
    print(t3-t2)


# hepmass dataset (11d)
def learn_model_hepmass():
    n_hiddens = 1024
    learning_rate = 0.002 
    batch_size = 5000   
    n_iter = 20000

    ln.load_data('hepmass')
    # ln.train_DiffLabel(n_hiddens,learning_rate,batch_size,n_iter)
    
    t1=time()
    ln.train_DiffLabel_orbit(batch_size)
    t2=time()
    ln.train_DiffLabel_NN(n_hiddens,learning_rate,n_iter)
    t3=time()
    print('hepmass')
    print('ODE time')
    print(t2-t1)
    print('NN training time')
    print(t3-t2)



# miniboone dataset (43d)
def learn_model_miniboone():
    n_hiddens = 1400
    learning_rate = 0.002 
    batch_size = 2000   
    n_iter = 20000
    
    ln.load_data('miniboone')
    # ln.train_DiffLabel(n_hiddens,learning_rate,batch_size,n_iter)
    
    t1=time()
    ln.train_DiffLabel_orbit(batch_size)
    t2=time()
    ln.train_DiffLabel_NN(n_hiddens,learning_rate,n_iter)
    t3=time()
    print('miniboone')
    print('ODE time')
    print(t2-t1)
    print('NN training time')
    print(t3-t2)


# main 
def main():
    
    methods = dict()
    methods['power'] = learn_model_power
    methods['gas'] = learn_model_gas
    methods['hepmass'] = learn_model_hepmass
    methods['miniboone'] = learn_model_miniboone

    for name in sys.argv[1:]:
        methods[name]()


if __name__ == '__main__':
    
    main()