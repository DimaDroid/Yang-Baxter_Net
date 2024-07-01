import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from scipy.special import ellipj

from YBtrain import EarlyStopper, train_cycle
from YBnet import YangBaxterNet, mae_metric, generate_data
from YBtarget import compare_results, hamiltonian_target, hamiltonian_trained

f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()
del f

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Set up NN dtype
torch.set_default_dtype(torch.float64)

# Set up execution on GPU
if device != 'cpu':
    torch.set_default_dtype(torch.cuda.DoubleTensor)

# Fixed Parameters
input_dim, output_vars = 1, 12 # 1 for future generalisations; 12 = 6 complex entries of the R-matrix

n_pts = 1280
n_pts_val = int(n_pts * 0.1)

Hparams = {"eta": 0.5, "m": 0}

model_name = 'XXZ_swish'
activation='swish'

# Variable perameters
batch_size = 32

num_feat_layers = 16
num_neurons = 64

learning_rate = 1e-3

beta1 = 0.9 # optimiser parameter1
beta2 = 0.99 # optimiser parameter2

# Create datasets
train_data, val_data = generate_data(input_dim, n_pts, n_pts_val)

# Instantiate the model
ybnet = YangBaxterNet(model_name, input_dim, output_vars, device, lambda_hc=1, lambda_reg=1, lambda_mc=1, 
                    num_feat_layers=num_feat_layers, num_neurons=num_neurons, activation=activation, 
                    init_from_saved=False, PATH=None)

ybnet.to(device) # send to gpu if possible    

# Define optimizer
optimizer = optim.Adam(ybnet.parameters(), lr=learning_rate, betas=(beta1, beta1), fused=False) # fused option for GPU run
    
# Define 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, 
                                           threshold_mode='abs', cooldown=0, min_lr=1e-8, eps=1e-06)

# set up early stopper
early_stopper = EarlyStopper(patience=30, threshold=1e-4)

model = [ybnet, scheduler, early_stopper]
metric = mae_metric

# Train to the first point from cold
num_epochs = 250
stop_epoch = num_epochs
period = int((n_pts/batch_size)/5) # Period (No of batches) to avarage loss, default to output every 5 batches

train_cycle(num_epochs, model, metric, optimizer, train_data, val_data, n_pts, n_pts_val, batch_size, period, Hparams)

# Save weights
ybnet.save_model('XXZ_test')

plt.plot( np.arange(len(ybnet.loss_curve))* batch_size*period/n_pts, ybnet.loss_curve)
plt.plot( np.arange(len(ybnet.val_curve)), ybnet.val_curve)
plt.yscale('log')
plt.grid('both')

plt.savefig( 'XXZ_test' + '.png', bbox_inches='tight')
# plt.show()

# Target R-matrix
eta = 0.5
m = 0   

n_pts_compare = 128
u12 = np.expand_dims(np.linspace(-1.,1.,n_pts_compare), axis=-1)
u13 = np.expand_dims(np.linspace(-1.,1.,n_pts_compare), axis=-1) 

Ra_target, Rb_target, Rc_target, Rd_target = hamiltonian_target(u12, m, eta)
Ra_train, Rb_train, Rc_train, Rd_train = hamiltonian_trained(ybnet, u12, u13)

print(compare_results(np.real(Ra_train/Rc_train), np.reshape(Ra_target/Rc_target, n_pts_compare)))
print(compare_results(np.real(Rb_train/Rc_train), np.reshape(Rb_target/Rc_target, n_pts_compare)))
print(compare_results(np.real(Rd_train/Rc_train), np.reshape(Rd_target/Rc_target, n_pts_compare)))