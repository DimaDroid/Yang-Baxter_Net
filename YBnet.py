import torch
import torch.nn as nn
import torch.autograd.functional as F

import numpy as np

from scipy.special import ellipj

# Set up default NN dtype
torch.set_default_dtype(torch.float32)

# Sin activation function
class activ_Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# Define metrics
loss_MSE_metric = nn.MSELoss(reduction = 'mean')
loss_MAE_metric = nn.L1Loss(reduction = 'mean')

def mse_metric(inp):
    # mse loss
    abs_val = torch.abs(inp)
    return loss_MSE_metric(abs_val, torch.zeros_like(abs_val))

def max_metric(inp):
    # abs, max along all non-batch dims and sum
    max_range = tuple(range(len(inp.shape))[1:])
    return torch.mean(torch.amax(torch.abs(inp), max_range))

def mae_metric(inp):
    # mse loss
    abs_val = torch.abs(inp)
    return loss_MAE_metric(abs_val, torch.zeros_like(abs_val))

# R-matrix functions

def to_R_matrix(x):
    ''' 
    1. reshape the input features into 4x4x1.
    2. add offset
    3. reshape into (2,2,2,2) to evaluate Yang Baxter equation
    '''
    x = add_offset(r_matrix_seed(x))
    x = torch.reshape(x,(-1,2,2,2,2)) #Reshape((2,2,2,2),input_shape=(4,4,1))(x)
    return x

def add_offset(inputs):
    '''careful that this offset is different from previous versions. the
    12 and 21 entries are zero here.'''
    offset_re = torch.tensor([[1.,0.,0.,0.],
                              [0.,0.,1.,0.],
                              [0.,1.,0.,0.],
                              [0.,0.,0.,1.]])
    offset_im = torch.tensor([[0.,0.,0.,0.],
                              [0.,0.,0.,0.],
                              [0.,0.,0.,0.],
                              [0.,0.,0.,0.]])
    offset = torch.complex(offset_re,offset_im)
    offset = torch.unsqueeze(offset,-1)
    return offset+inputs

def ybe(inputs, metric):
    R12,R13,R23 = inputs 
    
    lhs = torch.einsum('aijkl,akmno,alopq->aijmnpq', R12, R13, R23)
    rhs = torch.einsum('aijkl,amlno,ankpq->amijpqo', R23, R13, R12)
    
    ybe_value = metric(lhs-rhs)
    
    return ybe_value

def rtt(inputs, metric):
    R12,R13,R23 = inputs 
    
    monodromy1 = torch.einsum('aijkl,akmno,anpqr->aipmjqrol', R13, R13, R13)
    monodromy2 = torch.einsum('aijkl,akmno,anpqr->aipmjqrol', R23, R23, R23)
    
    lhs = torch.einsum('aijkl,akmnopqrs,alqrstuvw->aijmnoptuvw', R12, monodromy1, monodromy2)
    rhs = torch.einsum('aijklmnop,aqnoprstu,armvw->aqijklvwstu', monodromy2, monodromy1, R12)
    
    rtt_value = metric(lhs-rhs)
    
    return rtt_value

def ybe_loss(y,y_pred, metric):
    ''' expects the y_pred in the shape (3,batch,output_vars).
    The 3 is for the u12,u23,u13 direction. The 'true' output y
    is a dummy argument which is ignored.'''
        
    f12, f13, f23 = torch.unbind(y_pred)
    r12, r13, r23 = [to_R_matrix(f) for f in (f12,f13,f23)]
    
    loss = ybe([r12,r13,r23], metric)

    return loss

def rtt_loss(y, y_pred, metric):
    ''' expects the y_pred in the shape (3,batch,output_vars).
    The 3 is for the u12,u23,u13 direction. The 'true' output y
    is a dummy argument which is ignored.'''
        
    f12, f13, f23 = torch.unbind(y_pred)
    r12, r13, r23 = [to_R_matrix(f) for f in (f12,f13,f23)]
    
    loss = rtt([r12,r13,r23], metric)

    return loss

def r_matrix_seed(inputs):
    '''Expected what shape???'''
    
    layer = torch.complex(inputs, torch.zeros_like(inputs))
    layer = torch.unsqueeze(layer,-1)
    
    torch.tensor([1, 2], dtype=torch.float32)
    
    real = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    imag = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    zero = torch.complex(real, imag)
    
    real = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.])
    imag = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    one = torch.complex(real, imag)
    
    real = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    imag = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    row00 = torch.complex(real, imag)
    
    real = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    imag = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    row11 = torch.complex(real, imag)
    
    real = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    imag = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    row12 = torch.complex(real, imag)
    
    real = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    imag = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
    row21 = torch.complex(real, imag)
    
    real = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    imag = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
    row22 = torch.complex(real, imag)
    
    real = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
    imag = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    row33 = torch.complex(real, imag)
      
    mat1 = torch.stack([row00,zero,zero,zero])
    mat2 = torch.stack([zero,row11,row12,zero])
    mat3 = torch.stack([zero,row21,row22,zero])
    mat4 = torch.stack([zero,zero,zero,row33])
    
    col1, col2 = mat1 @ layer, mat2 @ layer
    col3, col4 = mat3 @ layer, mat4 @ layer
    
    out = torch.stack([col1,col2,col3,col4],dim=1)
    return out

def batch_diagonal(inp):
    '''works on tensor of shape (mb,:,:,mb,:,:)'''
    
    tensor = torch.permute(inp, (0,3,1,2,4,5))
    tensor = torch.diagonal(tensor, offset=0, dim1=0, dim2=1)
    tensor = torch.permute(tensor, (4,0,1,2,3))
    return tensor

def generate_data(input_dim, n_pts, n_pts_val):
    '''input_dim - input dim of the model
    n_pts - number of points for u variables for training
    n_pts_val - number of points for u variables for validation
    Generates train and valudation datasets (inputs x=u12, u13) by drawing from uniform distribution [-1,1].
    Outputs y are dummy.'''

    u12_zero = torch.zeros(1,input_dim)
    u13_zero = torch.zeros(1,input_dim)
    x_zero = torch.stack([u12_zero,u13_zero],dim=1)

    # Train and Validation data
    u12 = 2 * torch.rand(n_pts,input_dim) - 1.
    u13 = 2 * torch.rand(n_pts,input_dim) - 1.
    x_train = torch.stack([u12,u13],dim=1)
    y_train = torch.zeros(n_pts)

    train_data = [x_train, x_zero, y_train]

    u12_val = 2 * torch.rand(n_pts_val,input_dim) - 1.
    u13_val = 2 * torch.rand(n_pts_val,input_dim) - 1.
    x_val = torch.stack([u12,u13],dim=1)
    y_val = torch.zeros(n_pts_val)

    val_data = [x_val, x_zero, y_val]

    return train_data, val_data

# YBNet
class RVariableLayer(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, init_from_saved, activation):
        super(RVariableLayer, self).__init__()
        
        if activation == 'ELU':
            self.activation = nn.ELU() 
        elif activation == 'sin':
            self.activation = activ_Sin()
        elif activation == 'swish':
            self.activation = nn.SiLU() 
        elif activation == 'tanh': 
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else: 
            self.activation = nn.ReLU()
        
        self.linear_stack = nn.Sequential()
        self.linear_stack.append(nn.Linear(input_dim, n_neurons))
        for i in range(n_layers):
            self.linear_stack.append(nn.Linear(n_neurons, n_neurons))
            self.linear_stack.append(self.activation)
        # Last linear layer
        self.linear_stack.append(nn.Linear(n_neurons, 1))
        
        # Initialise weights
        if not init_from_saved:
            self.linear_stack.apply(self.weights_init_rule)
        
    
    def weights_init_rule(self, module):
        # A Xavier initialization
        bias_variance = 0.001
        if isinstance(module, nn.Linear):
            # # Normal initialisation
            # module.weight.data.normal_(mean=0.0, std=.22)
            # if module.bias is not None:
            #     module.bias.data.uniform_(-bias_variance,bias_variance)
            
            gain = 1.
            if self.activation == nn.ELU():
                gain = 1.41 # not sure, need more study
            elif self.activation == activ_Sin():
                gain = 1 # not sure, need more study
            elif self.activation == nn.SiLU(): # swish
                gain = 1 # not sure, need more study
            elif self.activation == nn.Tanh():
                gain = 1.66
            elif self.activation == nn.Sigmoid():
                gain = 1
            else: 
                gain = 1.41 # for ReLU

            # # Xavier initialisation
            torch.nn.init.xavier_uniform_(module.weight, gain)
            if module.bias is not None:
                module.bias.data.uniform_(-bias_variance,bias_variance)
            
            # # He initialisation
            # torch.nn.init.kaiming_normal_(module.weight, a=0)
            # if module.bias is not None:
            #     module.bias.data.uniform_(-bias_variance,bias_variance)
    
    def forward(self, inputs):
        output = self.linear_stack(inputs)
        return output

    
class YangBaxterNet(nn.Module):
    def __init__(self, model_name, input_dim, output_dim, device, lambda_hc, lambda_reg, lambda_mc, num_feat_layers, num_neurons,
                 activation='swish', init_from_saved=False, PATH=None):
        super(YangBaxterNet, self).__init__()
        self.model_name = model_name
        self.lambda_hc = lambda_hc
        self.lambda_reg = lambda_reg
        self.lambda_mc = lambda_mc
        
        # running losses for training
        self.running_loss = 0.0
        self.running_ybe_loss = 0.0
        self.running_rtt_loss = 0.0
        self.running_reg_loss = 0.0
        self.running_hc_loss = 0.0
        self.running_mc_loss = 0.0
        
        # Loss curve to plot after training
        self.loss_curve = []
        self.val_curve = []
        
        self.device = device # device where the model is stored
        
        self.h_two = torch.empty((4,4))
        self.rc_history = [] # ideally this should be in a callback or metric

        self.block = nn.ModuleList([RVariableLayer(input_dim, num_feat_layers, num_neurons, init_from_saved, activation) for _ in range(output_dim)])
        
        if init_from_saved:
            self.model = torch.load(PATH)
            print('Model loaded from: ',PATH)
        
    def get_device(self):
        return self.device
    
    def reset_losses(self):
        self.running_loss = 0.0
        self.running_ybe_loss = 0.0
        self.running_rtt_loss = 0.0
        self.running_reg_loss = 0.0
        self.running_hc_loss = 0.0
        self.running_mc_loss = 0.0
        
    def reset_curves(self):
        self.loss_curve = []
        self.val_curve = []
        
    def save_model(self, filename=None):
        if filename==None:
            PATH = self.model_name + '.pt'
        else:
            PATH = filename + '.pt'
        torch.save(self.state_dict(), PATH)
        print('Model saved to: ',PATH)
    
    def call_r(self,inputs):
        z = inputs
        output_layer = []
        for layer in self.block:
            output_layer.append(layer(z))
        
        outputs = torch.cat(output_layer, dim=1)
        
        return outputs
    
    def forward(self, inputs):
        ''' processes input to output. note that:
        computes the RC loss in the following steps:
        1. y_pred.shape = (3,batch,output_vars), 
        2. x.shape = (batch,3,input_dim)
        The 3 is for the u12,u23,u13 direction.
        '''
        u12, u13 = torch.unbind(inputs,dim=1)
        o12,o13,o23 = [self.call_r(u) for u in (u12,u13,u13-u12)]
        outputs = torch.stack([o12, o13, o23], dim=0)
        return outputs
    
    def compute_h_two(self, u_inp):
        '''the formula for the two-particle hamiltonian by differentiating the 
        R matrix, now across the minibatch.'''
        
        u_inp.requires_grad_()
        
        def R12(inp):
            preR12, _, _ = torch.unbind(self.forward(inp),dim=0)
            R12 = to_R_matrix(preR12)
            R12 = torch.reshape(R12,(-1,4,4))
            return R12

        def R12_re(inp):
            return R12(inp).real
        
        def R12_im(inp):
            return R12(inp).imag

        dR12_re = F.jacobian(R12_re, u_inp, vectorize=True, strategy='forward-mode') # shape (mb,4,4,mb,2,1)
        dR12_im = F.jacobian(R12_im, u_inp, vectorize=True, strategy='forward-mode') # shape (mb,4,4,mb,2,1)
        
        dR12 = torch.complex(dR12_re,dR12_im)
        dR12 = batch_diagonal(dR12)

        dR12 = dR12[:,:,:,0,0] # last dim is dummy, dim=-1 is for deriv wrt u12 and u13, we need only u12 deriv ->  shape (mb,4,4)
        
        # generate Pmat - batched permutation matrix
        Pmat_re = torch.tensor([[1.,0.,0.,0.],
                              [0.,0.,1.,0.],
                              [0.,1.,0.,0.],
                              [0.,0.,0.,1.]])
        Pmat_im = torch.tensor([[0.,0.,0.,0.],
                              [0.,0.,0.,0.],
                              [0.,0.,0.,0.],
                              [0.,0.,0.,0.]])
        Pmat = torch.complex(Pmat_re, Pmat_im)
        Pmat = torch.broadcast_to(Pmat, (dR12.size()[0],) + Pmat.size()) # (dR12.size()[0],) to generate batched dim

        htwo = Pmat @ dR12
        
        return htwo
    
  
    def hc_loss(self, h_two, metric):
        '''Comutes loss based on Hermiticity property of Hamiltonian.''' 
        
        h_two_c = h_two.adjoint()
        
        loss = self.lambda_hc * metric(h_two - h_two_c)
        
        return loss
    
    def reg_loss(self, y_zero, metric):
        '''Computes regularity loss using the fact that
            R-matrix at 0 is equal to permutation matrix P'''
        
        preR12, _, _ = torch.unbind(y_zero)
        R12 = to_R_matrix(preR12)
        R12 = torch.reshape(R12,(-1,4,4))
        
        P = torch.tensor([[[1., 0., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 0., 1.]]])
        
        loss = self.lambda_reg * metric(R12 - P)
        
        return loss
        
    def mc_loss(self,h_two, metric, **Hparams):
        # Collapse to the XYZ Hamiltonian
        #eta, m = *params
        eta = Hparams["eta"]
        m = Hparams["m"]
        sn, cn, dn, _ = ellipj(2 * eta, m)
        sqrt_m = np.sqrt(m)
        Jx = 1 + sqrt_m * sn / 2
        Jy = 1 - sqrt_m * sn / 2
        Jz = cn * dn
        
        target_hamilt = torch.tensor([[[Jz,    0.,  0.,  Jx-Jy],
                                       [0.,    -Jz, 2,   0.   ],
                                       [0.,    2,   -Jz, 0.   ],
                                       [Jx-Jy, 0.,  0.,  Jz   ]]])
        target_hamilt = torch.unsqueeze(target_hamilt,axis=0)
        #penalize_zero = lambda x : tf.exp(-tf.abs(x))
        #make_equal = lambda x,y : torch.mean(torch.abs(x-y))
    
        mcloss = self.lambda_mc * metric(target_hamilt - h_two)

        return mcloss