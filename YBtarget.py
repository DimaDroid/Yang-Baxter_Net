import torch
import numpy as np
from scipy.special import ellipj

from YBnet import add_offset, r_matrix_seed

# function to compare results
def compare_results(pred_vals, true_vals):
    
    diff = np.abs(pred_vals - true_vals)
    max_diff = np.max(diff)
    av_diff = np.mean(diff)
    
    return max_diff, av_diff

# generate elements of target R-matrix
def hamiltonian_target(u,m,eta):
    
    sn, cn, dn, _ = ellipj(2 * eta, m)
    w = 2 * sn 
    
    sn1, cn1, dn1, _ = ellipj(2 * eta + w*u, m)
    sn2, cn2, dn2, _ = ellipj(w*u, m)
    
    exp = np.exp(- cn * dn / (2 * sn) * w * u)
    
    a = sn1/sn * exp
    b = sn2/sn * exp
    c = exp
    d = np.sqrt(m) * sn2 * sn1 * exp 
    
    return a, b, c, d

def hamiltonian_trained(model, u12, u13):
    u12 = torch.from_numpy(u12.astype(np.float64))
    u13 = torch.from_numpy(u13.astype(np.float64))

    u = torch.stack([u12,u13],dim=1).to(model.device)

    model.eval()
    R_stack = model(u)

    preR_u12, _, _ = torch.unbind(R_stack)

    R_u12 = add_offset(r_matrix_seed(preR_u12))
    R_u12 = torch.squeeze(R_u12).cpu().detach().numpy()

    return R_u12[:,0,0], R_u12[:,1,1], R_u12[:,1,2], R_u12[:,3,0]
