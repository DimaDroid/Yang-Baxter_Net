import torch
import torch.nn as nn

from time import time

from YBnet import ybe_loss, rtt_loss

# Early stopper to terminate learning on plateau

class EarlyStopper:
    def __init__(self, patience=10, threshold=0.0001):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss * (1-self.threshold)):
            self.min_validation_loss = validation_loss
            self.counter = 0
            print('new minimum found')
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            print('waiting patienntly ... {}'.format(self.counter))
            if self.counter >= self.patience:
                return True
        else:
            self.counter += 1
            print('waiting patienntly, but the new loss is almost thye same...')
            if self.counter >= self.patience:
                return True
        return False
    
def train_loop(model, metric, optimizer, x_train, x_zero, y_train, n_pts, batch_size, period, Hparams):    
    model.train()

    # set a permutation to sample batches from
    permutation = torch.randperm(n_pts)

    for i in range(0, n_pts, batch_size):
        indices = permutation[i:i+batch_size]
        x_batch, y_batch = x_train[indices].to(model.get_device()), y_train[indices].to(model.get_device())

        # Compute YBE and RTT losses
        outputs = model(x_batch)
        ybe_loss_val = ybe_loss(y_batch,outputs, metric)
        rtt_loss_val = rtt_loss(y_batch,outputs, metric)

        # Compute H_two and R at zero input
        outputs_zero = model(x_zero)
        h_two = model.compute_h_two(x_zero)

        # Compute HC, Reg and MC loses 
        hc_loss_val = model.hc_loss(h_two, metric)
        reg_loss_val = model.reg_loss(outputs_zero, metric)
        mc_loss_val = model.mc_loss(h_two, metric, **Hparams)

        loss = ybe_loss_val + rtt_loss_val + hc_loss_val + reg_loss_val + mc_loss_val

        # Backpropagation and optimization
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print statistics
        model.running_loss += loss.item()
        model.running_ybe_loss += ybe_loss_val.item()
        model.running_rtt_loss += rtt_loss_val.item()
        model.running_hc_loss += hc_loss_val.item()
        model.running_reg_loss += reg_loss_val.item()
        model.running_mc_loss += mc_loss_val.item()

        if (i/batch_size)%period == (period-1):    # every period (of mini-batches)
            model.loss_curve.append(model.running_loss / period)
            print('{prc:.0f}% -- loss: {rnloss:.2e};'.format(prc=i/n_pts*100, rnloss=model.running_loss / period), 
                  ' ybe-rtt loss: {ybeloss:.2e}-{rttloss:.2e}; hc loss: {hcloss:.2e};'.format(ybeloss=model.running_ybe_loss / period, rttloss=model.running_rtt_loss / period, hcloss=model.running_hc_loss / period),
                  ' reg loss: {regloss:.2e}; mc loss: {mcloss:.2e};'.format(regloss=model.running_reg_loss / period, mcloss=model.running_mc_loss / period))
            
            model.reset_losses()

def valid_loop(model, metric, x_val, x_zero, y_val, n_pts_val, batch_size, Hparams):
    
    model.eval()
    
    valid_loss = 0
    for i in range(0, n_pts_val, batch_size):
        x_batch, y_batch = x_val[i:i+batch_size].to(model.get_device()), y_val[i:i+batch_size].to(model.get_device())
        outputs = model(x_batch)
        
        # Compute YBE and RTT losses
        ybe_loss_val = ybe_loss(y_batch,outputs, metric)
        rtt_loss_val = rtt_loss(y_batch,outputs, metric)
        
        # Compute H_two and R at zero input
        outputs_zero = model(x_zero)
        h_two = model.compute_h_two(x_zero)
        
        # Compute HC, Reg and MC loses 
        hc_loss_val = model.hc_loss(h_two, metric)
        reg_loss_val = model.reg_loss(outputs_zero, metric)
        mc_loss_val = model.mc_loss(h_two, metric, **Hparams)
        
        # calculate loss
        loss = ybe_loss_val + rtt_loss_val + hc_loss_val + reg_loss_val + mc_loss_val
        
        # updata the validation loss
        valid_loss += loss
    
    # update validation curve
    model.val_curve.append(valid_loss.item())
    
    # To keep an eye on valudation loss
    ybe_loss_val2 = ybe_loss(y_batch,outputs, metric)
    rtt_loss_val2 = rtt_loss(y_batch,outputs, metric)
    print('ybe val loss: ', ybe_loss_val2.item(), '; rtt val loss: ', rtt_loss_val2.item())
    
    return valid_loss


def train_cycle(num_epochs, model, metric, optimizer, train_data, val_data, n_pts, n_pts_val, batch_size, period, Hparams):
    
    ybnet, scheduler, early_stopper = model[:]
    x_train, x_zero, y_train = train_data[:]
    x_val,        _, y_val = val_data[:]

    stop_epoch = num_epochs
    start = time()
    # Training loop
    for epoch in range(num_epochs):
        
        # Forward pass
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loop(ybnet, metric, optimizer, x_train, x_zero, y_train, n_pts, batch_size, period, Hparams)
        
        # validate the model
        valid_loss = valid_loop(ybnet, metric, x_val, x_zero, y_val, n_pts_val, batch_size, Hparams)

        # update lr if valid loss not decreases 
        if early_stopper.early_stop(valid_loss):       
            print('Stopping with validation loss={:.2e}'.format(valid_loss))
            stop_epoch = epoch
            break
        scheduler.step(valid_loss)
        print('Learning rate={:.1e}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        now = time()
        print('Time ellapsed: {:.2f}'.format(now - start))

    end = time()
    print('Time total:',(end - start))
    print('Time per epoch:',(end - start)/stop_epoch)
