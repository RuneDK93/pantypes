import time
import torch
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from settings import *
import numpy as np

def _train_or_test(is_elim,v_factor,model, optimizer=None, dataloader=None):
    
    print('Volume Factor:',v_factor)
    print('IS ELIM:',is_elim)
    sim_score_list = []
    z_list = []    
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_recons_loss = 0
    total_kl_loss = 0
    total_vol_loss = 0
    total_orth_loss = 0
    
    class_volume_loss_list = list(torch.zeros(model.num_classes))    
    

    for i, (image, label) in enumerate(dataloader):
        input = image.to(device)
        target = label.to(device)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, decoded, kl_loss, vol_loss, sim_scores, z, orth_loss,volume_loss_list = model(input, label, is_train,is_elim)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            recons = torch.nn.functional.mse_loss(decoded, input, reduction="mean")
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_recons_loss += recons.item()
            total_kl_loss += kl_loss.item()

            
            total_vol_loss += vol_loss.item()            
            
            total_orth_loss += orth_loss.item()                       
            sim_score_list.append(sim_scores)            
            z_list.append(z)                 
            class_volume_loss_list = [x + y for x, y in zip(class_volume_loss_list, volume_loss_list)]
            

        # compute gradient and do SGD step
        if is_train:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                      + coefs['recon'] * recons
                      + coefs['kl'] * kl_loss
                      + coefs['vol'] * vol_loss * v_factor
                      + coefs['orth'] * orth_loss                      
                        )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del decoded

    end = time.time()

    print('\ttime: \t{0}'.format(end -  start))
    print('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    print('\trecons: \t{0}'.format(total_recons_loss / n_batches))
    print('\tKL: \t{0}'.format(total_kl_loss / n_batches))
    print('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    print('\tVol Loss: \t\t{0}'.format(total_vol_loss / n_batches)) 
    print('\tOrth Loss: \t\t{0}'.format(total_orth_loss / n_batches))
    t = np.array(class_volume_loss_list) / n_batches
    # Optionally print volume loss pr class
#    print('--- Volume Loss ---')
#    for i in range(len(t)):
#        print(f'Class {i}: {t[i]}')
#    print('-------------------')      

    
    return n_correct / n_examples, total_cross_entropy/n_batches, total_recons_loss/n_batches, total_kl_loss/n_batches, total_vol_loss/n_batches, sim_score_list, z_list, total_orth_loss/n_batches 


def train(is_elim,v_factor,model, optimizer=None, dataloader=None):
    assert(optimizer is not None)
    
    print('\ttrain')
    model.train()
    return _train_or_test(is_elim,v_factor,model, optimizer=optimizer, dataloader=dataloader)


def test(is_elim,v_factor,model, optimizer=None, dataloader=None):
    print('\ttest')
    model.eval()
    return _train_or_test(is_elim,v_factor,model, optimizer=optimizer, dataloader=dataloader)


