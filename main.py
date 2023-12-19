import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from helpers import makedir
import model
import train_and_test as tnt
import save
import matplotlib.pyplot as plt
import numpy as np
import dataloader_qd as dl
from settings import *
from matplotlib.pyplot import show

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

model_dir = './saved_models/' + data_name + '/'
makedir(model_dir)
prototype_dir = model_dir + 'prototypes/'
makedir(prototype_dir)


# all datasets
if (data_name == "cifar10"):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
     
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.CIFAR10(root=data_path, train=True,
                                download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_path, train=False,
                               download=True, transform=transform_test)      

elif (data_name == "mnist"):
    mean = (0.5)
    std = (0.5)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.MNIST(root=data_path, train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(root=data_path, train=False,
                             download=True, transform=transform)

elif (data_name == "fmnist"):
    mean = (0.5)
    std = (0.5)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    trainset = datasets.FashionMNIST(root=data_path, train=True,
                              download=True, transform=transform)
    testset = datasets.FashionMNIST(root=data_path, train=False,
                             download=True, transform=transform)

elif (data_name == "svhn"):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = datasets.SVHN(root=data_path, split="train",
                             download=True, transform=transform)
    testset = datasets.SVHN(root=data_path, split="test",
                            download=True, transform=transform)


elif (data_name == "quickdraw"):
    trainset = dl.QuickDraw(ncat=num_classes, mode='train', root_dir=data_path)
    testset = dl.QuickDraw(ncat=num_classes, mode='test', root_dir=data_path)
    
    
elif (data_name == "utk"):
    ##############
    import pickle
    import csv
    # Import compressed data
    data_loc = 'Data/UTK/utk_compress.pkl'
    target_loc = 'Data/UTK/sex.pkl'

    file = open(data_loc,'rb')
    X = pickle.load(file)
    file.close()
    
    # Load targets
    file = open(target_loc,'rb')
    y = pickle.load(file)
    file.close()   
    
    # Masking age
    file = open('Data/UTK/age.pkl','rb')
    age = pickle.load(file)
    file.close()    
    age_mask = np.where(age>18)[0]
    X = X[age_mask]
    y = y[age_mask]

    # Setting train indicies
    train_start = int(0)
    train_end = int(16000)    
        
    # Shuffling data
    from sklearn.utils import shuffle            
    X, y = shuffle(X, y, random_state=0)
 
    
    ##############         
    # Normalizing each image with its own mean and std
    X_norm = np.zeros([X.shape[0],3,32,32])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    for i in range(X.shape[0]):
        image = transform(X[i])
        mean, std = image.mean([1,2]), image.std([1,2])
   
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])       
        img_normalized = transform_norm(X[i])
        X_norm[i] = img_normalized
         
    # convert into PyTorch tensors
    #train/test split
    X_train = torch.tensor(X_norm[0:train_end], dtype=torch.float32)
    y_train = torch.tensor(y[0:train_end], dtype=torch.float32).long()
    
    X_test = torch.tensor(X_norm[train_end:], dtype=torch.float32)
    y_test = torch.tensor(y[train_end:], dtype=torch.float32).long()  

    trainset = list(zip(X_train,y_train))
    testset = list(zip(X_test,y_test))
    ############## 

    
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

test_loader_expl = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=num_workers)

    

print('data : ',data_name)
print('training set size: {0}'.format(len(train_loader.dataset)))
print('test set size: {0}'.format(len(test_loader.dataset)))

jet = False
if(data_name == "mnist" or data_name=="fmnist" or data_name=="quickdraw"):
    jet = True

# construct the model
protovae = model.ProtoVAE().to(device)

print('COEFS',coefs)

## Training
if(mode=="train"):
    print('start training')
    optimizer_specs = \
            [{'params': protovae.features.parameters(), 'lr': lr},
             {'params': protovae.prototype_vectors, 'lr':lr},
             {'params': protovae.decoder_layers.parameters(), 'lr':lr},
             {'params': protovae.last_layer.parameters(), 'lr':lr}
             ]
    optimizer = torch.optim.Adam(optimizer_specs)

    Volume_array = np.zeros([num_train_epochs]) # array for storing volumes
    Orth_array = np.zeros([num_train_epochs])     
    
    ood_epoch = [] # number of OOD prototypes pr epoch
    ood_epoch_pr_class = [] # number of OOD prototypes pr epoch pr class    
    v_factor = 1 # volume loss scaling factor    
    v_factor_init = 1 # the initial volume loss scaling saved to potentially downscale volume factor post elim
    
    DB_epoch =  [] #DB score for epoch
    acc_epoch = [] # Accuracy for epoch     
    n_protos_init = protovae.prototype_vectors.shape[0]  # Number of prototypes at initialization 
    

    is_elim=False # Set ellimination flag 
    for epoch in range(num_train_epochs):
        print('\nEPOCH: \t{0}'.format(epoch))

        train_acc, train_ce, train_recon, train_kl, train_vol,sim_scores,train_z,train_orth = tnt.train(is_elim=is_elim,v_factor=v_factor,model=protovae, dataloader=train_loader,optimizer=optimizer)
        
        
        # ==================================== PROTOTYPE QUALITY DIAGNOSTICS =====================================
        if epoch >= 0: # DB score
            
            protovae.eval()
            with torch.no_grad():            
  
                # Creating flat list of similarity scores
                flat_list = [item for sublist in sim_scores for item in sublist]
                sim_scores = [t.detach().numpy() for t in flat_list]
                sim_scores = np.array([sim_scores])[0]
            
                # Creating flat list of latent z
                flat_list = [item for sublist in train_z for item in sublist]
                z = [t.numpy() for t in flat_list]
                z = np.array([z])[0]
            
                # Collecting the prototypes
                prototypes = protovae.get_prototypes().detach().numpy()
        
                # Computing observation proximity to prototypes
                n_protos = len(prototypes)
                prox_vectors = np.zeros([len(sim_scores),n_protos])
                for i in range(len(sim_scores)):
                    prox_vectors[i] = np.where(sim_scores[i]==np.max(sim_scores[i]),1,0)
    
                # Finding prototype labels based on maximal similarity
                labels = []
                for i in range(len(sim_scores)):
                    labels.append(np.argmax(prox_vectors[i]))
     
                # Computing average intra prototype/cluster distance
                d_avg_intra_k = np.zeros([n_protos])
                n_assigned_proto_k = []
                for k in range(n_protos):                
                    prototype_k = prototypes[k] # The k'th proto     
                    idx_assigned_proto_k = np.where(np.array(labels)==k)[0]  # Index of observations assigned to proto k
                    n_assigned_proto_k.append(len(idx_assigned_proto_k)) # Number of assigned observations per protype
                    z_assigned_proto_k = z[idx_assigned_proto_k] # Latent observation assigned to prototypes               
                    
                    # Computing the intra cluster distance (cluster diameter as average of intra cluster dist)
                    d = torch.cdist(torch.tensor(z_assigned_proto_k),torch.tensor(prototype_k[None,:]), p=2)
                    d_avg_intra_k[k] = torch.mean(d)
            

                # List of indicies for out-of-distribution prototypes
                # Out of distribution defined by a prototype not having any observations closest to it
                ood_protos_idx = np.where(np.array(n_assigned_proto_k)==0)[0] 
                
                # OOD PR CLASS
                ood_pr_class = np.zeros(protovae.num_classes)
                for k in range(protovae.num_classes):
                    ood_pr_class[k] = sum(protovae.prototype_class_identity[ood_protos_idx,k])                    
                
                ood_epoch_pr_class.append(ood_pr_class)
            
                # Saving the number of OOD protos pr epoch
                ood_epoch.append(len(ood_protos_idx))
                                                
                # Now computing DB metric 
            
                # Computing the inter cluter distances
                d_inter_k = torch.cdist(torch.tensor(prototypes),torch.tensor(prototypes), p=2)

                # Setting self distance to zero, which does not always happen in cdist due to imprecision
                for k in range(n_protos):
                    d_inter_k[k,k]= 0
                
                # Finding all in-distribution prototypes:
                p_elim_idx = ood_protos_idx # Indicies of protos to eliminate
                n_protos_pt = protovae.prototype_vectors.shape[0] # Total number of protos currently in model
                p_remain_full = list(range(0,n_protos_pt)) # List of indicies for all current protos in model 
                p_remain_idx = list(np.delete(p_remain_full, p_elim_idx)) # List of indicies for in-distribution protos
                                
                # Checking diagnostics on in distribution prototypes                                    
                d_inter_k = d_inter_k[p_remain_idx]
                d_inter_k = d_inter_k[:,p_remain_idx]    
                d_avg_intra_k = d_avg_intra_k[p_remain_idx]     
                n_protos=len(p_remain_idx)
            
                # Computing the DB cluster similarity measure
                R_ij = np.zeros([n_protos,n_protos])

                for i in range(n_protos):
                    for j in range(n_protos):
                        if i==j:
                            R_ij[i,j] = 0
                        else:
                            R_ij[i,j] = (d_avg_intra_k[i] + d_avg_intra_k[j]) / d_inter_k[i,j]  #(s_i + s_j / d_ij)
       
                # Creating a list of the max cluster similarity scores for each cluster
                R_ij_max = []
                for k in range(n_protos):
                    R_ij_max.append(np.max(R_ij[k,:]))
                
                # Computing the DB score as the mean 
                DB_score = np.mean(R_ij_max)  
                DB_epoch.append(DB_score)                                        
        # ================================================================================================


        # ==================================== PROTOTYPE ELIMINATION =====================================
        # Optional Prototype elimination During training. Can enter this stage based on various conditions.
        # Example of condition for entering elimination stage: Number of OOD protos is stable across 2 or 3 consecutive epochs
#        if epoch > 5 and len(ood_protos_idx) > 0 and ood_epoch[epoch] == ood_epoch[epoch-1]:                  
#        if epoch%10 ==0: # for every 10th epoch. 
        if epoch == 120:       
            protovae.eval()
            with torch.no_grad():            
                is_elim = True
                print('\nENTERING PROTOTYPE ELIMINATION PHASE')
                p_elim_idx = ood_protos_idx # Indicies of protos to eliminate
                n_protos_pt = protovae.prototype_vectors.shape[0] # Total number of protos currently in model
                p_remain_full = list(range(0,n_protos_pt)) # List of indicies for all current protos in model 
                p_remain_idx = list(np.delete(p_remain_full, p_elim_idx)) # List of indicies for protos after elimination
                                                                          
                # Save protos before elim
                prototypes_pre_elim = protovae.prototype_vectors # Save protos before elimination 
                # Eliminating prototypes. optionally dont add to optimizer to lock the prototypes.
                protovae.prototype_vectors = torch.nn.parameter.Parameter(protovae.prototype_vectors[p_remain_idx])
                optimizer.add_param_group({'params': protovae.prototype_vectors}) # Adding to optimizer                
                n_protos_post = protovae.prototype_vectors.shape[0] # Total number of protos after elim
            
                # Keeping of track of prototype class belongings after elimination
                proto_class_pre_elim = protovae.proto_class           
                protovae.proto_class = protovae.proto_class[p_remain_idx]
            
                # One-hot encoded prototype class vectors with dimension [n_protos,n_classes]
                prototype_class_identity_pre_elim =  protovae.prototype_class_identity                
                protovae.prototype_class_identity =  protovae.prototype_class_identity[p_remain_idx]
                
               
                # The current number of prototypes for each class. Used for diagnostics during training
                protos_pr_class = np.zeros(protovae.num_classes)
                for k in range(protovae.num_classes):
                    protovae.protos_pr_class[k] = sum(protovae.prototype_class_identity[:,k])
                                       
                # Save last layer before elim
                last_layer_pre_elim = protovae.last_layer.weight # Save layer before elimination
                                       
                # Updating the nodes in the last layer according the current number of active prototypes. 
                protovae.last_layer.weight = torch.nn.parameter.Parameter(protovae.last_layer.weight[:,p_remain_idx])           
                optimizer.add_param_group({'params': protovae.last_layer.weight}) # Adding parameter to optimizer 
              
                                             
                # Printing diagnostics
                print(f'Protos per class after elimination: {protovae.protos_pr_class}')
                print(f'Total number of protos after elimination: {int(sum(protovae.protos_pr_class))}')
                
                # Optionally updating the loss penality on volume after elimination stage
#                v_factor = v_factor * (n_protos_post / n_protos_pt)  
#                v_factor = v_factor_init * (n_protos_post/n_protos_init)                    
                
                # Optionally Reduce learning rate after potential ellimination
#                optimizer_specs = \
#                        [{'params': protovae.features.parameters(), 'lr': lr/10},
#                         {'params': protovae.prototype_vectors, 'lr':lr/10},
#                         {'params': protovae.decoder_layers.parameters(), 'lr':lr/10},
#                         {'params': protovae.last_layer.parameters(), 'lr':lr/10}
#                         ]
#                optimizer = torch.optim.Adam(optimizer_specs)               
        # ================================================================================================
             
        # Running model on test data
        test_acc, test_ce, test_recon, test_kl, test_vol, sim_score,z_test,test_orth = tnt.test(is_elim=is_elim,v_factor=v_factor,model=protovae, dataloader=test_loader)
        
        acc_epoch.append(test_acc)

        Volume_array[epoch] = train_vol
        Orth_array[epoch] = train_orth     

    print("saving..")
    save.save_model_w_condition(model=protovae, model_dir=model_dir, model_name=str(epoch), accu=test_acc,
                                target_accu=0)
    
    # saving volumes, final prototypes and final decoded prototypes
    import pickle
    output = open('./saved_models/'+ data_name + '/' +'Volumes.pkl', 'wb')
    pickle.dump(Volume_array, output)
    output.close()
    
    output = open('./saved_models/'+ data_name + '/' +'DB_scores.pkl', 'wb')
    pickle.dump(DB_epoch, output)
    output.close()  
    
    output = open('./saved_models/'+ data_name + '/' +'acc.pkl', 'wb')
    pickle.dump(acc_epoch, output)
    output.close()  
    
    output = open('./saved_models/'+ data_name + '/' +'ood_epoch.pkl', 'wb')
    pickle.dump(ood_epoch, output)
    output.close()
    
    output = open('./saved_models/'+ data_name + '/' +'ood_epoch_pr_class.pkl', 'wb')
    pickle.dump(ood_epoch_pr_class, output)
    output.close()    
    
    output = open('./saved_models/'+ data_name + '/' +'Orths.pkl', 'wb')
    pickle.dump(Orth_array, output)
    output.close()         
    
    prototypes = protovae.get_prototypes()
    output = open('./saved_models/'+ data_name + '/' +'prototypes.pkl', 'wb')
    pickle.dump(prototypes, output)
    output.close()
    
    decoded_prototype_images = protovae.get_prototype_images()
    output = open('./saved_models/'+ data_name + '/' +'decoded_prototypes.pkl', 'wb')
    pickle.dump(decoded_prototype_images, output)
    output.close()

    ## Save and plot learned prototypes
    protovae.eval()
    prototype_images = protovae.get_prototype_images()
    prototype_images = (prototype_images + 1) / 2.0
    num_prototypes = len(prototype_images)
    num_p_per_class = protovae.num_prototypes_per_class
    

    plt.figure("Prototypes")
    for j in range(num_prototypes):
        p_img_j = prototype_images[j, :, :, :].detach().cpu().numpy()
        if(jet!=True):
            p_img_j = np.transpose(p_img_j, (1, 2, 0))
        else:
            p_img_j = np.squeeze(p_img_j)

        if(jet!=True):
            plt.imsave(os.path.join(prototype_dir, 'prototype' + str(j) + '.png'), p_img_j, vmin=0.0, vmax=1.0)
        else:
            plt.imsave(os.path.join(prototype_dir, 'prototype' + str(j) + '.png'), p_img_j,vmin=0.0, vmax=1.0) 

        plt.subplot(num_classes, num_p_per_class, j + 1)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.imshow(p_img_j)

    plt.show()
    print("Prototypes stored in: ", prototype_dir)

else:
    ## Testing
    protovae.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)
    protovae.eval()
    test_acc, test_ce, test_recon, test_kl, test_vol = tnt.test(v_factor=v_factor,model=protovae, dataloader=test_loader,is_elim=False)


    ## Save and plot learned prototypes
    prototype_images = protovae.get_prototype_images()
    prototype_images = (prototype_images + 1) / 2.0
    num_prototypes = len(prototype_images)
    num_p_per_class = protovae.num_prototypes_per_class
    
    plt.figure("Prototypes")
    for j in range(num_prototypes):
        p_img_j = prototype_images[j, :, :, :].detach().cpu().numpy()
        if (jet != True):
            p_img_j = np.transpose(p_img_j, (1, 2, 0))
        else:
            p_img_j = np.squeeze(p_img_j)

        plt.subplot(num_classes, num_p_per_class, j + 1)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.imshow(p_img_j)
    show()

    if(expl):
        ## Generate LRP based location explanation maps
        print("\n")
        print("Generating explanations")
        from prp import *

        prp_path = model_dir + 'prp/'
        prp_train_path = prp_path + 'train/'
        prp_test_path = prp_path + 'test/'
        orig_test_path = prp_path + 'test-orig/'


        wrapper = model_canonized()
        # construct the model for generating LRP based explanations
        model_wrapped = model.ProtoVAE().to(device)
        wrapper.copyfrommodel(model_wrapped.features, protovae.features, lrp_params=lrp_params_def1,
                              lrp_layer2method=lrp_layer2method)

        generate_explanations(test_loader_expl, model_wrapped.features,protovae.prototype_vectors, protovae.num_prototypes, prp_path, orig_test_path,protovae.epsilon)
