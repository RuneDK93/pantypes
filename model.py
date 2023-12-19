import torch
import torch.nn as nn
from settings import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from settings import *
import numpy as np
from numpy.linalg import det
import sklearn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

    
class ProtoVAE(nn.Module):

    def __init__(self):

        super(ProtoVAE, self).__init__()
        self.img_size = img_size
        self.prototype_shape = (num_prototypes,latent)
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.epsilon = 1e-4
        
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1   
            
               
        # Creating a modifiable prototype class identifier vector for keeping track of indicies during potential elimination
        K = self.num_classes # number of classes
        M = int(self.num_prototypes / self.num_classes) # number of protos pr class
        proto_class = np.zeros(self.num_prototypes)
        for k in range(K):
            for m in range(M):
                proto_class[k*M+m] = k
        self.proto_class = proto_class
      
        
        # The number of prototypes pr class. This is dynamic and could change with prototype eliminaiton
        self.protos_pr_class = np.zeros(K)
        for k in range(K):
            self.protos_pr_class[k] = sum(self.prototype_class_identity[:,k])
   
        
        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

      
        if(data_name == "mnist" or data_name == "fmnist"):
            self.features = nn.Sequential(  ###### mnist & fmnist
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.AvgPool2d(2, stride=2), ## 14x14
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.AvgPool2d(2, stride=2), ##7x7
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, latent * 2),
                nn.ReLU(),
                nn.Linear(latent * 2, latent * 2)
            )

        if (data_name == "quickdraw"):
            self.features = nn.Sequential(  ###### quickdraw
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.AvgPool2d(2, stride=2), ## 14x14
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.AvgPool2d(2, stride=2), ##7x7
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.AvgPool2d(2, stride=2), ##3x3
                nn.Flatten(),
                nn.Linear(128 * 3 * 3, latent * 2),
                nn.ReLU(),
                nn.Linear(latent * 2, latent * 2)
            )

        elif(data_name=="svhn"):
            self.features = nn.Sequential(  ###### SVHN
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Flatten(),
                nn.Linear(256*2*2, latent*2),
                nn.ReLU(),
                nn.Linear(latent * 2, latent * 2)
            )

        elif (data_name == "cifar10" or data_name == "utk"):
            self.features = nn.Sequential(  ###### CIFAR-10
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.AvgPool2d(2, stride=2), ##16x16
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.AvgPool2d(2, stride=2), ##8x8
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.AvgPool2d(2, stride=2), ##4x4
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.AvgPool2d(2, stride=2), ##2x2

                nn.Flatten(),
                nn.Linear(256 * 2 * 2, latent * 2),
                nn.ReLU(),
                nn.Linear(latent * 2, latent * 2)
            )
            
            
        if(data_name == "mnist" or data_name == "fmnist" ):
            self.decoder_layers = nn.Sequential(  ###### MNIST
                nn.Linear(latent, 64 * 7 * 7),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(64 * 7 * 7),
                View((-1, 64, 7, 7)),

                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
        elif(data_name == "quickdraw"):
            self.decoder_layers = nn.Sequential(  ###### QuickDraw
                nn.Linear(latent, 128 * 3 * 3),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128 * 3 * 3),
                View((-1, 128, 3, 3)),

                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64), ##6x6

                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32), ##12x12

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32), ##24x24

                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
            )
            
            
        

        elif(data_name=="svhn"):
            self.decoder_layers = nn.Sequential(  ###### SVHN
                nn.Linear(latent, 256*2*2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256*2*2),
                View((-1, 256, 2, 2)),

                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),

                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),

                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),

        )
        elif(data_name=="cifar10" or data_name == "utk"):
            self.decoder_layers = nn.Sequential(  ###### CIFAR-10
                nn.Linear(latent, 256 * 2 * 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256 * 2 * 2),
                View((-1, 256,2,2)),

                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),    ##8x8

                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),  ##16x16

                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),  ##32x32

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),

            )
            
                    

        self._initialize_weights()


    def decoder(self, z):
        x = self.decoder_layers(z)
        x = torch.tanh(x)
        return x



    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps


    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))


    def forward(self, x, y=None, is_train=True,is_elim=False):
        conv_features = self.features(x)

        mu = conv_features[:,:latent] # Shape [batch_size,d]
        logVar = conv_features[:,latent:].clamp(np.log(1e-8), -np.log(1e-8))  # Shape [batch_size,d]
        z = self.reparameterize(mu, logVar)
        if(~is_train):
            z = mu

        sim_scores = self.calc_sim_scores(z) # Shape [batch_size,N_protos]
        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, y]).to(device)
        
        if is_elim == True: # if prototypes are elliminated during training.

        # List of lists with correct indicies for each obs in batch
            index_prototypes_of_correct_class = []
            for target in y:
                index_prototypes_of_correct_class.append(list(np.where(self.proto_class==int(target))[0]))
                     
        # List of lists with unequal length to numpy array padded with zeros.
        # This is used during the KL-loss
            b = np.zeros([len(index_prototypes_of_correct_class),len(max(index_prototypes_of_correct_class,key = lambda x: len(x)))])
            for i,j in enumerate(index_prototypes_of_correct_class):
                b[i][0:len(j)] = j 
            index_prototypes_of_correct_class = b            
             # index_prototypes.. stores the index of the prototypes that are associated with each obs in batch. 
             # This is a list of lists to take into account that each class can have different number of protos. 
             # I.e. [[ 0,  1,  2],[45, 46]...] for the first two obs, if class of i=1 has 3 protos and i=2 has 2 protos.
                                                   
            kl_loss = self.kl_divergence_nearest(mu, logVar, index_prototypes_of_correct_class, sim_scores,y,is_elim)    
            volume_loss,volume_loss_list = self.volume_loss()
            orth_loss = self.orth_loss()            
            out = self.last_layer(sim_scores)
            decoded = self.decoder(z)
            return out, decoded, kl_loss, volume_loss, sim_scores,z,orth_loss,volume_loss_list  
        
        
        if is_elim == False:         
            #Shape: torch.Size([640]) I.e. batch size * M (batch size = 128 and M = 5)
            index_prototypes_of_correct_class = (prototypes_of_correct_class == 1).nonzero(as_tuple=True)[1]
            #Shape: torch.Size([128, 5])
            index_prototypes_of_correct_class = index_prototypes_of_correct_class.view(x.shape[0],self.num_prototypes_per_class)            
             # index_prototypes.. stores the index of the M prototypes that are associated with each obs in batch. 
             # I.e. [[ 0,  1,  2,  3,  4],[45, 46, 47, 48, 49]...] for the first two obs in batch of class 0 and 9. 
            
            kl_loss = self.kl_divergence_nearest(mu, logVar, index_prototypes_of_correct_class, sim_scores,y,is_elim)
            volume_loss,volume_loss_list = self.volume_loss()
            orth_loss = self.orth_loss()        
            out = self.last_layer(sim_scores)
            decoded = self.decoder(z)

            return out, decoded, kl_loss, volume_loss, sim_scores,z,orth_loss,volume_loss_list


    def orth_loss(self):
        s_loss = 0
        for k in range(self.num_classes):
            p_k = self.prototype_vectors[np.where(self.proto_class==k)[0]] # Selecting the prototypes for class k
            p_k_mean = torch.mean(p_k, dim=0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2 @ p_k_2.T
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0]).to(device))
            s_loss+= torch.norm(s_matrix,p=2)
        return s_loss/self.num_classes


    def volume_loss(self):
        vol_loss = 0
        vol_loss_list = []
        for k in range(self.num_classes):
       
            # ========= Linear Kernal Funtion =======            
            p_k = self.prototype_vectors[np.where(self.proto_class==k)[0]] # Selecting the prototypes for class k        
            p_k = p_k.T #To form the parallelotope over the M prototypes we need the dimension to be [d x M]
            L = p_k.T@p_k # Creating the L-ensemble (similarity kernel)            
            # Optional scaling of the L-ensemble to avoid the volume exploding if using M > 15 protos pr class
            scale_factor = 1 # Use 0.1 at 20 protos pr class.            
            L = L*scale_factor     
            
            volume = torch.sqrt(torch.linalg.det(L)) # the volume of the class
            vol_loss += 1/volume # converting to loss - i.e. minimize the inverse volume   
            
           # vol_loss = torch.tensor(0) #If vol loss scaling of 0 potentially use this to avoid NaNs in volume computation
         
            vol_loss_list.append((1/volume)) # Saving list of volume loss for each class                                    
            # ========= Linear Kernal Funtion =======            

              
            # ========= Radial Basis Funtion =======
             # Optionally use another similarity kerne, such as the RBF kernel. 
#            c =1.0 / p_k.shape[0]    
#            c = 1.0 / 5
#            c=1.0
#            L = torch.exp(-c*torch.square(torch.norm((p_k.T.unsqueeze(1)-p_k.T), dim=2, p=2))) # RBF kernel
#            volume = torch.sqrt(torch.linalg.det(L)) # the volume of the class
#            vol_loss += 1/volume # converting to loss - i.e. minimize the inverse volume 
            # ========= Radial Basis Funtion =======
                                      
        return vol_loss/self.num_classes , vol_loss_list 

 
    def calc_sim_scores(self, z):
        d = torch.cdist(z, self.prototype_vectors, p=2)  ## Batch size x prototypes
        sim_scores = self.distance_2_similarity(d)
        return sim_scores


    
    def kl_divergence_nearest(self, mu, logVar, nearest_pt, sim_scores,y,is_elim):
         # Nearest pt is the index of prototypes for each obs in batch. Shape: torch.Size([128, 5])
         # Prototype_Vectors has Shape: [n_protos,d], for example [50,256]
         # Sim_scores: Shape [128,50]
        
        if is_elim==False:        
            kl_loss = torch.zeros(sim_scores.shape).to(device) # kl loss shape [batch size, n_protos] i.e. [128,50]
            for i in range(self.num_prototypes_per_class): # looping over the M prototypes each obs can belong to. 
                p = torch.distributions.Normal(mu, torch.exp(logVar / 2))
                # p_v is 1 of the prototypes in the class that each obs in batch can belong to. We loop over all M prototypes. 
                p_v = self.prototype_vectors[nearest_pt[:,i],:]  # shape [batch size,d] i.e. [128,256]
                q = torch.distributions.Normal(p_v, torch.ones(p_v.shape).to(device))
                kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim=1) # shape[128] mean KL over shape [128,256]
                # kl_loss is inserted for each obs to the i'th proto. We loop over all M protos.
                # The kl_loss for all other prototypes than the correct class is set to zero. 
                kl_loss[np.arange(sim_scores.shape[0]),nearest_pt[:,i]] = kl # Shape [128,50]                                    

        if is_elim==True:          
            # ========= Prototype elimination version =======   
            kl_loss = torch.zeros(sim_scores.shape).to(device) # kl loss shape [batch size, n_protos] i.e. [128,50]
            for k in range(self.num_classes):     
                for i in range(int(self.protos_pr_class[k])):
                    p = torch.distributions.Normal(mu[y==k], torch.exp(logVar / 2)[y==k])
                    p_v = self.prototype_vectors[nearest_pt[y==k,i],:] 
                    q = torch.distributions.Normal(p_v, torch.ones(p_v.shape).to(device))
                    kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim=1) 
                    kl_loss[np.arange(sim_scores.shape[0])[y==k],nearest_pt[y==k,i]] = kl                 
            # ========= Prototype elimination version =======    
     
        kl_loss = kl_loss*sim_scores
        mask = kl_loss > 0 # only the kl_loss associated with the correct class. 
        kl_loss = torch.sum(kl_loss, dim=1) / (torch.sum(sim_scores * mask, dim=1)) # normalizing the sim_scores
        kl_loss = torch.mean(kl_loss)
        return kl_loss
    

    def get_prototype_images(self):
        p_decoded = self.decoder(self.prototype_vectors)
        return p_decoded
    
    def get_prototypes(self):
        prototypes = self.prototype_vectors
        return prototypes


    def pred_class(self, x):
        conv_features = self.features(x)

        mu = conv_features[:, :latent]
        z = mu

        sim_scores = self.calc_sim_scores(z)

        out = self.last_layer(sim_scores)

        return out, sim_scores


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)





