# -*- coding: utf-8 -*-
# =============================================================================
# author: Andrew R. Wilzman
# functions:
#    MTDataset(Dataset): Data loader 
#       this is used to fish for data as needed instead of loading in 
#       all of the pointclouds at once.
#       only the filepath matrix is needed to tag data for batching
# =============================================================================
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import time
import h5py
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
#%% Loss Functions
class Chamfer_Loss(nn.Module):
    def __init__(self):
        super(Chamfer_Loss, self).__init__()        
    def forward(self, inputs, preds):
        return self.compute_chamfer_loss(inputs, preds)
                
    def compute_chamfer_loss(self, inputs, preds):
        loss = self.compute_loss(inputs, preds)
        
        self.clip_gradients() # only you can prevent gradient explosions
        
        return loss
    
    def clip_gradients(self):
        # Iterate through all parameters
        for param in self.parameters():
            if param.grad is not None:
                nn.utils.clip_grad_norm_(param, self.clip_threshold)

    def compute_loss(self, cloud1, cloud2):
        # Compute squared Euclidean distances using broadcasting and vectorized operations
        diff = cloud1.unsqueeze(2) - cloud2.unsqueeze(1)  # Shape: (batch_size, num_points_cloud1, num_points_cloud2, 3)
        distances_squared = torch.sum(diff ** 2, dim=-1)  # Shape: (batch_size, num_points_cloud1, num_points_cloud2)
        
        # Find the mean minimum distance along the second and third dimension
        min_distances1, _ = torch.min(distances_squared, dim=1)
        min_distances2, _ = torch.min(distances_squared, dim=2)
        min1 = torch.mean(min_distances1)
        min2 = torch.mean(min_distances2)
        
        # Take the max
        return max(min1, min2)

class GAN_Loss(nn.Module):
    def __init__(self):
        super(GAN_Loss, self).__init__()
        self.guess_loss = nn.BCELoss()
    
    def forward(self, predicted, actual, pred_cloud=None, tgt_cloud=None):
        loss = self.guess_loss(predicted, actual)
        
        # If predicted point cloud and target point cloud are provided, compute KL divergence loss
        if pred_cloud is not None and tgt_cloud is not None:
            loss_cloud = Chamfer_Loss.compute_loss(self,pred_cloud, tgt_cloud)
            loss += loss_cloud * 0.1
        
        return loss
    
class JSD_Loss(nn.Module):
    def __init__(self):
        super(JSD_Loss,self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        # Compute probabilities from logits using softmax
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        # Calculate the average probability distribution
        m = 0.5 * (net_1_probs + net_2_probs)

        # Initialize loss
        loss = 0.0

        # Calculate Kullback-Leibler (KL) divergence for both networks
        # and accumulate the loss
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean")

        return 0.5 * loss

#%% Datasets
class MTDataset(Dataset):
    def __init__(self, file_paths, num_points=1024, k=16):
        self.file_paths = file_paths
        self.num_points = num_points
        self.k = k
        self.data = []
        
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as hf:
                bone = hf['Surface'][:]
                mt_no = int(np.array(hf['MTno']))
                side = np.array(hf['Side'])
                side_int = 0 if 'L' in str(side) else 1
                
                num_samples = min(self.num_points, bone.shape[0])
                sampled_indices = torch.randperm(bone.shape[0])[:num_samples]
                sampled_bone = bone[sampled_indices]
                
                edge_index = self.compute_knn_graph(sampled_bone, k=self.k)
                
                self.data.append((sampled_bone, mt_no, side_int, edge_index))
                
                # Generate extra samples by rotating 180 degrees on x and y axes
                R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                R_xy = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                
                rotate_x_bone = np.dot(sampled_bone, R_x)
                rotate_y_bone = np.dot(sampled_bone, R_y)
                rotate_xy_bone = np.dot(sampled_bone, R_xy)
                
                edge_index_rotate_x = self.compute_knn_graph(rotate_x_bone, k=self.k)
                edge_index_rotate_y = self.compute_knn_graph(rotate_y_bone, k=self.k)
                edge_index_rotate_xy = self.compute_knn_graph(rotate_xy_bone, k=self.k)
                
                self.data.append((rotate_x_bone, mt_no, side_int, edge_index_rotate_x))
                self.data.append((rotate_y_bone, mt_no, side_int, edge_index_rotate_y))
                self.data.append((rotate_xy_bone, mt_no, side_int, edge_index_rotate_xy))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def compute_knn_graph(self, positions, k):
        neighbors = NearestNeighbors(n_neighbors=k+1).fit(positions)
        distances, indices = neighbors.kneighbors(positions)
        edge_index = []
        
        for i in range(positions.shape[0]):
            for j in range(1, k+1):  # Start from 1 to skip the point itself
                edge_index.append([i, indices[i, j]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

def custom_collate(batch):
    batch_x = [torch.FloatTensor(item[0]) for item in batch]
    batch_mt_no = torch.tensor([item[1] for item in batch], dtype=torch.int8)
    batch_side = torch.tensor([item[2] for item in batch], dtype=torch.int8)
    batch_edge_index = []

    num_nodes = [item[0].shape[0] for item in batch]
    cumsum_nodes = np.cumsum([0] + num_nodes)
    
    for i, item in enumerate(batch):
        edge_index = item[3] + cumsum_nodes[i]
        batch_edge_index.append(edge_index)
    
    batch_edge_index = torch.cat(batch_edge_index, dim=1)
    
    return {
        'surf': torch.stack(batch_x),
        'mt_no': batch_mt_no,
        'side': batch_side,
        'knn': batch_edge_index
    }
#%% Training functions
# Please add saving function to report model parameters during training
def train_autoencoder(training_inputs, network, epochs, learning_rate, wtdecay, 
                      batch_size, loss_function, print_interval, device,  
                      num_points=1024,cycles=1):
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
        
    track_losses = []
    start = time.time()
    print('Timer started')
    
    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            batch_surf = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            encoded_X = batch_surf.clone().to(device)
            
            for _ in range(cycles):
                encoded_X = network.encode(encoded_X, edge_index)
                encoded_X = network.decode(encoded_X, batch_surf.shape[1])
            
            loss = loss_function(batch_surf, encoded_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        training_loss = loss.sqrt()
        track_losses.append(training_loss.item())
        
        if epoch % print_interval == 0:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
        
        if time.time() - start > end_time:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
            break
        
    return network, track_losses

def train_vae(training_inputs, network, epochs, learning_rate, wtdecay,
              batch_size, loss_function, print_interval, device, num_points=1024,cycles=1):
    
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
        
    track_losses = []
    start = time.time()
    print('Timer started')
    
    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            batch_surf = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            encoded_X = batch_surf.clone().to(device)
            
            for _ in range(cycles):
                encoded_X = network.encode(encoded_X, edge_index)
                latent = encoded_X.clone()
                encoded_X = network.decode(encoded_X, batch_surf.shape[1])
                
            mean = latent.mean(dim=2, keepdim=True)
            std = latent.std(dim=2, keepdim=True)
            
            # Create a normal distribution for each feature
            epsilon = 1e-6  # Small constant to avoid zero std deviation
            std = latent.std(dim=2, keepdim=True) + epsilon
            
            normal_dist = torch.distributions.Normal(mean[:, 0, 0], std[:, 0, 0])
            
            std_normal = torch.distributions.Normal(0,1)
            
            loss_kl = torch.distributions.kl.kl_divergence(normal_dist, std_normal).mean()
            
            recon_loss = loss_function(batch_surf, encoded_X)
            
            loss = recon_loss + (loss_kl*100)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        training_loss = loss.sqrt()
        track_losses.append(training_loss.item())
        
        if epoch % print_interval == 0:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
        
        if time.time() - start > end_time:
            elapsed_time = time.time() - start
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {elapsed_time:7.1f}s')
            break
        
    return network, track_losses

def train_diffusion(training_inputs, network, epochs, learning_rate, wtdecay,
                    batch_size, loss_function, print_interval, device, noise_level=1, num_points=1024, cycles=1):
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
        
    track_losses = []
    start = time.time()
    print('Timer started')
    
    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            batch_surf = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            encoded_X = batch_surf.clone().to(device)
            
            lamb_t = max(min(noise_level/10, 1), 0.05)
            sig_t = 1 - lamb_t
                
            encoded_X = network.encode(encoded_X, edge_index)
            
            # Diffusion process
            
            noise = noise_level * torch.randn_like(encoded_X).to(device)
            noisy = sig_t * encoded_X + lamb_t * noise
            
            encoded_X = network.decode(noisy, batch_surf.shape[1])
            
            # Final output
            loss = loss_function(batch_surf, encoded_X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        training_loss = loss.item()**0.5
        track_losses.append(training_loss)
        
        if time.time() - start > end_time:
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {time.time() - start:7.1f}s')
            break
        
        if epoch % print_interval == 0:
            print(f'Epoch: {epoch:4d}, Training Loss: {training_loss:10.3e}, Time: {time.time() - start:7.1f}s')
        
    return network, track_losses

# GAN training function
def train_GD(training_inputs, Gnet, Dnet, dec_hid, epochs, learning_rate, decay,
             batch_size, loss_function, print_interval, device, num_points=1024):
    
    dataset = MTDataset(training_inputs, num_points)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=custom_collate)
    
    optimizerG = torch.optim.Adam(Gnet.parameters(), lr=learning_rate, weight_decay=decay)
    optimizerD = torch.optim.Adam(Dnet.parameters(), lr=learning_rate, weight_decay=decay)
    
    G_losses = []
    D_losses = []
    
    Gnet = Gnet.to(device)
    Dnet = Dnet.to(device)
    
    start = time.time()
    print('Timer started')
    
    end_time = float('inf')
    if epochs < 0:
        end_time = -epochs
        epochs = int(1e9)
    
    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader):
            data = batch['surf'].to(device)
            edge_index = batch['knn'].to(device)
            
            b_size = len(data)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            
            # Train with real data
            output = Dnet(data).view(-1)
            lossD_real = loss_function(output, label)
            
            optimizerD.zero_grad()
            lossD_real.backward()
            optimizerD.step()
            
            D_x = output.mean().item()  # 1 if discriminator is doing well
            
            b = Gnet.encode(data, edge_index)
            
            noise = torch.randn(b_size, b.shape[1], b.shape[2], device=device)
            fake = Gnet.decode(noise, data.shape[1])
            
            label.fill_(0)
            output = Dnet(fake).view(-1)
            
            D_G_z1 = output.mean().item()  # 0 if discriminator is doing well
            
            lossD_fake = loss_function(output, label)  # Closer to 1 means generator is fooling
            
            optimizerD.zero_grad()
            lossD_fake.backward()
            optimizerD.step()
            
            output = Dnet(fake).view(-1)
            D_G_z2 = output.mean().item()  # 0 if discriminator is doing well
            
            # Update generator
            fake = Gnet.decode(noise,data.shape[1])
            label.fill_(1)
            output = Dnet(fake).view(-1)
            lossG = loss_function(output, label, fake, data)
            
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            
            # Track losses
            lossD = lossD_real + lossD_fake
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
        
        if epoch % print_interval == 0:
            print('[%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % 
                  (epoch, time.time() - start, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        
        if time.time() - start > end_time:
            print('[%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % 
                  (epoch, time.time() - start, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
            break
    
    return Gnet, G_losses, Dnet, D_losses

def model_eval_chamfer(x, model, num_nodes, device, batch_size=10):
    ds = MTDataset(x, num_nodes)
    loader = DataLoader(ds,batch_size=batch_size,num_workers=4,shuffle=False,collate_fn=custom_collate)
    
    chamfer_losses = []
    jensen_shannon_divergences = []
    with torch.no_grad():
        for X in loader:
            b = X['surf'].to(device)
            k = X['knn'].to(device)
            encoded_tensor = model.encode(b,k)
            
            rec = model.decode(encoded_tensor, num_nodes)
            
            dist_x_to_y = torch.cdist(b.view(-1, 3), rec.view(-1, 3))
            dist_y_to_x = torch.cdist(rec.view(-1, 3), b.view(-1, 3))
            chamfer_loss = dist_x_to_y.min(dim=1).values.mean() + dist_y_to_x.min(dim=1).values.mean()
            chamfer_losses.append(chamfer_loss.item())
            
            jsd_loser = JSD_Loss()
            jensen_shannon_divergences.append(jsd_loser(b,rec))
        
        avg_chamfer_loss = sum(chamfer_losses) / len(chamfer_losses)
        avg_jsd = sum(jensen_shannon_divergences) / len(chamfer_losses)
        
    return avg_chamfer_loss, avg_jsd