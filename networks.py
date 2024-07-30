# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:03:37 2024

@author: Andrew R Wilzman
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from itertools import product        

class jarvis(nn.Module): # Discriminator network
    def __init__(self, insize):
        super(jarvis, self).__init__()
        self.activate = nn.SELU()
        self.h1 = 256
        self.h2 = 64
        self.h3 = 32
        self.h4 = 16
        # First encoding MLP
        self.fc_encoder1 = nn.Sequential(
            nn.Linear(3, self.h4),  
            self.activate,
            nn.Linear(self.h4, self.h3),
            self.activate,
            nn.Linear(self.h3, self.h2),
            nn.Dropout(0.2),
            nn.Linear(self.h2, self.h1),
            self.activate,
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.2),
            self.activate,
            nn.Linear(self.h2, self.h3),
            self.activate,
            nn.Linear(self.h3, self.h4),
            self.activate,
            nn.Linear(self.h4, 1),
            nn.Sigmoid())
        
        self.pooler = nn.AdaptiveAvgPool1d(self.h1)
        
        self.fc_encoder2 = nn.Sequential(
            nn.Linear(self.h1, self.h2),
            nn.Dropout(0.2),
            self.activate,
            nn.Linear(self.h2, self.h3),
            self.activate,
            nn.Linear(self.h3, self.h4),
            self.activate,
            nn.Linear(self.h4, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        y = self.fc_encoder1(x)
        y = self.pooler(y.permute(0,2,1))
        y = self.fc_encoder2(y)
        x = y.permute(0,2,1)
        return x
    
class arw_FoldingNet(nn.Module):
    def __init__(self, h1, h3, initial_state=None, max_depth=4):
        super(arw_FoldingNet, self).__init__()
        self.activate = nn.SELU()
        self.h1 = h1
        self.h3 = h3
        self.input_dim = 12
        
        self.max_depth = max_depth
        self.max_width = h1
        
        self.e_layers1 = nn.ModuleList()
        self.e_layers2 = nn.ModuleList()
        self.d_layers1 = nn.ModuleList()
        self.d_layers2 = nn.ModuleList()
        
        self.initialize_state(initial_state)
        
        # GraphConv layer for graph-based encoder
        self.graph_encoder1 = GCNConv(self.h3, self.h3*2)
        self.graph_encoder2 = GCNConv(self.h3*2, self.h1)
        
        self.pooler = nn.AdaptiveMaxPool1d(1)
        
        
    def initialize_state(self, initial_state):
        if initial_state is None:
            initial_state = [[(self.input_dim,self.h3)],[(self.h1,self.h1)],[(self.h1+2,3)],[(self.h1+3,3)]]
        for widths, layer_list in zip(initial_state, [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]):
            for insize, outsize in widths:
                layer_list.append(nn.Linear(insize, outsize))
                
    def compute_local_covariances(self, point_cloud, neighborhood_size):
        batch, num_points, dim = point_cloud.shape
        all_covariances = torch.zeros((batch, num_points, dim * dim), dtype=point_cloud.dtype).to(point_cloud.device)

        for i in range(num_points):
            neighborhood_start = max(0, i - neighborhood_size // 2)
            neighborhood_end = min(num_points, i + neighborhood_size // 2 + 1)
            neighborhood = point_cloud[:, neighborhood_start:neighborhood_end, :]

            centered_neighborhood = neighborhood - torch.mean(neighborhood, dim=1, keepdim=True)
            covariance_matrix = torch.matmul(centered_neighborhood.transpose(1, 2), centered_neighborhood) / (neighborhood_size - 1)
            flattened_covariance = covariance_matrix.flatten(start_dim=1)

            all_covariances[:, i, :] = flattened_covariance

        return all_covariances
    
    def get_layer_list(self, mlp_no):
        
        if mlp_no == 1:
            return self.e_layers1
        elif mlp_no == 2:
            return self.e_layers2
        elif mlp_no == 3:
            return self.d_layers1
        elif mlp_no == 4:
            return self.d_layers2
        else:
            return [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]
        
    def add_layer(self, mlp_no, index):
        layer_list = self.get_layer_list(mlp_no)
        if len(layer_list) < self.max_depth:
            if index == 0:
                if mlp_no == 1:
                    prev_width = self.input_dim
                elif mlp_no == 2:
                    last = self.get_layer_list(1)
                    prev_width = self.h1
                elif mlp_no == 3:
                    prev_width = self.h1 + 2
                else:
                    prev_width = self.h1 + 3
            else:
                prev_width = layer_list[index - 1].out_features
            new_layer = nn.Linear(prev_width, prev_width)
            nn.init.xavier_uniform_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
            layer_list.insert(index, new_layer)
        else:
            # add width expansion
            print(f'maxed layer depth for MLP {mlp_no}')

    def change_width(self, mlp_no, index, new_width):
        layer_list = self.get_layer_list(mlp_no)
        new_width = min(new_width,self.max_width)
        if 0 <= index < len(layer_list)-1:
            prev_in = layer_list[index].in_features
            layer_list[index] = nn.Linear(prev_in, new_width)
    
            # Update subsequent layer
            if index < len(layer_list) - 1:
                next_width = layer_list[index + 1].out_features
                layer_list[index + 1] = nn.Linear(new_width, next_width)
                
    def encode(self, data, knn):
        batch_size, num_nodes, num_features = data.size()
        cov = self.compute_local_covariances(data, 24)
        x = torch.cat([data, cov], dim=2)
        
        for layer in self.e_layers1:
            x = self.activate(layer(x))
        
        x = x.view(-1, x.size(-1))
        
        x = self.activate(self.graph_encoder1(x, knn))
        x = self.activate(self.graph_encoder2(x, knn))
        
        x = x.view(batch_size, num_nodes, -1)
        
        x = x.permute(0, 2, 1)
        x = self.pooler(x)
        x = x.permute(0, 2, 1)
        
        for layer in self.e_layers2:
            x = self.activate(layer(x))

        return x

    def decode(self, x, num_nodes):
        num_nodes_x = int(1+(num_nodes * 120 / 60) ** 0.5)
        num_nodes_y = int(1+(num_nodes * 60 / 120) ** 0.5)
        x_grid = torch.linspace(1, 120, num_nodes_x).to(x.device)
        y_grid = torch.linspace(1, 60, num_nodes_y).to(x.device)
        grid_points = torch.cartesian_prod(x_grid, y_grid)
        grid_points = grid_points[:num_nodes]
        y = grid_points.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = x.repeat(1,num_nodes,1)
        
        k = torch.cat([x, y], dim=2)
        
        for layer in self.d_layers1:
            k = self.activate(layer(k))
        x = torch.cat([x, k], dim=2)
        for layer in self.d_layers2:
            x = self.activate(layer(x))
        
        return x
    
    def forward(self, input_data, knn):
        
        _, num_nodes, _ = input_data.size()
        x = self.encode(input_data, knn)
        y = self.decode(x, num_nodes)

        return y

    
class arw_TRSNet(nn.Module):
    def __init__(self, h1,h3,initial_state=None,max_depth=8):
        super(arw_TRSNet, self).__init__()
        self.activate = nn.SELU()
        self.h1 = h1
        while self.h1%16 != 0:
            self.h1 += 1
            print(f'changing h1 to {self.h1}')
        self.h4 = max(1,self.h1//16)
        
        self.input_dim = 3
        
        self.max_depth = max_depth
        self.max_width = self.h1
        
        self.e_layers1 = nn.ModuleList()
        self.e_layers2 = nn.ModuleList()
        self.d_layers1 = nn.ModuleList()
        self.d_layers2 = nn.ModuleList()
        
        self.initialize_state(initial_state)
        
        self.pooler1 = nn.AdaptiveMaxPool1d(self.h1)
        
        self.trs_encoder = nn.TransformerEncoderLayer(self.h1,self.h4,batch_first=True)        
        
        self.pooler2 = nn.AdaptiveMaxPool1d(1)
        # Result: 1 x h1 "codeword"
        
        self.trs_decoder = nn.TransformerDecoderLayer(self.h1, self.h4, batch_first=True)
        
    def initialize_state(self, initial_state):
        if initial_state is None:
            initial_state = [[(self.input_dim,self.h4),(self.h4,self.input_dim)],
                             [(self.input_dim*2,self.h1)],[(self.h1+2,3)],[(self.h1+3,3)]]
        for widths, layer_list in zip(initial_state, [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]):
            for insize, outsize in widths:
                layer_list.append(nn.Linear(insize, outsize))

    def add_layer(self, mlp_no, index):
        layer_list = self.get_layer_list(mlp_no)
        if len(layer_list) < self.max_depth:
            if index == 0:
                if mlp_no == 1:
                    prev_width = self.input_dim
                elif mlp_no == 2:
                    prev_width = self.input_dim*2
                elif mlp_no == 3:
                    prev_width = self.h1 + 2
                else:
                    prev_width = self.h1 + 3
            else:
                prev_width = layer_list[index - 1].out_features
            new_layer = nn.Linear(prev_width, prev_width)
            nn.init.xavier_uniform_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
            layer_list.insert(index, new_layer)
        else:
            # add width expansion
            print(f'maxed layer depth for MLP {mlp_no}')

    def change_width(self, mlp_no, index, new_width):
        layer_list = self.get_layer_list(mlp_no)
        new_width = min(new_width,self.max_width)
        if 0 <= index < len(layer_list)-1:
            prev_in = layer_list[index].in_features
            layer_list[index] = nn.Linear(prev_in, new_width)
    
            # Update subsequent layer
            if index < len(layer_list) - 1:
                next_width = layer_list[index + 1].out_features
                layer_list[index + 1] = nn.Linear(new_width, next_width)
                
    def get_layer_list(self, mlp_no):
        
        if mlp_no == 1:
            return self.e_layers1
        elif mlp_no == 2:
            return self.e_layers2
        elif mlp_no == 3:
            return self.d_layers1
        elif mlp_no == 4:
            return self.d_layers2
        else:
            return [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]
        
    def encode(self, x, knn=None):
        y = x.clone()
        
        for layer in self.e_layers1:
            y = self.activate(layer(y))
            
        y = torch.cat([x,y],dim=2)
        
        for layer in self.e_layers2:
            y = self.activate(layer(y))
        
        x = y.permute(0,2,1)
        x = self.pooler2(self.trs_encoder(self.pooler1(x)))
        x = x.permute(0,2,1)
        
        return x

    def decode(self, x, num_nodes):
        num_nodes_x = int(1+(num_nodes * 120 / 60) ** 0.5)
        num_nodes_y = int(1+(num_nodes * 60 / 120) ** 0.5)
        x_grid = torch.linspace(1, 120, num_nodes_x).to(x.device)
        y_grid = torch.linspace(1, 60, num_nodes_y).to(x.device)
        grid_points = torch.cartesian_prod(x_grid, y_grid)
        grid_points = grid_points[:num_nodes]
        y = grid_points.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = x.repeat(1,num_nodes,1)
        
        k = torch.cat([x,y], dim=2)
        
        for layer in self.d_layers1:
            k = self.activate(layer(k))
        
        x = self.trs_decoder(x, x)
        
        x = torch.cat([x,k],dim=2)
        
        for layer in self.d_layers2:
            x = self.activate(layer(x))
        
        return x
    
    def forward(self, input_data, knn=None):
        
        x = self.encode(input_data)
        y = self.decode(x)

        return y

class arw_MLPNet(nn.Module):
    def __init__(self, h1,h3, initial_state=None,max_depth=8,max_width=1024):
        super(arw_MLPNet, self).__init__()
        self.activate = nn.SELU()
        
        self.h1 = h1
        self.h3 = h3
        
        self.input_dim = 3
        
        self.max_depth = max_depth
        self.max_width = max_width
        
        self.e_layers1 = nn.ModuleList()
        self.e_layers2 = nn.ModuleList()
        self.d_layers1 = nn.ModuleList()
        self.d_layers2 = nn.ModuleList()
        
        self.initialize_state(initial_state)
        
        self.pooler1 = nn.AdaptiveMaxPool1d(1)
        
        # Result: 1 x h1 "codeword"
        
    def initialize_state(self, initial_state):
        if initial_state is None:
            initial_state = [[(self.input_dim,self.h1)],[(self.h3,self.h1)],[(self.h1+2,3)],[(self.h1+3,3)]]
        for widths, layer_list in zip(initial_state, [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]):
            for insize, outsize in widths:
                layer_list.append(nn.Linear(insize, outsize))

    def add_layer(self, mlp_no, index):
        layer_list = self.get_layer_list(mlp_no)
        if len(layer_list) < self.max_depth:
            if index == 0:
                if mlp_no == 1:
                    prev_width = self.input_dim
                elif mlp_no == 2:
                    last = self.get_layer_list(1)
                    prev_width = last[-1].out_features
                elif mlp_no == 3:
                    prev_width = self.h1 + 2
                else:
                    prev_width = self.h1 + 3
            else:
                prev_width = layer_list[index - 1].out_features
            new_layer = nn.Linear(prev_width, prev_width)
            nn.init.xavier_uniform_(new_layer.weight)
            nn.init.zeros_(new_layer.bias)
            layer_list.insert(index, new_layer)
        else:
            # add width expansion
            print(f'maxed layer depth for MLP {mlp_no}')

    def change_width(self, mlp_no, index, new_width):
        layer_list = self.get_layer_list(mlp_no)
        new_width = min(new_width,self.max_width)
        if 0 <= index < len(layer_list)-1:
            prev_in = layer_list[index].in_features
            layer_list[index] = nn.Linear(prev_in, new_width)
    
            # Update subsequent layer
            if index < len(layer_list) - 1:
                next_width = layer_list[index + 1].out_features
                layer_list[index + 1] = nn.Linear(new_width, next_width)
                
    def get_layer_list(self, mlp_no):
        
        if mlp_no == 1:
            return self.e_layers1
        elif mlp_no == 2:
            return self.e_layers2
        elif mlp_no == 3:
            return self.d_layers1
        elif mlp_no == 4:
            return self.d_layers2
        else:
            return [self.e_layers1, self.e_layers2, self.d_layers1, self.d_layers2]
    
    def encode(self, x, knn=None):
        y = x.clone()
        
        for layer in self.e_layers1:
            y = self.activate(layer(y))
            
        y = y.permute(0,2,1)
        y = self.pooler1(y)
        y = y.permute(0,2,1)
        
        for layer in self.e_layers2:
            y = self.activate(layer(y))
        
        return y

    def decode(self, x, num_nodes):
        
        num_nodes_x = int(1+(num_nodes * 120 / 60) ** 0.5)
        num_nodes_y = int(1+(num_nodes * 60 / 120) ** 0.5)
        x_grid = torch.linspace(1, 120, num_nodes_x).to(x.device)
        y_grid = torch.linspace(1, 60, num_nodes_y).to(x.device)
        grid_points = torch.cartesian_prod(x_grid, y_grid)
        grid_points = grid_points[:num_nodes]
        y = grid_points.unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        x = x.repeat(1,num_nodes,1)
        
        k = torch.cat([x,y], dim=2)
        
        for layer in self.d_layers1:
            k = self.activate(layer(k))
            
        x = torch.cat([x,k],dim=2)
        
        for layer in self.d_layers2:
            x = self.activate(layer(x))
        
        return x
    
    def forward(self, input_data, knn=None):
        
        x = self.encode(input_data)
        y = self.decode(x)

        return y

class arw_app_map(nn.Module):
    def __init__(self):
        super(arw_app_map, self).__init__()
        self.activate = nn.SiLU()
        
    def encode(self, x):
        
        return x

    def decode(self, x):
        num_points = x.shape[1]
        
        x_points = torch.rand(num_points)
        y_points = torch.rand(num_points)
        z_points = torch.rand(num_points)
        
        point_cloud = torch.stack([x_points, y_points, z_points], dim=1)
        point_cloud, _ = torch.sort(point_cloud, dim=0)
        
        y = point_cloud.unsqueeze(0).repeat(x.shape[0],1,1).to(x.device)
        
        return x
    
    def forward(self, input_data):
        
        x = self.encode(input_data)
        y = self.decode(x)

        return y