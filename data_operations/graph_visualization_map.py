import numpy as np

from data_operations.train_operations import *
from data_operations.load_data import *
from data_operations.train_operations import *
from data_operations.image_interaction import *

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import networkx as nx

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad



class graph_visualization_maps(object):
    def __init__(self, mode, node_dim):
        self.mode = mode
        self.node_dim = node_dim
        self.nodes = node_dim ** 2
        self.activation = {}



    def select_analysis_mode(self):
        if self.mode == 'probability_model':
            self.probability_model()
        if self.mode == 'visuals':
            self.get_visuals()
        if self.mode == 'node':
            self.get_node_visuals()
        if self.mode == 'GVM':
            self.GVM()
        if self.mode == 'GVM_cosine_similarity':
            self.GVM_cosine_sim()
        if self.mode == 'select_node':
            self.GVM_select_node()




    def prepare_data(self, train=False):
        dataproc = data_preprocess(config)
        if train:
            images = dataproc.train_imgs
            gt = dataproc.train_gt
        else:
            images = dataproc.val_imgs
            gt = dataproc.val_gt

        val_dataset = ImageDataset(images, gt, config.input_dim, config.augment_validation)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
        if config.load_weights:
            model = config.model
            model.load_state_dict(torch.load(config.resume_training_path).state_dict())
            model = model.cuda()
        else:
            model = torch.load(config.resume_training_path)

        #model.norm.register_forward_hook(self.get_activation('edge_weight'))
        #model.backbone.fc.register_forward_hook(self.get_activation('nodes'))
        #model.norm2.register_forward_hook(self.get_activation('node_output'))
        #self.target_layers = model.backbone.fc

        return val_loader, model

    def extract_model_features(self):
        edge_weights = self.activation.get('edge_weight')
        edge_weights = edge_weights[0, :]
        edge_weights = edge_weights.cpu().numpy()
        node_vals = self.activation.get('nodes')
        node_vals = node_vals[0, :]
        node_vals = node_vals.cpu().numpy()
        node_output = self.activation.get('node_output')
        node_output = node_output.cpu().numpy()

        return edge_weights, node_vals, node_output


    def GVM(self):
        val_loader, model = self.prepare_data()
        counter = 0
        for x, y in val_loader:
            output, y, cls = self.get_model(model, x, y)
            if y == 1:
                edge_weights, node_vals, node_output = self.extract_model_features()

                ### generate edges
                g = nx.complete_graph(self.nodes)
                edges = list(g.edges)
                image_inter = image_interaction(x, self.node_dim, edges=edges, edge_weights=edge_weights, node_values=node_vals,
                                                node_output=node_output)
                image_inter.GVM_n(label=counter, cls=cls)

    def probability_model(self):
        train_loader, model = self.prepare_data(train=True)
        size = len(train_loader)

        counter = 0
        for x, y in train_loader:
            output, y, cls = self.get_model(model, x, y)
            edge_weights = model.edge_arr[0]
            edge_weights = edge_weights.cpu().detach().numpy()

            ### generate edges
            g = nx.complete_graph(self.nodes)
            edges = list(g.edges)
            image_inter = image_interaction(x, self.node_dim, edges=edges, edge_weights=edge_weights,
                                            node_values=None, node_output=None)
            adj = image_inter.adjacency
            if counter == 0:
                avg_adj = np.zeros_like(adj)
            avg_adj = avg_adj + adj
            counter += 1
            print(counter)

        avg_adj = avg_adj / size
        print(avg_adj)
        degree_matrix = np.sum(avg_adj, axis=1)

        quit()



    def GVM_cosine_sim(self):
        val_loader, model = self.prepare_data()
        counter = 0
        for x, y in val_loader:
            output, y, cls = self.get_model(model, x, y)
            if y == 1 or y == 2:
                edge_weights = model.edge_arr[0]
                edge_weights = edge_weights.cpu().detach().numpy()

                ### generate edges
                g = nx.complete_graph(self.nodes)
                edges = list(g.edges)
                image_inter = image_interaction(x, self.node_dim, edges=edges, edge_weights=edge_weights, node_values=None, node_output=None)
                image_inter.visualize_on_degree_matrix(label=counter, cls='pc')
                counter += 1
                if config.CAM:
                    self.CAM_visual(model, x, image_inter)


    def GVM_select_node(self):
        val_loader, model = self.prepare_data()
        counter = 0
        for x, y in val_loader:
            output, y, cls = self.get_model(model, x, y)
            if y == 1 or y == 2:
                edge_weights = model.edge_arr[0]
                edge_weights = edge_weights.cpu().numpy()

                ### generate edges
                g = nx.complete_graph(self.nodes)
                edges = list(g.edges)
                image_inter = image_interaction(x, self.node_dim, edges=edges, edge_weights=edge_weights, node_values=None,
                                                node_output=None)
                image_inter.visualize_multi_node_neighborhood(node_count_selector=1, label=counter, cls=cls)
                counter += 1



    def get_visuals(self):
        val_loader, model = self.prepare_data()
        counter = 0
        for x, y in val_loader:
            counter += 1
            output, y, cls = self.get_model(model, x, y)
            if y == 1:
                edge_weights, node_vals, node_output = self.extract_model_features()
                ### generate edges
                g = nx.complete_graph(self.nodes)
                edges = list(g.edges)

                image_inter = image_interaction(x, self.node_dim, edges=edges, edge_weights=edge_weights, node_values=node_vals, node_output=node_output)
                image_inter.visualize_on_degree_matrix(label=counter, cls=cls)


    def get_node_visuals(self):
        val_loader, model = self.prepare_data()
        counter = 0
        for x, y in val_loader:
            output, y, cls = self.get_model(model, x, y)

            if y != 0:
                edge_weights, node_vals, node_output = self.extract_model_features()

                ### generate edges
                g = nx.complete_graph(self.nodes)
                edges = list(g.edges)
                image_inter = image_interaction(x, self.node_dim, edges=edges, edge_weights=edge_weights, node_values=node_vals, node_output=node_output)
                image_inter.visualize_on_nodes(label=counter, cls=cls)


                if config.CAM:
                    self.CAM_visual(model, x, image_inter)

            counter += 1

    def CAM_visual(self, model, x, image_inter):
        cam = GradCAM(model=model, target_layers=[self.target_layers], use_cuda=True)
        x = x.cuda()
        x = Variable(x)
        heat_map = cam(input_tensor=x)
        heat_map = heat_map.squeeze()
        # heat_map = np.swapaxes(heat_map, 0, -1)
        plt.imshow(image_inter.image)
        plt.imshow(heat_map, alpha=0.4, cmap="jet")
        plt.show()

    def get_model(self, model, x, y):
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        y = torch.argmax(y, dim=1)
        y = y.item()
        #print('class: {}'.format(y))
        output = model(x)
        output = torch.softmax(output, dim=1)

        cls = None

        cls = config.experiment_name
        return output, y, cls


    def get_activation(self, name):

        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


















