

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
from scipy.ndimage.filters import gaussian_filter
import cv2
from PIL import Image
from config import config

class image_visualization(object):
    def __init__(self, image, node_dim, edges=None, edge_weights=None, node_values=None, node_output=None):
        self.image = None
        self.grid_image = None
        self.overlay = None
        self.node_size = None
        self.tensor_image = image
        self.node_dim = node_dim
        self.edges = edges
        self.edge_weights = edge_weights
        self.node_values = node_values
        self.node_output = node_output
        self.generate_graph_info()
        self.transform_tensor_to_image()
        self.create_grid_image()

    def generate_graph_info(self):
        self.fully_connected_node = self.node_dim**2
        #self.generate_node_values()
        #self.generate_node_output()
        self.generate_weighted_edge()
        self.calculate_degree_matrix()

    def transform_tensor_to_image(self):
        if self.tensor_image.is_cuda:
            image = self.tensor_image.cpu().numpy()
        else:
            image = self.tensor_image.numpy()

        image = image.squeeze()

        image = np.swapaxes(image, 0, -1)
        image = np.swapaxes(image, 0, 1)
        image = image * 255
        image = np.uint8(image)
        self.image = image

        if len(image.shape) == 3:
            H, W, C = image.shape
        else:
            H, W = image.shape
        self.overlay = np.zeros((H, W))

    def view_map_and_image(self, heat_map):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        axes[0].imshow(self.image)
        axes[1].imshow(self.image)
        axes[1].imshow(heat_map, alpha=0.3, cmap='jet')
        manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.show()
        #plt.draw()
        #plt.pause(1)

        #plt.close(fig)

    def generate_node_values(self):
        self.node_averages = np.mean(self.node_values, axis=0)
        self.node_averages = np.ndarray.flatten(self.node_averages)

    def generate_node_output(self):
        self.node_output = np.swapaxes(self.node_output, 0, -1)
        self.node_output_averages = np.mean(self.node_output, axis=0).squeeze()


    def generate_weighted_edge(self):
        connection_arr = np.zeros((self.node_dim**2, self.node_dim**2))
        for connection, weight in zip(self.edges, self.edge_weights):
            connection_arr[connection[0], connection[1]] = weight
            connection_arr[connection[1], connection[0]] = weight

        self.adjacency = connection_arr

    def calculate_degree_matrix(self):
        adjacency_matrix = np.copy(self.adjacency)
        degree_matrix = np.sum(adjacency_matrix, axis=1)
        self.degree_matrix = degree_matrix

    def calculate_discrete_degree(self):
        adjacency_matrix = np.copy(self.adjacency)
        adjacency_matrix[adjacency_matrix != 0] = 1
        degree_matrix = np.sum(adjacency_matrix, axis=1)
        self.degree_matrix = degree_matrix


    

    def normalize_array(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        vals = (arr - min_val) / (max_val - min_val)

        return vals

    def log_scale(self, arr):
        arr = np.log(arr)

        return arr

    def z_score(self, arr):
        mean_val = np.mean(arr)
        sd_val = np.std(arr)
        vals = (arr - mean_val) / sd_val
        return vals

    def create_grid_image(self):
        image = np.copy(self.image)
        if len(image.shape) == 3:
            H, W, C = image.shape
        else:
            H, W = image.shape
        self.node_size = int(H / self.node_dim)
        #image[0:H:self.node_size, :, :] = 255
        #image[:, 0:W:self.node_size, :] = 255
        #self.grid_image = image

    def mark_image_grid(self, node, path=None):
        vals = self.adjacency[node, :]
        vals = self.normalize_array(vals)
        heat_map = np.copy(self.overlay)
        image = np.copy(self.image)
        counter = 0
        #vals[vals < (np.average(vals) + np.std(vals))] = 0
        for i in range(0, self.image.shape[0], self.node_size):
            for z in range(0, self.image.shape[1], self.node_size):
                heat_map[i:(i+self.node_size), z:(z+self.node_size)] = 255 * vals[counter]
                if counter == node:
                    image[i:(i+self.node_size), z:(z+self.node_size), 1] = 255
                counter += 1
        heat_map = self.normalize_array(heat_map)
        return image, heat_map

    def mark_image_grid_p(self, node, path):
        for c in range(0, 3):
            vals = self.adjacency[node, :]
            vals = self.normalize_array(vals)
            heat_map = np.copy(self.overlay)
            image = np.copy(self.image)
            counter = 0
            if c > 1:
                vals[vals < 0.7] = 0
            for i in range(0, self.image.shape[0], self.node_size):
                for z in range(0, self.image.shape[1], self.node_size):
                    heat_map[i:(i+self.node_size), z:(z+self.node_size)] = 255 * vals[counter]
                    if counter == node:
                        print('[INFO] marking grid at {}'.format(node))
                        image[i:(i+self.node_size), z:(z+self.node_size), 1] = 255
                    counter += 1
            #heat_map = gaussian_filter(heat_map, sigma=20)
            heat_map = self.normalize_array(heat_map)
            self.view_map(image, heat_map, path, c)


    def visualize_max(self):
        combined_heat = np.copy(self.overlay)
        nodes = self.node_dim**2
        arr = np.zeros(nodes)
        for i in range(0, nodes):
            vals = self.adjacency[i, :]
            heat_map = np.copy(self.overlay)
            image = np.copy(self.image)
            counter = 0
            vals[vals < np.max(vals)] = 0
            for i in range(0, self.image.shape[0], self.node_size):
                for z in range(0, self.image.shape[1], self.node_size):
                    heat_map[i:(i + self.node_size), z:(z + self.node_size)] = 255 * vals[counter]
                    arr[counter] = vals[counter]
                    counter += 1
            combined_heat = combined_heat + heat_map
        heat_map = self.normalize_array(combined_heat)
        heat_map[heat_map < (np.average(heat_map) + np.std(heat_map))] = 0

        heat_map = heat_map * 255

        return arr


    def GVM(self, label, cls):
        vals = self.degree_matrix
        vals = self.normalize_array(vals)
        #vals[vals < np.mean(vals)] = 0
        heat_map = np.copy(self.overlay)
        image = np.copy(self.image)
        counter = 0
        max_5 = np.argsort(self.node_averages)[::-1][:5]
        print(max_5)
        for i in range(0, self.image.shape[0], self.node_size):
            for z in range(0, self.image.shape[1], self.node_size):
                if counter in max_5:
                    print('here')
                    heat_map[i:(i + self.node_size), z:(z + self.node_size)] = 255 * vals[counter]

                counter += 1
        self.view_map(heat_map, label)




    def visualize_on_degree_matrix(self, label, cls):
        vals = self.degree_matrix
        vals = self.normalize_array(vals)
        heat_map = np.copy(self.overlay)
        image = np.copy(self.image)
        counter = 0
        print(len(vals))
        #vals[vals < (np.average(vals) + (np.std(vals) / 2))] = 0
        #vals = self.normalize_array(vals)

        for i in range(0, self.image.shape[0], self.node_size):
            for z in range(0, self.image.shape[1], self.node_size):
                heat_map[i:(i + self.node_size), z:(z + self.node_size)] = 255 * vals[counter]

                counter += 1
        if config.erode:
            heat_map = gaussian_filter(heat_map, sigma=20)
            self.view_map(heat_map, label, cls)
            heat_map = cv2.erode(heat_map, kernel=np.ones((config.erode_kernel, config.erode_kernel)), iterations=20)
            heat_map = cv2.dilate(heat_map, kernel=np.ones((config.erode_kernel, config.erode_kernel)), iterations=10)
        if config.gaussian_filter:
            heat_map = gaussian_filter(heat_map, sigma=20)
        self.view_map(heat_map, label, cls)



    
    def visualize_on_nodes(self, label, cls):

        vals = self.degree_matrix

        vals = self.normalize_array(vals)
        heat_map = np.copy(self.overlay)
        image = np.copy(self.image)
        counter = 0
        #vals[vals < 0.75] = 0
        print('Average: {}'.format(np.average(vals)))
        vals = self.normalize_array(vals)

        for i in range(0, self.image.shape[0], self.node_size):
            for z in range(0, self.image.shape[1], self.node_size):
                if vals[counter] > np.average(vals):
                    heat_map[i:(i + self.node_size), z:(z + self.node_size)] = 255 * self.node_averages[counter] * vals[counter]

                counter += 1
        #heat_map = gaussian_filter(heat_map, sigma=20)
        self.view_map(heat_map, label, cls)

    def visualize_on_node_output(self, label, cls):

        vals = self.degree_matrix
        vals = self.normalize_array(vals)
        heat_map = np.copy(self.overlay)
        image = np.copy(self.image)
        counter = 0
        #vals[vals < 0.75] = 0
        print('Average: {}'.format(np.average(vals)))
        vals = self.normalize_array(vals)
        #vals = self.z_score(vals)

        for i in range(0, self.image.shape[0], self.node_size):
            for z in range(0, self.image.shape[1], self.node_size):

                heat_map[i:(i + self.node_size), z:(z + self.node_size)] = 255 * self.node_output_averages[counter]

                counter += 1
        #heat_map = gaussian_filter(heat_map, sigma=20)
        self.view_map(heat_map, label, cls)

    def count_in_matrix(self):
        vals = self.degree_matrix
        vals = self.normalize_array(vals)
        heat_map = np.copy(self.overlay)

        counter = 0

        vals[vals < 0.75] = 0
        non_zero = np.count_nonzero(vals)

        return vals, non_zero

    def find_node(self, coord_arr):
        h, w, c = self.image.shape
        coords = coord_arr[0]
        x = int(coords[0])
        y = int(coords[1])
        y_level = y // self.node_size
        x_level = x // self.node_size
        node = (y_level * self.node_dim) + x_level
        return node

    def interact_with_image(self):
        for i in range(0, 5):
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            coords = []
            def onclick(event):
                x, y = event.xdata, event.ydata

                coords.append((x, y))
                plt.close(fig)
                if len(coords) == 2:
                    fig.canvas.mpl_disconnect(cid)


            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.imshow(self.image)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.show()

            node = self.find_node(coords)
            self.mark_image_grid(node)





    def mark_heat_map(self, heat_map, node):
        counter = 0
        for i in range(0, self.image.shape[0], self.node_size):
            for z in range(0, self.image.shape[1], self.node_size):
                if counter == node:
                    print('[INFO] marking grid at {}'.format(node))
                    heat_map[i:(i+self.node_size), z:(z+self.node_size)] = 0.5
                counter += 1

        return heat_map

    def visualize_graphs(self):
        edges = np.copy(self.edges)
        edge_weights = np.copy(self.edge_weights)
        remove_edges = np.where(edge_weights == 0)[0]
        curr = edge_weights[edge_weights != 0]
        arr = []
        for count, edge in enumerate(edges):
            if count in remove_edges:
                continue
            else:
                arr.append(edge)

        edge_weights = torch.FloatTensor(curr)
        edges = torch.LongTensor(arr)
        ### generate nodes
        nodes = torch.randn(81, 1)
        ### convert edges to tensor
        ### create data object
        data = Data(x=nodes, edge_index=edges.t().contiguous(), edge_weights=edge_weights.t().contiguous())
        ng = torch_geometric.utils.to_networkx(data, to_undirected=True)
        print(ng)
        nx.draw(ng)
        plt.show()

    def view_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, cls=None):

        if cls != None:
            im = Image.fromarray(np.uint8(self.image))
            im.save(config.gvm_out_path + str(path) + '_' + cls + '.png')
        else:
            im = Image.fromarray(np.uint8(self.image))
            im.save(config.gvm_out_path + str(path) + '.png')


    def view_map(self, heat_map, path=None, cls=None):
        image = self.image
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.imshow(heat_map, alpha=0.3, cmap='jet')
        if path != None and config.save_figures:
            plt.savefig(config.gvm_out_path + str(path) + '_heat_' + cls + '.png',bbox_inches='tight', pad_inches=0)
            self.save_image(path, cls)

        plt.show()


    def view_node(self, image, heat_map, path=None, cls=None):
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        def close(event):
            plt.close(fig)
            fig.canvas.mpl_disconnect(cid)

        cid = fig.canvas.mpl_connect('button_press_event', close)
        plt.imshow(image)
        plt.imshow(heat_map, alpha=0.3, cmap='jet')
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        if path != None and config.save_figures:
            plt.savefig(config.gvm_out_path + str(path) + '_heat_' + cls + '.png',bbox_inches='tight', pad_inches=0)
            self.save_image(path, cls)
        plt.show()





