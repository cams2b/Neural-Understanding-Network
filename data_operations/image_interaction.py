import numpy as np


from data_operations.image_visualizaiton import *

class image_interaction(image_visualization):
    def __init__(self, image, node_dim, edges=None, edge_weights=None, node_values=None, node_output=None):
        super(image_interaction, self).__init__(image, node_dim, edges=edges, edge_weights=edge_weights, node_values=node_values, node_output=node_output)



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
            return node
            #self.mark_image_grid(node)

    def visualize_multi_node_neighborhood(self, node_count_selector=4, label=None, cls=None):
        combined_heat_map = []
        h, w, c = self.image.shape
        combination = np.zeros((h, w))
        arr = []
        for i in range(node_count_selector):
            current_node = self.interact_with_image()
            combined_heat_map.append(current_node)
            image, heat_map = self.mark_image_grid(current_node)
            self.view_node(image, heat_map, label, cls + '_grid')
            #heat_map = self.mark_heat_map(heat_map, current_node)
            arr.append(heat_map)
            if i == 0:
                combination = combination + heat_map

            else:
                combination = combination + heat_map
                combination = gaussian_filter(combination, sigma=10)
        arr = np.array(arr)
        #print(arr.shape)
        arr = np.max(arr, axis=0)

        print('[INFO] visualizing nodes: {}'.format(combined_heat_map))
        combination = self.normalize_array(combination)
        #combination = self.z_score(combination)
        combination = gaussian_filter(combination, sigma=25)
        self.view_node(self.image, combination, label, cls + '_gaussian')



