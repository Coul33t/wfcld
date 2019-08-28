import numpy as np
from PIL import Image
import random as rn

offsets = {'top': (0, -1),
           'right': (1, 0),
           'bottom': (0, 1),
           'left': (-1, 0)}

class MyImage:
    def __init__(self, path_to_img=None, img=None):
        if path_to_img:
            self.img = Image.open("test.png")
        else:
            self.img = img
        self.img = np.array(self.img.convert('RGB'))
        self.w = len(self.img[0])
        self.h = len(self.img)

class Block:
    def __init__(self, value):
        self.value = value
        self.w = len(value[0])
        self.h = len(value)
        self.neighbors = {'top': [], 'right': [], 'bottom': [], 'left': []}
        self.frequency = 1
        self.probability = -1



class Constraints:
    def __init__(self, block_x_size, block_y_size, path_to_img=None, img=None):
        self.original_image = MyImage(path_to_img, img)
        self.block_x = block_x_size
        self.block_y = block_y_size
        self.blocks_list = []

    def add_block(self, i, j):
        """
            Adds a block (NxN) to the blocks list. It first checks if the current
            block exists in the list of blocks. If it doesn't exist in the list,
            it sets its neighbors and add the block to the list. If it does, it
            add the neighbors to the block already in the list.
        """
        # i = row
        # j = col
        # TODO: change the i,j / x,y notation to avoid further errors
        # Get the block
        block = Block(self.original_image.img[i:i+self.block_y, j:j+self.block_x])
        # Get its neigbhors
        neighbors = self.get_neighbors(i, j)
        # print(neighbors)

        # If the block doesn't exist in the list, add it
        if not any([np.array_equal(block.value, x.value) for x in self.blocks_list]):
            for k, v in neighbors.items():
                if v != []:
                    block.neighbors[k] = [v]
            self.blocks_list.append(block)

        # Else, find it in the list
        else:
            block_in_list = [x for x in self.blocks_list if np.array_equal(block.value, x.value)][0]
            # Then, for each direction, check if the neighbors exist or not
            for n_direction, n_value in neighbors.items():
                if n_value != []:
                # If there is no block in the neighbors in this direction
                    if len(block_in_list.neighbors[n_direction]) == 0:
                        block_in_list.neighbors[n_direction] = [n_value]
                    else:
                        # Else we look for this specific neighbor in the specified direction
                        try:
                            idx = next(i for i, x in enumerate(block_in_list.neighbors[n_direction]) if np.array_equal(n_value.value, x.value))
                            block_in_list.neighbors[n_direction][idx].frequency += 1
                        # If we didn't find it, we just add this new block
                        except StopIteration:
                            block_in_list.neighbors[n_direction].append(n_value)

        self.occurences_to_probabilities()


    def get_neighbors(self, i, j) -> dict:
        # i = row
        # j= col
        neighbors_dict = {'top': [],
                          'right': [],
                          'bottom': [],
                          'left': []}

        # img[[x], y1:y2] keeps the dimensions of the original array, while
        # img[x, y1:y2] doesn't (see https://stackoverflow.com/questions/2640147/preserving-the-dimensions-of-a-slice-from-a-numpy-3d-array)
        if i > 0:
            neighbors_dict['top'] = Block(self.original_image.img[[i-1], j:j+self.block_x])
        if i < self.original_image.h - self.block_y:
            neighbors_dict['bottom'] = Block(self.original_image.img[[i+self.block_y], j:j+self.block_x])
        if j > 0:
            neighbors_dict['left'] = Block(self.original_image.img[i:i+self.block_y, [j-1]])
        if i < self.original_image.w - self.block_x:
            neighbors_dict['right'] = Block(self.original_image.img[i:i+self.block_y, [j+self.block_x]])

        return neighbors_dict

    def occurences_to_probabilities(self):
        for current_block in self.blocks_list:
            for _, n_block_list in current_block.neighbors.items():

                cum_sum = 0
                for n_block in n_block_list:
                    cum_sum += n_block.frequency

                if cum_sum == 0:
                    cum_sum = 1

                else:
                    for n_block in n_block_list:
                        n_block.probability = n_block.frequency / cum_sum

    def compute_entropy(self, i, j, possibilities_array):
        entropy = len(possibilities_array[i][j])
        print(f'{i}, {j}')
        if i > 0:
            entropy += len(possibilities_array[i-1][j])
        if i < len(possibilities_array) - 1:
            entropy += len(possibilities_array[i+1][j])
        if j > 0:
            entropy += len(possibilities_array[i][j-1])
        if j < len(possibilities_array[0]) - 1:
            entropy += len(possibilities_array[i][j+1])
        return entropy

    def select_lower_entropy(self, possibilities_array):
        row_col = (-1,-1)
        lowest_entropy = 9999999

        for i in range(len(possibilities_array)):
            for j in range(len(possibilities_array[i])):
                current_entropy = self.compute_entropy(i, j, possibilities_array)

                if current_entropy < lowest_entropy:
                    lowest_entropy = current_entropy
                    row_col = (i, j)

                if current_entropy == lowest_entropy:
                    if rn.random() > 0.5:
                        row_col = (i, j)

        return row_col

    def create_new_image(self, output_size):
        if not (output_size[0] / self.block_x).is_integer() or not (output_size[1] / self.block_y).is_integer():
            print(f'ERROR: please specify an output size that is a multiple of the blocks size.')
            print(f'(Current block size : {self.block_x}x{self.block_y}, current output size: {output_size[0]}x{output_size[1]})')
            return

        possibilities_array = [[self.blocks_list for x in range(int(output_size[1] / self.block_y))] for y in range(int(output_size[0] / self.block_x))]
        tmp_image = np.zeros(shape=output_size)

        contradiction = False
        first = True

        while not contradiction:
            if first == True:
                first = False

            else:
                row_col = self.select_lower_entropy(possibilities_array)
                breakpoint()



    def display_blocks_list(self):
        print(f'Images to display: {len(self.blocks_list)}')
        all_blocks = None

        for block in self.blocks_list:
            if all_blocks is None:
                all_blocks = block.value
                continue
            else:
                all_blocks = np.concatenate((all_blocks, block.value), axis=1)

        img = Image.fromarray(all_blocks)
        img.show()

    # def display_blocks_and_prob(self):
    #     all_blocks = None

    #     for block in self.blocks_list:
    #         if all_blocks is None:
    #             tmp_h = []
    #             tmp_v = []
    #             breakpoint()
    #             for k, v in block.neighbors.items():
    #                 if k == 'top' or k == 'bottom':
    #                     tmp_v.extend(x.value for x in v)
    #                 else:
    #                     tmp_h.extend(x.value for x in v)

    #             tmp_h = np.hstack(tmp_h)
    #             tmp_v = np.vstack(tmp_v)
    #             breakpoint()
    #             all_blocks = np.vstack((block.value, tmp_v))
    #             all_blocks = np.hstack((all_blocks, tmp_h))
    #             continue
    #         else:
    #             all_blocks = np.concatenate((all_blocks, block.value), axis=1)

    #     img = Image.fromarray(all_blocks)
    #     img.show()



if __name__ == '__main__':
    print('Importing image')

    cons = Constraints(2, 2, path_to_img="test.png")

    for i in range(cons.original_image.h - cons.block_y):
        for j in range(cons.original_image.w - cons.block_x):
            # print(f'{i}, {j}')
            cons.add_block(i, j)

    # cons.display_blocks_list()
    cons.create_new_image((20,20))

