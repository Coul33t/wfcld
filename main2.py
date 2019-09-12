# --------------
# PYTHON IMPORTS
# --------------
import random as rn
import itertools
import argparse
from enum import auto, Enum
# --------------

# ---------------
# LIBRARY IMPORTS
# ---------------
import numpy as np
from PIL import Image
# Used to export the process to a video file
# (GIF compress too much for small images)
import cv2
# ---------------

offsets = {'top': (0, -1),
           'right': (1, 0),
           'bottom': (0, 1),
           'left': (-1, 0)}

def parse_args():
    """ Arguments parser. """

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Toggle debug mode Debug mode", action='store_true')
    parser.add_argument("-v", "--video", help="Export the creation process to a video file", action='store_true')
    parser.add_argument("-g", "--gif", help="Export the creation process to a gif file", action='store_true')
    args = parser.parse_args()

    return args

class MyImage:
    """
        A class containing an image (numpy array), its width and its height
        for practical purpose.
    """
    def __init__(self, path_to_img=None, img=None):
        if img:
            self.img = img
        else:
            self.img = Image.open(path_to_img)

        self.img = np.array(self.img.convert('RGB'))
        self.w = len(self.img[0])
        self.h = len(self.img)

class Block:
    """
        A NxM block in the image. This class is used to store all the possible
        NxM blocks contained into the original image, and also the neighbors
        (Nx1 for the top and bottom neighbors, 1xM for the left and right ones).
        It also stores the neighbors themselves and the frequency of this block.
    """
    def __init__(self, value):
        self.value = value
        self.w = len(value[0])
        self.h = len(value)
        self.neighbors = {'top': [], 'right': [], 'bottom': [], 'left': []}
        self.frequency = 1
        self.is_border =  {'top-left': 0,
                           'top-right': 0,
                           'bottom-right': 0,
                           'bottom-left': 0,
                           'top': 0,
                           'right': 0,
                           'bottom': 0,
                           'left': 0}

    def set_border(self, i, j, b_x, b_y, w, h):
        if i == 0 and j == 0:
            self.is_border['top-left'] = 1
        elif i == 0 and j == w-b_x:
            self.is_border['top-right'] = 1
        elif i == h-b_y and j == w-b_x:
            self.is_border['bottom-right'] = 1
        elif i == h-b_y and j == 0:
            self.is_border['bottom-left'] = 1
        elif i == 0:
            self.is_border['top'] = 1
        elif j == 0:
            self.is_border['right'] = 1
        elif i == h-b_y:
            self.is_border['bottom'] = 1
        elif j == w-b_x:
            self.is_border['left'] = 1



class Constraints:
    """
        This is the main class of the file. It consists of the original image
        (MyImage class), the size of the blocks (x and y), and the list of all
        differents blocks contained in the original image. It also offers a set
        of methods to compute everything needed for wave-function collapsing.
    """
    def __init__(self, block_x_size, block_y_size, path_to_img=None, img=None, debug=False):
        self.path = '/'.join(path_to_img.split('/')[0:-1])
        self.image_name = path_to_img.split('/')[-1].split('.')[0]
        self.original_image = MyImage(path_to_img, img)
        self.block_x = block_x_size
        self.block_y = block_y_size
        self.blocks_list = []
        self.new_image = None
        self.gif_array = []
        self.debug = debug

    def add_block(self, i, j):
        """
            Adds a block (NxM) to the blocks list. It first checks if the current
            block exists in the list of blocks. If it doesn't exist in the list,
            it sets its neighbors and add the block to the list. If it does, it
            add the neighbors to the block already in the list.
        """
        # i = row
        # j = col
        # TODO: change the i,j / x,y notation to avoid further errors
        # Get the block

        block = Block(self.original_image.img[i:i+self.block_y, j:j+self.block_x])

        block.set_border(i, j, self.block_x, self.block_y, self.original_image.w, self.original_image.h)
        # Get its neigbhors
        neighbors = self.get_neighbors(i, j)

        # If the block doesn't exist in the list, add it
        if not any([np.array_equal(block.value, x.value) for x in self.blocks_list]):
            for k, v in neighbors.items():
                if v != []:
                    block.neighbors[k] = [v]
            self.blocks_list.append(block)

        # Else, find it in the list
        else:
            block_in_list = [x for x in self.blocks_list if np.array_equal(block.value, x.value)][0]

            # adds the values of the dictionnary key-wise
            # (https://stackoverflow.com/questions/45713887/add-values-from-two-dictionaries)
            block_in_list.is_border = {key: block_in_list.is_border.get(key, 0) + block.is_border.get(key, 0)
                                       for key in set(block_in_list.is_border) | set(block.is_border)}

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

    def get_neighbors(self, i, j) -> dict:
        """
            This method returns a dictionnary of neighbors, from a set of
            coordinates (the N and M values are stored into the class, so there's
            no need to pass them to this function).
        """
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
            neighbors_dict['bottom'] = Block(self.original_image.img[[i+self.block_y-1], j:j+self.block_x])
        if j > 0:
            neighbors_dict['left'] = Block(self.original_image.img[i:i+self.block_y, [j-1]])
        if i < self.original_image.w - self.block_x:
            neighbors_dict['right'] = Block(self.original_image.img[i:i+self.block_y, [j+self.block_x-1]])

        return neighbors_dict


    def compute_entropy(self, i, j, possibilities_array) -> float:
        """
            CURRENTLY UNUSED (because it makes the algorithm systematically
            select the corners first, then the borders, etc.)
            Computes the entropy for a cell. It uses the following formula:
            cell_entropy = current_cell_entropy +
                           top_cell_entropy +
                           bottom_cell_entropy +
                           left_cell_entropy +
                           right_cell_entropy
            The entropy is defined as the sum of the remaining possibilities
            for the concerned block.
        """
        entropy = len(possibilities_array[i][j])
        # print(f'{i}, {j}')
        if i > 0:
            entropy += len(possibilities_array[i-1][j])
        if i < len(possibilities_array) - 1:
            entropy += len(possibilities_array[i+1][j])
        if j > 0:
            entropy += len(possibilities_array[i][j-1])
        if j < len(possibilities_array[0]) - 1:
            entropy += len(possibilities_array[i][j+1])
        return entropy

    def select_lower_entropy(self, possibilities_array) -> dict:
        """
            Select the cell with the lowest entropy in the possibilities array.
            If a cell has a lower entropy than the current lowest value, then it
            become the new lowest entropy cell. If it's the same value, there's
            a 50% chance that the current cell become the new lowest entropy cell.
        """
        # It stores the lowest entropy and all the cells that match this value.
        # It used to be
        #   if current_entropy < lowest_entropy: 50% to take the current cell
        # But this code would actually hugely favours the last cells of the array
        entropy_and_position = {'entropy': 9999999, 'position': [(-1, -1)]}

        for i in range(len(possibilities_array)):
            for j in range(len(possibilities_array[i])):

                if possibilities_array[i][j] == "COLLAPSED":
                    continue

                # current_entropy = self.compute_entropy(i, j, possibilities_array)
                current_entropy = len(possibilities_array[i][j])

                if current_entropy < entropy_and_position['entropy']:
                    entropy_and_position['entropy'] = current_entropy
                    entropy_and_position['position'] = [(i, j)]

                if current_entropy == entropy_and_position['entropy']:
                    entropy_and_position['position'].append((i, j))

        row_col = rn.choice(entropy_and_position['position'])

        # print(f"entropy: {entropy_and_position['entropy']}\nrow_col: {row_col}\nnumber of candidates: {len(entropy_and_position['position'])}")

        return row_col

    def compare_matrix_to_submatrix(self, direction, matrix, submatrix) -> bool:
        # if self.debug:
        #     print('new matrixes to test')
        #     print(f'Matrix = {matrix}')
        #     print(f'Submatrix = {submatrix}')

        # Matrix is [[[]]] at first, need to be [[]]
        # TODO: better explanation
        if submatrix.shape[0] == 1:
            submatrix = submatrix[0]
        elif submatrix.shape[1] == 1:
            submatrix = submatrix[:,0]

        if direction == 'top':
            if np.array_equal(matrix[-1,:], submatrix):
                return True
        elif direction == 'bottom':
            if np.array_equal(matrix[0,:], submatrix):
                return True
        elif direction == 'left':
            if np.array_equal(matrix[:,-1], submatrix):
                return True
        elif direction == 'right':
            if np.array_equal(matrix[:,0], submatrix):
                return True

        return False

    def propagate(self, chosen_cell, row_col, possibilities_array):
        """
            Once a celle has been collapsed, this function propagate the
            possibilities to the adjacent ones, regarding to the neighbors
            list of the current cell. If a contradiction is reached, it returns
            False.
        """
        # TODO: probas for the propagated cells
        # How to : add frequencies
        x = row_col[1]
        y = row_col[0]

        if self.debug:
            print('-----------\nPropagation\n-----------')

        for k, v in chosen_cell.neighbors.items():

            new_x = x + offsets[k][0]
            new_y = y + offsets[k][1]

            if self.debug:
                print(f'{k} at {new_x} {new_y}')

            # -1 is a valid index in a Python list...
            if (v != []
            and new_x < len(possibilities_array)
            and new_y < len(possibilities_array[0])
            and new_x >= 0
            and new_y >= 0
            and possibilities_array[new_y][new_x] != "COLLAPSED"):

                to_delete = []

                has_legal_block = False
                for i, possible_block in enumerate(possibilities_array[new_y][new_x]):
                    has_one_match = False
                    print('begin')
                    for neigh in v:
                        if self.compare_matrix_to_submatrix(k, possible_block.value, neigh.value):
                            possible_block.frequency += 1
                            print(f'has true for {neigh}')
                            has_legal_block = True
                            has_one_match = True

                    if not has_one_match:
                        to_delete.append(i)
                        breakpoint()

                if not has_legal_block:
                    if self.debug:
                        print(f'Contradiction at {new_y} {new_x}')
                        print(f'Original block at {row_col[1]} {row_col[0]}')
                        print(f'Original block value: {chosen_cell.value}')
                        print(f'Original block neighbors: {chosen_cell.neighbors}')
                        breakpoint()
                    return False

                to_delete.sort(reverse=True)

                for i in to_delete:
                    del possibilities_array[new_y][new_x][i]

            else:
                if self.debug:
                    print(f'{new_x} < {len(possibilities_array)} (nb cols): {new_x < len(possibilities_array)}')
                    print(f'{new_y} < {len(possibilities_array[0])} (nb rows): {new_y < len(possibilities_array[0])}')
                    print(f'{new_x} > 0: {new_x > 0}')
                    print(f'{new_y} > 0: {new_y > 0}')
                    if (0 <= new_x < len(possibilities_array)) and (0 <= new_y < len(possibilities_array[0])):
                            print(f'{possibilities_array[new_y][new_x]} is not collapsed: {possibilities_array[new_y][new_x] != "COLLAPSED"}')

        return True

    def compute_corner_possibilities(self, direction, possibilities_array):
        row = None
        col = None

        if 'top' in direction:
            row = 0
        if 'bottom' in direction:
            row = -1
        if 'left' in direction:
            col = 0
        if 'right' in direction:
            col = -1

        to_delete = []

        for i, block in enumerate(possibilities_array[row][col]):
            if block.is_border[direction] == 0:
                to_delete.append(i)

        to_delete.sort(reverse=True)

        for idx in to_delete:
            del possibilities_array[row][col][idx]



    def compute_border_possibilities(self, possibilities_array):
        # Start with corners
        # TODO: bug somewhere around here
        self.compute_corner_possibilities('top-left', possibilities_array)
        self.compute_corner_possibilities('top-right', possibilities_array)
        self.compute_corner_possibilities('bottom-right', possibilities_array)
        self.compute_corner_possibilities('bottom-left', possibilities_array)

        for i in range(1, len(possibilities_array)-1):
            to_delete = []
            for j, block in enumerate(possibilities_array[i][0]):
                if block.is_border['left'] == 0:
                    to_delete.append(j)

            to_delete.sort(reverse=True)

            for idx in to_delete:
                del possibilities_array[i][0][idx]

            to_delete = []
            for j, block in enumerate(possibilities_array[i][-1]):
                if block.is_border['right'] == 0:
                    to_delete.append(j)

            to_delete.sort(reverse=True)
            for idx in to_delete:
                del possibilities_array[i][-1][idx]


        for j in range(1, len(possibilities_array[0])-1):
            to_delete = []
            for i, block in enumerate(possibilities_array[0][j]):
                if block.is_border['top'] == 0:
                    to_delete.append(i)

            to_delete.sort(reverse=True)
            for idx in to_delete:
                del possibilities_array[0][j][idx]

            to_delete = []
            for i, block in enumerate(possibilities_array[-1][j]):
                if block.is_border['bottom'] == 0:
                    to_delete.append(i)

            to_delete.sort(reverse=True)
            for idx in to_delete:
                del possibilities_array[-1][j][idx]




    def create_new_image(self, output_size):
        """
            Creates a new image from the computed constraints. It is possible
            that the function reach a contradiction: it means that a cell can't
            have any value because of its neighbors. In this case, the generation
            failed.
        """

        print('Creating new image')
        # If the size of the output image isn't a multiple (in both x and y)
        # of the blocks size, abort.
        # TODO: change the size of the output image so that it matches the closest
        # possible value.
        if not (output_size[0] / self.block_x).is_integer() or not (output_size[1] / self.block_y).is_integer():
            print(f'ERROR: please specify an output size that is a multiple of the blocks size.')
            print(f'(Current block size : {self.block_x}x{self.block_y}, current output size: {output_size[0]}x{output_size[1]})')
            return

        # A list containing all the possibles values for each block. Each processed
        # tiles get replaced by "COLLAPSED" in this array
        # TODO: numpy array to be consistent for accessing values?
        possibilities_array = [[self.blocks_list.copy() for x in range(int(output_size[1] / self.block_y))] for y in range(int(output_size[0] / self.block_x))]
        # Initialise a new matrix: N*ixM*jx3 (3 = RGB)
        # WARNING: this numpy array does not have the same size as the
        # possibilities_array:
        #     - posibilities_array has one cell for one block
        #     - new_image has one cell for one pixel
        self.new_image = np.full(shape=(output_size[0], output_size[1], 3), fill_value=[255,0,128])
        counter = 0

        #TODO: first pass of possible values for border
        self.compute_border_possibilities(possibilities_array)

        chosen_cell = None

        while True:
            counter += 1
            if self.debug:
                print(f'Processing tile number {counter}')

            # Get the top-left coordinates of the block with the lowest entropy
            row_col = self.select_lower_entropy(possibilities_array)

            if self.debug:
                print(f'Row/Col = {row_col}')

            # Collapse (choose a value for this block)
            chosen_cell = rn.choices(population=possibilities_array[row_col[0]][row_col[1]],
                                    weights=[x.frequency for x in possibilities_array[row_col[0]][row_col[1]]],
                                    k=1)[0]

            if self.debug:
                print(f'Chosen cell: {chosen_cell.value}')

            # Put this block into the new image
            self.new_image[row_col[0]*self.block_x:row_col[0]*self.block_x+self.block_x,
                        row_col[1]*self.block_y:row_col[1]*self.block_y+self.block_y] = chosen_cell.value

            possibilities_array[row_col[0]][row_col[1]] = 'COLLAPSED'
            self.gif_array.append(np.copy(self.new_image))

            if all(x == 'COLLAPSED' for x in itertools.chain(*possibilities_array)):
                print('Done!')
                return 'Done'

            # Unlucky boi
            if not self.propagate(chosen_cell, row_col, possibilities_array):
                print('Contradiction reached: aborted.')
                if self.debug:
                    Image.fromarray(self.new_image.astype(np.uint8)).show()
                    breakpoint()
                return counter

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
        final_name = 'all_blocks_' + self.image_name + '.png'
        img.save(final_name, "PNG")


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

    def to_video(self, gif=None):
        final_gif_array = self.gif_array
        if gif:
            final_gif_array = gif

        fps = 2
        size = (self.new_image.shape[0], self.new_image.shape[1])
        out = cv2.VideoWriter('final_process.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for img in final_gif_array:
            out.write(img.astype(np.ubyte))
        out.release()

    def to_gif(self, gif=None):
        final_gif_array = self.gif_array
        if gif:
            final_gif_array = gif

        for i, _ in enumerate(final_gif_array):
            final_gif_array[i] = Image.fromarray(final_gif_array[i].astype(np.uint8))

        final_gif_array[0].save('final_process.gif', format='GIF', append_images=final_gif_array[1:], save_all=True, duration=500, loop=0)

def main(args):
    print('Importing image')

    cons = Constraints(8, 8, path_to_img="test_images/subtest2.png", debug=args.debug)

    print('Building constraints')
    for i in range(cons.original_image.h - cons.block_y + 1):
        for j in range(cons.original_image.w - cons.block_x + 1):
            cons.add_block(i, j)

    cons.display_blocks_list()

    i = 0
    highest_counter = -1

    tmp_gif = []
    tmp_img = None

    while i < 10:
        cons.gif_array = []

        new_image = cons.create_new_image((40, 40))

        if isinstance(new_image, int):
            print(f'Number of iterations: {new_image}')
            if new_image > highest_counter:
                highest_counter = new_image
                tmp_gif = cons.gif_array
                tmp_img = cons.new_image

        elif new_image == 'Done':
            break

        i += 1

    if new_image == 'Done':
        Image.fromarray(cons.new_image.astype(np.uint8)).show()
        Image.fromarray(cons.new_image.astype(np.uint8)).save("final.png", "PNG")
        if args.video:
            cons.to_video()
        if args.gif:
            cons.to_gif()
    else:
        Image.fromarray(tmp_img.astype(np.uint8)).show()
        Image.fromarray(tmp_img.astype(np.uint8)).save("best.png", "PNG")
        if args.video:
            cons.to_video(tmp_gif)
        if args.gif:
            cons.to_gif(tmp_gif)

if __name__ == '__main__':
    args = parse_args()
    main(args)


