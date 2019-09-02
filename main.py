# Commente ton putain de code

import numpy as np
from PIL import Image
import random as rn
import itertools

DEBUG = False

offsets = {'top': (0, -1),
           'right': (1, 0),
           'bottom': (0, 1),
           'left': (-1, 0)}

GIF = []

class MyImage:
    def __init__(self, path_to_img):
        self.array = []
        self.size = [-1, -1]
        self.load_image(path_to_img)

    def load_image(self, path_to_img):
        """
            Since the array representing the image is badly constructed (imo),
            I'm putting it in the shape of a 2D array, where each value = (r, g, b)
        """
        img_tmp = Image.open(path_to_img)
        img_tmp = np.array(img_tmp.convert('RGB'))

        for x, row in enumerate(img_tmp):
            new_row = []
            for y, _ in enumerate(row):
                new_row.append(list(img_tmp[x][y][0:3]))
            self.array.append(new_row)

        self.size = [len(self.array[0]), len(self.array[1])]



class Pixel:
    def __init__(self, value):
        self.value = value
        self.neighbors = {'top': [],
                          'right': [],
                          'bottom': [],
                          'left': []}
        self.proba = -1
        self.weight = -1

    def add_neighbors(self, neighbors_dict):
        for k,v in neighbors_dict.items():
            if v:
                v.proba = 1
                self.neighbors[k] = [v]

class Constraints:
    def __init__(self, path_to_img):
        self.img = MyImage(path_to_img)
        self.pixels_list = []

    def add_pixel(self, pixel):
        if pixel.value not in [x.value for x in self.pixels_list]:
            self.pixels_list.append(pixel)
        else:
            # When we have found the pixel in the pixels list
            pixel_in_list = [x for x in self.pixels_list if x.value == pixel.value][0]

            for k, v in pixel.neighbors.items():
                # If there is a value for the neighbor
                if v:
                    # We try to find an existing pixel in the neighbor list
                    for n_pixel in pixel_in_list.neighbors[k]:
                        if n_pixel.value == v[0].value:
                            n_pixel.weight += 1
                            break
                    # If we didn't find it, we add a new one
                    else:
                        pixel_in_list.neighbors[k].append(v[0])

    def occurences_to_probabilities(self):
        for current_pixel in self.pixels_list:
            for _, pixel_list in current_pixel.neighbors.items():

                cum_sum = 0
                for pixel in pixel_list:
                    cum_sum += pixel.weight

                # TODO: pixel???
                if cum_sum == 0:
                    pixel.proba = 0
                else:
                    for pixel in pixel_list:
                        pixel.proba = pixel.weight / cum_sum

    def get_neighbors(self, origin_x, origin_y):
        neighbors_dict = {'top': [],
                          'right': [],
                          'bottom': [],
                          'left': []}

        x_val = []
        y_val = []

        if origin_x > 0:
            x_val.append(-1)
        if origin_y > 0:
            y_val.append(-1)

        x_val.append(0)
        y_val.append(0)

        if origin_x < self.img.size[0] - 1:
            x_val.append(1)
        if origin_y < self.img.size[1] - 1:
            y_val.append(1)

        for y in y_val:
            for x in x_val:
                if not (x == 0 and y == 0):
                    if x == 0 and y == -1:
                        neighbors_dict['top'] = Pixel(self.img.array[origin_y + y][origin_x + x])
                    elif x == 1 and y == 0:
                        neighbors_dict['right'] = Pixel(self.img.array[origin_y + y][origin_x + x])
                    elif x == 0 and y == 1:
                        neighbors_dict['bottom'] = Pixel(self.img.array[origin_y + y][origin_x + x])
                    elif x == -1 and y == 0:
                        neighbors_dict['left'] = Pixel(self.img.array[origin_y + y][origin_x + x])

        return neighbors_dict

    def build_constraints(self):
        for y, rows in enumerate(self.img.array):
            for x, pixel in enumerate(rows):
                neighbors_dict = self.get_neighbors(x, y)
                pix = Pixel(pixel)
                pix.add_neighbors(neighbors_dict)
                self.add_pixel(pix)

        self.occurences_to_probabilities()

    def get_lowest_entropy(self, possibilities_array):
        lowest_entropy = 999999999
        possible_cells = []

        for row_idx, rows in enumerate(possibilities_array):
            for col_idx, col in enumerate(rows):

                if col == 'Collapsed':
                    continue

                else:
                    # TODO: true entropy computation
                    current_cell_entropy = len(col) + sum([len(possibilities_array[row_idx][col_idx]) for i in (-1,0,1) for j in (-1,0,1) if i != 0 or j != 0])
                    if DEBUG:
                        print(f'Current cell entropy: {current_cell_entropy} (vs {lowest_entropy})')
                    if current_cell_entropy < lowest_entropy:
                        if DEBUG:
                            print(f'New lowest entropy ({possibilities_array[row_idx][col_idx]})')
                        possible_cells.clear()
                        possible_cells.append((row_idx, col_idx))
                        lowest_entropy = current_cell_entropy
                    elif current_cell_entropy == lowest_entropy:
                        if DEBUG:
                            print(f'Same entropy ({possibilities_array[row_idx][col_idx]})')
                        possible_cells.append((row_idx, col_idx))
                    else:
                        if DEBUG:
                            print('Higher entropy')

        if len(possible_cells) > 1:
            return rn.choice(possible_cells)
        elif len(possible_cells) == 1:
            return possible_cells[0]
        else:
            print('Contradiction reached: aborted.')
            return None

    def propagate(self, cell, coordinates, possibilities_array):
        x = coordinates[0]
        y = coordinates[1]
        for k, v in cell.neighbors.items():
            new_x = x + offsets[k][0]
            new_y = y + offsets[k][1]

            if (v != []
            and new_y < len(possibilities_array)
            and new_x < len(possibilities_array[0])
            and possibilities_array[new_y][new_x] != 'Collapsed'):
                possibilities_array[new_y][new_x] = [x for x in possibilities_array[new_y][new_x] if x.value in [yo.value for yo in v]]
                if possibilities_array[new_y][new_x] == []:
                    return False

        return True

    def build_new(self, output_w, output_h):

        possibilities_array = [[self.pixels_list for x in range(output_w)] for y in range(output_h)]
        tmp_new_image = [[None for x in range(output_w)] for y in range(output_h)]
        idx = 0

        while True:

            row_col = self.get_lowest_entropy(possibilities_array)
            if row_col is None:
                return None

            if any(x == [] for x in itertools.chain(*possibilities_array)):
                breakpoint()

            else:
                chosen_cell = rn.choices(population=possibilities_array[row_col[0]][row_col[1]],
                                        weights=[x.proba for x in possibilities_array[row_col[0]][row_col[1]]])[0]
                print(f'{chosen_cell.value}')
                tmp_new_image[row_col[0]][row_col[1]] = chosen_cell.value

                if not self.propagate(chosen_cell, row_col, possibilities_array):
                    print('Contradiction reached: aborted.')
                    return None

                possibilities_array[row_col[0]][row_col[1]] = 'Collapsed'
                self.export_tmp_image(tmp_new_image, idx)
                idx += 1

                if all(x == 'Collapsed' for x in itertools.chain(*possibilities_array)):
                    print('Done!')
                    break

        return tmp_new_image

    def export_tmp_image(self, img, idx):
        for r, row in enumerate(img):
            for c, col in enumerate(row):
                if not col:
                    img[r][c] = [255,0,128]
        GIF.append(Image.fromarray(np.array(img).astype(np.uint8)))

if __name__ == '__main__':
    print('Importing image')
    constraints = Constraints("test.png")
    print('Building constraints')
    constraints.build_constraints()
    new_img = None

    if constraints:
        for i in range(20):
            print('Building new image')
            new_img = constraints.build_new(32,32)
            if new_img:
                break

    if new_img:
        GIF[0].save('result.gif', format='GIF', append_images=GIF[1:], save_all=True, duration=100, loop=0)
        if new_img:
            img = Image.fromarray(np.array(new_img))
            img.show()