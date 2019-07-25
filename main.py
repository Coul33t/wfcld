import numpy as np
from PIL import Image
import random as rn
import itertools

DEBUG = False

offsets = {'top': (0, -1), 
		   'top-right': (1, -1),
		   'right': (1, 0),
		   'bottom-right': (1, 1),
		   'bottom': (0, 1),
		   'bottom-left': (-1, 1),
		   'left': (-1, 0),
		   'top-left': (-1, -1)}

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
			for y, col in enumerate(row):
				new_row.append(list(img_tmp[x][y][0:3]))
			self.array.append(new_row)

		self.size = [len(self.array[0]), len(self.array[1])]



class Pixel:
	def __init__(self, value):
		self.value = value
		self.neighbors = {'top': [], 
						  'top-right': [],
						  'right': [],
						  'bottom-right': [],
						  'bottom': [],
						  'bottom-left': [],
						  'left': [],
						  'top-left': []}

	def add_neighbors(self, neighbors_dict):
		self.neighbors = neighbors_dict

class Constraints:
	def __init__(self, path_to_img):
		self.img = MyImage(path_to_img)
		self.pixels_list = []

	def add_pixel(self, pixel):
		if pixel.value not in [x.value for x in self.pixels_list]:
			self.pixels_list.append(pixel)
		else:
			for x in self.pixels_list:
				if x.value == pixel.value:
					for k, v in pixel.neighbors.items():
						# Todo: probabilities instead of appending only if doesn't exists (" not in ")
						if v and v[0] not in x.neighbors[k]:
							x.neighbors[k].extend(v)

	def get_neighbors(self, origin_x, origin_y):
		neighbors_dict = {'top': [], 
						  'top-right': [],
						  'right': [],
						  'bottom-right': [],
						  'bottom': [],
						  'bottom-left': [],
						  'left': [],
						  'top-left': []}

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
						neighbors_dict['top'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == 1 and y == -1:
						neighbors_dict['top-right'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == 1 and y == 0:
						neighbors_dict['right'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == 1 and y == 1:
						neighbors_dict['bottom-right'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == 0 and y == 1:
						neighbors_dict['bottom'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == -1 and y == 1:
						neighbors_dict['bottom-left'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == -1 and y == 0:
						neighbors_dict['left'] = [self.img.array[origin_y + y][origin_x + x]]
					elif x == -1 and y == -1:
						neighbors_dict['top-left'] = [self.img.array[origin_y + y][origin_x + x]]

		return neighbors_dict


	def build_constraints(self):
		for y, rows in enumerate(self.img.array):
			for x, pixel in enumerate(rows):
				neighbors_dict = self.get_neighbors(x, y)
				pix = Pixel(pixel)
				pix.add_neighbors(neighbors_dict)
				self.add_pixel(pix)

	def get_lowest_entropy(self, possibilities_array):
		lowest_entropy = len(self.pixels_list) + 1
		x_lowest = -1
		y_lowest = -1
		possible_cells = []
		
		for x, rows in enumerate(possibilities_array):
			for y, col in enumerate(rows):
				
				if col == 'Collapsed':
					continue

				else:
					current_cell_entropy = len(col)
					if DEBUG:
							print(f'Current cell entropy: {current_cell_entropy} (vs {lowest_entropy})')
					if current_cell_entropy < lowest_entropy:
						if DEBUG:
							print('New lowest entropy')
						possible_cells.clear()
						possible_cells.append((x, y))
						lowest_entropy = current_cell_entropy
					elif current_cell_entropy == lowest_entropy:
						if DEBUG:
							print('Same entropy')
						possible_cells.append((x, y))
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
			and possibilities_array[new_x][new_y] != 'Collapsed'):
				possibilities_array[new_x][new_y] = [x for x in possibilities_array[new_x][new_y] if x.value in v]
				if possibilities_array[new_x][new_y] == []:
					return False

		return True

	def build_new(self, output_w, output_h):
		
		possibilities_array = [[self.pixels_list for x in range(output_w)] for y in range(output_h)]
		tmp_new_image = [[None for x in range(output_w)] for y in range(output_h)]
		
		while True:
			
			coordinates = self.get_lowest_entropy(possibilities_array)

			if any(x == [] for x in itertools.chain(*possibilities_array)):
				breakpoint()
			
			if coordinates is None:
				no_contradiction = False

			else:
				chosen_cell = rn.choice(possibilities_array[coordinates[0]][coordinates[1]])
				tmp_new_image[coordinates[0]][coordinates[1]] = chosen_cell.value
				
				if not self.propagate(chosen_cell, coordinates, possibilities_array):
					print('Contradiction reached: aborted.')
					return None

				possibilities_array[coordinates[0]][coordinates[1]] = 'Collapsed'

				if all(x == 'Collapsed' for x in itertools.chain(*possibilities_array)):
					print('Done!')
					break

		return tmp_new_image

if __name__ == '__main__':
	print('Importing image')
	constraints = Constraints("test.png")
	print('Building constraints')
	constraints.build_constraints()
	new_img = None

	if constraints:
		for i in range(20):
			print('Building new image')
			new_img = constraints.build_new(100,100)
			if new_img:
				break

	if new_img:
		img = Image.fromarray(np.array(new_img))
		img.show()