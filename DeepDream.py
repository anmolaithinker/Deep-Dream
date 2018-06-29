import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
from PIL import Image
import os
from scipy.ndimage.filters import gaussian_filter

from IPython.display import Image, display

# Python 2.7
import urllib
import sys
import zipfile
import tarfile



########################################################################################################

def PrintDownloadProgress(count , block_size , total_size):
  pct_complete = float(count * block_size) / total_size
  msg = "\r- Download progress: {0:.1%}".format(pct_complete)
  sys.stdout.write(msg)
  sys.stdout.flush()



########################################################################################################

def downloadInception5H(data_dir , path_graph_def):
  data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
  data_dir = data_dir
  file_name = data_url.split('/')[-1]
  file_path = os.path.join(data_dir , file_name)
  if not os.path.exists(file_path):
    if not os.path.exists(data_dir):
      os.create(data_dir)
  
   # Retrieving data from file
    file_path , _ = urllib2.urlopen(url = data_url , )
    urllib.urlretrieve(url = data_url , filename = filepath , reporthook = PrintDownloadProgress)
  
    print ()
    print (" Downloading finished !! Now Extracting the files ......")
  
    # Extraction starts
  
    # Zip File
    if file_path.endswith('.zip'):
      zipfile.ZipFile(file = file_path , mode = 'r').extractall(data_dir)
  
    # Tar File
    elif file_path.endswith('.tar.gz' , '.tgz'):
      tarfile.open(name = file_path , mode = "r:gz").extractall(data_dir)
  
    print ('-' * 50)
    print ('Ohh Yeah Done Extracting --->')
  
  else:
    print ('-' * 50)
    print ('Data already downloaded and extracted !!!!! ')
    print ('-' * 50)



########################################################################################################

class inception5h:
  	
  	path_graph_def = "tensorflow_inception_graph.pb"

  	# Commonly used layers in inception

    layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
                   'mixed3a', 'mixed3b',
                   'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
                   'mixed5a', 'mixed5b']


    tensor_name_input_image = 'input:0'               

	def __init__(self,data_dir):

		# New Tensorflow Computational Graph
		self.graph = tf.Graph()

		# Making graph as default graph
		with self.graph.as_default():
			path = os.path.join(data_dir , path_graph_def)
			with tf.gfile.FastGFile(path , 'rb') as file:

				# Saved Copy of Tensorflow graph
				graph_def = tf.GraphDef()

				# Load the protobuff file in graph_def
				graph_def.ParseFromString(file.read())

				# Finally Import the graph-def to the default Tensorflow graph
				tf.import_graph_def(graph_def , name = '')

				# Now self.graph holds the Inception Model 	from Proto-Buf file

			
			# Getting refrence for input
			self.input = self.graph.get_tensor_by_name('input:0')

			# Getting refrence for Layer Tensors
			self.layer_tensors = [self.graph.get_tensor_by_name(name + ':0') for names in layer_names]


	# Method providing for feed dict used while training		
	def create_feed_dict(self,image = None):

		# Because there is only a single image so we have to expand the image as inception will take the input with batch size
		image = np.expand_dims(image , axis = 0)
		
		# Making of feed dict
		feed_dict = {self.tensor_name_input_image : image}
		return feed_dict


	# Gradients
	def get_Gradients(self , tensor):

		# making the current graph as default graph
		with self.graph.as_default():

			# Square the tensor values
			tensor = tf.square(tensor)

			# Calculate the mean
			tensor_mean = tf.reduce_mean(tensor)

			gradient = tf.gradients(tensor_mean , self.input)[0]


		return gradient	


########################################################################################################

# Load Images in type float
def Load_Image(filename):
	image = Image.open(filename)
	return np.float32(image)


########################################################################################################

# Saving image function
def save_Image(image , file_name):
	image = np.clip(image , 0.0 , 255.0)

	# Convert the image into bytes
	image = image.astype(np.uint8)

	# Write the image file in jpeg format
	with open(file_name , 'wb') as file:
		Image.fromarray(image).save(file , 'jpeg')


########################################################################################################

# Function for plotting the images
def plot_image(image):
	
	'''
	 np.clip -> Clip (limit) the values in an array.
				Given an interval, values outside the interval are clipped to the interval edges. 
				For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
	'''
	image = np.clip(image , 0.0 , 255.0)

	# Convert the image into bytes
	image = image.astype(np.uint8)

	# Show Image
	display(Image.fromarray(image))



########################################################################################################

# Normalization ( :) Just remember the formula )
def Normalization(x):

	# Applying Norm. Formula
	xnorm = (x - x.min()) / (x.max() - x.min())
	return xnorm



########################################################################################################

def Plot_Gradient(gradient):

	# Normalize the gradient between 0.0 and 1.0
	normalized_gradient = Normalization(gradient)

	# Normalized Gradient show
	plt.imshow(normalized_gradient , interpolation = 'bilinear')
	plt.show()


########################################################################################################

# For Resizing One image 
# Arguments -> Size -> You can specify size of final Image
# 			-> Factor -> You can specify the factor of how much to scale the Image in height and width
def resize_image(image , size = None , factor = None):

	if factor is not None:

		# Scale numpy array (height and width)
		size = np.array(image.shape[0:2]) * factor

		# as we have scaled down so it must be having floating point values
		# For PIL convert it into integers

		size = size.astype(int)

	else:
		size = size[0:2]

	
	# Numpy vs PIL -> Height and Width are reversed
	size = tuple(reversed(size))

	# Make Clipping Bro
	image = np.clip(image , 0.0 , 255.0)

	# Convert into bytes
	image = image.astype(np.unint8)

	image = Image.fromarray(image)

	# Resized the image
	img_resized = image.resize(size , Image.LANCZOS)

	# Converting into float
	image_final = np.float32(img_resized)

	return image_final

#############################################################################################################

'''
Gradient

The following helper-functions calculate the gradient of an input image for use in the DeepDream algorithm. The Inception 5h model can accept images of any size, but very large images may use many giga-bytes of RAM. In order to keep the RAM-usage low we will split the input image into smaller tiles and calculate the gradient for each of the tiles.
However, this may result in visible lines in the final images produced by the DeepDream algorithm. We therefore choose the tiles randomly so the locations of the tiles are always different. This makes the seams between the tiles invisible in the final DeepDream image.
This is a helper-function for determining an appropriate tile-size. The desired tile-size is e.g. 400x400 pixels, but the actual tile-size will depend on the image-dimensions.

'''

## Get Appropriate Tile size according to the input image dimension
def get_tiles_size(num_pixels , tile_size = 400):

	# Make a assumption tile size
	# getting number of tiles approx
	num_tiles = int(round(num_pixels/tile_size))

	# Ensure that there is at least 1 tile.
	num_tiles = max(1 , num_tiles)

	# Actual Tile size
	# Make sure you take ceil not floor
	actual_tile_size = math.ceil(num_pixels / num_tiles)

	return actual_tile_size

# Calculate the gradient on titles 
def tiled_gradient(gradient , image , tile_size = 400):

	# Allocate an array for gradients for an image
	grad = np.zeros_like(image)

	x_max , y_max , _ = image.shape

	x_tile_size = get_tiles_size(num_pixels = x_max , tile_size = tile_size)

	y_tile_size = get_tiles_size(num_pixels = y_max , tile_size = tile_size)


	# Divide both sizes by 4

	x_tile_size4 = x_tile_size // 4
	y_tile_size4 = y_tile_size // 4

	# Taking random values for x_start between -3/4 and -1/4 of tile size because eliminate borders
	x_start = random.randint(-3 * x_tile_size4 , -1 * x_tile_size4)

	while x_start < x_max:

		# End Position for tiles
		x_end = x_start + x_tile_size

		# Tile start are valid
		x_start_lim = max(x_start , 0)

		# Tile end are valid
		x_end_lim = min(x_end , x_max)

		# Taking random values for y_start between -3/4 and -1/4 of tile_size
		y_start = random.randint(-3 * y_tile_size4 , -1 * y_tile_size4)

		while y_start < y_max:
		
			# End position for tiles
			y_end = y_start + y_tile_size

			# Make sure start and end lim are valid
			y_start_lim = max(0 , y_start)
			y_end_lim = min(y_end , y_max)

			############## LETS START ############################

			# Finally get the image tile
			img_tile = image[x_start_lim : x_end_lim , y_start_lim : y_end_lim , :]

			# Feed Dict Creation (You Know that i thought so)
			feed_dict = model.create_feed_dict(image = img_tile)

			# Tensorflow will calculate gradient for us
			g = session.run(gradient , feed_dict = feed_dict)

			# Normalize the gradient ( Very Important Step )
			g /= (np.std(g) + 1e-8)

			# Putting the gradient into tiles of grad defined above # Check Above 
			grad[x_start_lim : x_end_lim , y_start_lim : y_end_lim , :] = g

			y_start = y_end

		x_start = x_end

	# Return the new image with gradient 	
	return grad	


##########################################################################################################


'''
Optimize Image
This function is the main optimization-loop for the DeepDream algorithm.
It calculates the gradient of the given layer of the Inception model with regard to the input image. 
The gradient is then added to the input image so the mean value of the layer-tensor is increased. 
This process is repeated a number of times and amplifies whatever patterns the Inception model sees in the input image.
'''	

def OptimizeImageFunc(layer_tensor , image , num_iterations = 10 , step_size = 0.3 , tile_size = 400 , show_gradient = False):

	'''
		Layer_Tensor -> Refrence to a tensor that will be maximized
		image -> input image
	'''

	# make a copy
	img = image.copy()


	# Lets See some images
	print ('Image Before : ')
	plot_image(img)

	# Now Lets start some preprocessing
	print ('Processing Image : ')

	# get Gradient
	gradient = model.get_gradient(layer_tensor)

	for i in range(num_iterations):

		# Take the grads from tiles_gradient function
		grad = tiled_gradient(gradient = gradient , image = img)

		# Now we will be making use of gaussian filters
        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.		

        ###### Note Blur-Amount -> Sigma
        sigma = (i * 4.0)/num_iterations + 0.5

        # 1. With same sigma  2. with double sigma  3. with half sigma 
        grad_smooth1 = gaussian_filter(grad , sigma = sigma) 
        grad_smooth2 = gaussian_filter(grad , sigma = sigma * 0.5)
        grad_smooth3 = gaussian_filter(grad , sigma = sigma * 2)

        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scaling the step size as we have scaled the gradient
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image
        img += grad * step_size_scaled

        # Plot show if show = True
        if show_gradient:
        	Plot_Gradient(grad)


    # Plotting the image    	
    print ()
    print ('Image After ::::  ')
    plot_image(img)


    # Return image
    return img










