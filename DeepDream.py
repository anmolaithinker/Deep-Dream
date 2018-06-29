import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
import PIL.Image
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

def downloadInception5H(data_dir):
  data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
  data_dir = data_dir
  file_name = data_url.split('/')[-1]
  file_path = os.path.join(data_dir , file_name)
  print ("File is downloading in  : " + str(file_path))
  if not os.path.exists(file_path):
    if not os.path.exists(data_dir):
      os.create(data_dir)
  
   # Retrieving data from file
    file_path , _  = urllib.request.urlretrieve(url = data_url , filename = file_path , reporthook = PrintDownloadProgress)
  
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

path_graph_def = "tensorflow_inception_graph.pb"

  	# Commonly used layers in inception

layer_names = ['conv2d0', 'conv2d1', 'conv2d2','mixed3a', 'mixed3b','mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e','mixed5a', 'mixed5b']


tensor_name_input_image = 'input:0'               

class inception5h:
  	
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
		  self.layer_tensors = [self.graph.get_tensor_by_name(names + ':0') for names in layer_names]


	# Method providing for feed dict used while training		
  def create_feed_dict(self,image = None):

		# Because there is only a single image so we have to expand the image as inception will take the input with batch size
	  image = np.expand_dims(image , axis = 0)
		
		# Making of feed dict
	  feed_dict = {tensor_name_input_image : image}
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
	image = PIL.Image.open(filename)
	return np.float32(image)


########################################################################################################

# Saving image function
def save_Image(image , file_name):
	image = np.clip(image , 0.0 , 255.0)

	# Convert the image into bytes
	image = image.astype(np.uint8)

	# Write the image file in jpeg format
	with open(file_name , 'wb') as file:
		PIL.Image.fromarray(image).save(file , 'jpeg')


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
	display(PIL.Image.fromarray(image))


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

'''
Steps to perform resize the image according to size or factor

-> Lets take according to the factor first
-----> Scale the size of image by factor given 
-----> Convert the result of above into int (PIL)
-----> Perform numpy clipping
-----> convert into bytes
-----> getting numpy array from PIL.Image
-----> Perform Resize operation ( Note LANCZOS === ANTIALIAS)
-----> Convert into float
-----> return 

-> Lets take according to the size first
-----> Perform numpy clipping
-----> convert into bytes
-----> getting numpy array from PIL.Image
-----> Perform Resize operation ( Note LANCZOS === ANTIALIAS)
-----> Convert into float
-----> return 
'''
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

	image = PIL.Image.fromarray(image)

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


############################################################################################################

'''

Steps to follow in this function:

-----> Initialize the grad variable with all zeroes value having same shape of image
-----> Getting the tile size (x and y axis) with get_tile_size function 
-----> get starting of x tile 
-----> get starting of y tile
-----> x end = x start + x tile size
-----> y end = y start + y tile size
-----> Get image tile given ( xstart , xend ) and ( ystart , yend )
-----> Calculate the gradient with tensorflow
-----> Normalize the gradient
-----> set grad (xstart , xend) and (ystart , yend) = g
-----> return grad
'''

# Calculate the gradient on titles 
# Calculate the gradient on titles 
def tiled_gradient(gradient , image , tile_size = 400 , model = None , session = None):

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

'''
Steps that will be performed in this function

-----> Call get gradient function to get the gradients to the corresponding layer tensor
-----> Iterate till the number of iterations given
-----> We are inside the loop
-----> Call tiled_gradient function and get grad (image of gradient of input image)
-----> smooth and normalize the gradients
-----> add the gradient into image
-----> return image 
'''

def OptimizeImageFunc(layer_tensor , image , num_iterations = 10 , step_size = 0.3 , tile_size = 400 , show_gradient = False , model = None,session = None):
  
  img = image.copy()
  print ('Image Before : ')
  plot_image(img)
  print ('Processing Image : ')
  gradient = model.get_Gradients(layer_tensor)

  for i in range(num_iterations):
    grad = tiled_gradient(gradient = gradient , image = img , model = model , session = session)
    sigma = (i*4.0)/num_iterations + 0.5
    grad_smooth1 = gaussian_filter(grad , sigma = sigma) 
    grad_smooth2 = gaussian_filter(grad , sigma = sigma * 0.5)
    grad_smooth3 = gaussian_filter(grad , sigma = sigma * 2)
    grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
    step_size_scaled = step_size / (np.std(grad) + 1e-8)
    img += grad * step_size_scaled
    if show_gradient:
      Plot_Gradient(grad)

  print ()
  print ('Image After ::::  ')
  plot_image(img)


  # Return image
  return img


########################################################################################################

'''
Recursive Image Optimization
The Inception model was trained on fairly small images.
The exact size is unclear but maybe 200-300 pixels in each dimension. 
If we use larger images such as 1920x1080 pixels then the optimize_image() function above will add many small patterns to the image.
This helper-function downscales the input image several times and runs each downscaled version through the optimize_image() function above. This results in larger patterns in the final image. It also speeds up the computation.
'''    

'''
Arguments

1. Layer_Tensor
2. image
3. num_repeats = 4
4. rescale factor = 0.7
5. ?? blend = 0.2	
6.	num_iterations = 10
7. step size = 3.0
8. tile_size = 400
'''

'''
Steps to follow recursive optimization
-> Base condition
-> Blur the image
-> Downscale the image
-> Apply Recursion ( num_repeats = num_repeats - 1 , image = img_down )
-> (BackTracking) -> Upsampling and blending
-> End Loop
-> Call Optimized loop and save into image result
-> return image result

'''
# def Recusrive_Optimization(layer_tensor , image , num_repeats = 4 , rescale_factor = 0.7 , blend = 0.2 , num_iterations = 10 , step_size = 3.0 , tile_size = 400):

# 	# Base condition to stop recursive is num_repeats > 0
# 	if num_repeats > 0:
# 		# Blur the input image to prevent artifacts when downscaling... <--
# 		# For blur the input image we will set the value of sigma
# 		# We are using gaussian filter
# 		sigma  = 0.5
# 		blur_image = gaussian_filter(image , sigma = (sigma,sigma,0.0))

# 		# Downscale the image
# 		img_down = resize_image(image = blur_image , factor = rescale_factor)

# 		# Make Recursive call
# 		# num_repeats = num_repeats - 1
# 		# Use img_down

# 		img_result = Recusrive_Optimization(layer_tensor = layer_tensor ,
# 											image = img_down ,
# 											num_repeats = num_repeats-1 , 
# 											rescale_factor = rescale_factor, 
# 											blend = blend, 
# 											num_iterations = num_iterations, 
# 											step_size = step_size, 
# 											tile_size = tile_size)


# 		# Now do upsampling
# 		img_up = resize_image(image = img_result , size = image.shape)

# 		# Blend(Mix two images)
# 		# image_Upsampled and true image

# 		image = blend * image + (1.0 - blend) * img_up 

# 	print ("Recursive Level : " , num_repeats)
	
# 	# Apply the function to merge image with gradients 
# 	img_result = optimize_image(layer_tensor = layer_tensor ,
# 								 image = image , 
# 								 num_iterations = num_iterations , 
# 								 step_size = step_size , 
# 								 tile_size = tile_size)

# 	return img_result


############################################################################################################

def Main(image_path , layer_tensor_num , recursive = False):

	if image_path == '' or layer_tensor_num == '' or layer_tensor_num < 0:
		print ('Error!!!!!!')
	
	else:

		downloadInception5H('./')
		model = inception5h('./')
		
		# 12 should be the output
		print ("Number of layers in Inception : " + str(len(model.layer_tensors)))


		# To Execute the graph we need tensorflow session . 
		# We are using interactive session
		session = tf.InteractiveSession(graph = model.graph)
		
		#Load image
		image = Load_Image(image_path)

		# Plot Images
		plot_image(image)

		# Taking layer tensor whose gradient has to be maximized
		layer_tensor = model.layer_tensors[layer_tensor_num]
		print ("Taking Layer Tensor " + str(layer_tensor))

		if recursive:
			print ('Sorry but recursive not yet supported !!!!!!!')
			# img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
   #               num_iterations=10, step_size=3.0, rescale_factor=0.7,
   #               num_repeats=4, blend=0.2 , model = model)
		else:
			img_result = OptimizeImageFunc(layer_tensor, image,
                   num_iterations=10, step_size=6.0, tile_size=400,
                   show_gradient=True , model = model , session = session)	
















