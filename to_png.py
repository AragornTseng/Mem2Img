import numpy
from PIL import Image
import binascii
import errno    
import os
import math
import sys
import argparse
from pyentrp import entropy as ent
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io


def getBinaryData(filename):
	"""
	Extract byte values from binary executable file and store them into list
	:param filename: executable file name
	:return: byte value list
	"""

	binary_values = []

	with open(filename, 'rb') as fileobject:

		# read file byte by byte
		data = fileobject.read(1)

		while data != b'':
			binary_values.append(ord(data))
			data = fileobject.read(1)

	return binary_values

def get_entropy(data):
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    binary = get_bin(data,8)
    ts = []
    for i in binary:
        ts.append(i)
    sample_entropy = ent.shannon_entropy(ts)
    return sample_entropy


def createRGBImage(filename):
	"""
	Create RGB image from 24 bit binary data 8bit Red, 8 bit Green, 8bit Blue
	:param filename: image filename
	"""
	index = 0
	rgb_data = []
	# Read binary file
	binary_data = getBinaryData(filename)
	binary_data = numpy.array(binary_data)
	rn = int(math.sqrt(len(binary_data)))
	fh = numpy.reshape(binary_data[:rn*rn],(-1,rn)) 
	fh = numpy.uint8(fh)
	entropy_img = entropy(fh,disk(10))
	entropy_img = entropy_img.flatten()
    # Create R,G,B pixels
	while index  < len(entropy_img):
		R = binary_data[index]
		G = int(entropy_img[index]*15)
		B = int(get_entropy(binary_data[index])*60)
		index += 1
		rgb_data.append((R, G, B))
    
	size = int(math.sqrt(len(rgb_data)))
	rn1 = int(math.sqrt(len(rgb_data)))+1
	image = Image.new('RGB',(rn1,rn1))
	image.putdata(rgb_data)
	return image

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main(argv=[]):

    parser = argparse.ArgumentParser(description="convert binary to png")
    parser.add_argument('input directory', metavar='input_dir', help='path of input directory')
    parser.add_argument('output directory', metavar='output_dir', help='path of output directory')
    args = parser.parse_args()
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    paths = [[input_dir+'\Train', output_dir+'\Train'],[input_dir+'\Test', output_dir+'\Test']]
    for p in paths:
        for i, d in enumerate(os.listdir(p[0])):
            dir_full = os.path.join(p[1], str(d))
            mkdir_p(dir_full)
            for f in os.listdir(os.path.join(p[0], d)):
                bin_full = os.path.join(p[0], d, f)
                im = createRGBImage(bin_full)
                png_full = os.path.join(dir_full, os.path.splitext(f)[0]+'.png')
                im.save(png_full)


if __name__ == '__main__':
    main(sys.argv)