# import dependencies
import argparse
import os

import numpy as np
import PIL
import PIL.Image as Image

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--orig_data',
                    default=None,
                    type=str,
                    required=True,
                    help='absolute path to original data',
                    metavar='PATH')
parser.add_argument('--new_data',
                    default=None,
                    type=str,
                    required=True,
                    help='absolute path to rotated data',
                    metavar='PATH')

args = parser.parse_args()


# define function
def rotate(img, deg):
    '''rotate and save image'''
    # add RGB channels to monochrome image
    if len(np.array(img).shape) < 3:
        img = np.stack((img, ) * 3, axis=-1)

    # rotate image
    if deg == 0:
        img = Image.fromarray(np.array(img))
    elif deg == 90:
        img = Image.fromarray(np.flipud(np.transpose(img, (1, 0, 2))))
    elif deg == 180:
        img = Image.fromarray(np.fliplr(np.flipud(img)))
    elif deg == 270:
        img = Image.fromarray(np.transpose(np.flipud(img), (1, 0, 2)))
    else:
        raise ValueError('only 0, 90, 180, or 270 degrees rotations allowed')

    return img


# process data
for dir_ in os.listdir(args.orig_data):
    idx = 0
    for img_name in os.listdir(os.path.join(args.orig_data, dir_)):
        if img_name.endswith('JPEG'):
            img = Image.open(os.path.join(args.orig_data, dir_, img_name))

            if idx % 4 == 0:
                deg = 0
            elif idx % 4 == 1:
                deg = 90
            elif idx % 4 == 2:
                deg = 180
            elif idx % 4 == 3:
                deg = 270

            img = rotate(img, deg)
            img.save(os.path.join(args.new_data, str(idx % 4), img_name))
            idx += 1

print('rotation dataset created')
