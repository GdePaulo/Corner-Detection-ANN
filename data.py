import os
from os import path
import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from numpy import asarray

# 
def get_images(path):
    images = []
    # Loop through all images in the given directory.
    for filename in os.listdir(path):
        image = asarray(Image.open(path + filename))[:, :, 0]
        # Flipping zeros and ones because the lines from the input images are black instead of white.
        # Clipping the image to values of 0 or 1.
        image = 1 - np.clip(image, 0, 1)
        # Generating the 4 possible rotations of the input image.
        for i in range(4):
            images.append(np.rot90(image, i).flatten())
    return images

def draw_shapes2(draw, width):
    triangleSize = ( 20, 20 )
    pointySize = ( 15, 30 )
    tiltedSize = ( 30, 15 )


    t = [get_triangle_points(triangleSize, x) for x in range(4)]
    shiftPoints(t[0], (98, 10))
    shiftPoints(t[1], (98, 98))
    shiftPoints(t[2], (10, 98))
    shiftPoints(t[3], (10, 10))

    s = [get_pointy_square_points(pointySize, x) for x in range(4)]
    shiftPoints(s[0], (40, 24))
    shiftPoints(s[1], (73, 24))
    shiftPoints(s[2], (73, 74))
    shiftPoints(s[3], (40, 74))
    
    r = [get_tilted_square_points(tiltedSize, x) for x in range(2)]
    shiftPoints(r[0], (5, 49))
    shiftPoints(r[1], (93, 49))
    
    # We should eventually work to an automated shape generator
    corner_coordinates = []
    for triangle in t:
        corner_coordinates.append(draw_triangle(draw, width, *triangle))    
    for pointySquare in s:
        corner_coordinates.append(draw_trapezium(draw, width, *pointySquare))    
    for tiltedSquare in r:
        corner_coordinates.append(draw_trapezium(draw, width, *tiltedSquare))    
    
    return [item for sublist in corner_coordinates for item in sublist]

def draw_shapes(draw, width):
    # We should eventually work to an automated shape generator
    corner_coordinates = [draw_rectangle(draw, width, (4, 4), (20, 20)),
                          draw_triangle(draw, width, (30, 4), (30, 30), (56, 30)),
                          draw_trapezium(draw, width, (80, 4), (100, 4), (116, 20), (64, 20)),
                          draw_octagon(draw, width, (50, 60), (67, 43), (91, 43), (108, 60))]
    draw_line(draw, width, (4, 30), (20, 30))
    draw_circle(draw, width, (4, 80), (40, 116))

    return [item for sublist in corner_coordinates for item in sublist]

def shiftPoints(points, shift):
    for i, point in enumerate(points):
        currentPoint = (point[0]+shift[0], point[1]+shift[1])
        points[i]=currentPoint

def get_triangle_points(size, rotation):
    bounding_points = np.array([(0, 0), (size[0], 0), size, ( 0, size[0])])
    indices = range(rotation, rotation + 3)
    
    points = bounding_points.take(indices, axis=0, mode= "wrap")
    return list(zip(points[:,0], points[:,1]))
def get_pointy_square_points(size, rotation):
    bounding_points = [(0, 0), (size[0], 0), (size[0], size[1] / 2), size, (0, size[1]), (0, size[1] / 2)]
    points = []
    if rotation==0:
        points = [x for i,x in enumerate(bounding_points) if i!=1 if i!=5]
    elif rotation==1:
        points = [x for i,x in enumerate(bounding_points) if i!=2 if i!=0]
    else:
        points = [x for i,x in enumerate(bounding_points) if i!=rotation if i!=rotation+2]
    return points
def get_tilted_square_points(size, rotation):
    bounding_points = [(0, 0), (size[0] / 2, 0), (size[0], 0), (size[0], size[1]), (size[0] / 2, size[1]), (0, size[1])]
    points = []
    if rotation==0:
        points = [x for i,x in enumerate(bounding_points) if i!=2 if i!=5]
    elif rotation==1:
        points = [x for i,x in enumerate(bounding_points) if i!=0 if i!=3]
    return points


# No corner shapes

def draw_line(draw, width, left, right):
    draw.line((left, right), fill="white", width=width)
    return


def draw_circle(draw, width, left, right):
    draw.ellipse((left, right), outline="white", width=width)
    return


# Corner shapes

def draw_rectangle(draw, width, top_left, bottom_right):
    draw.rectangle((top_left, bottom_right), outline="white", width=width)
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return top_left, top_right, bottom_left, bottom_right


def draw_triangle(draw, width, c1, c2, c3):
    draw.line((c1, c2, c3, c1), fill="white", width=width)
    return c1, c2, c3


def draw_trapezium(draw, width, c1, c2, c3, c4):
    draw.line((c1, c2, c3, c4, c1), fill="white", width=width)
    return c1, c2, c3, c4


def draw_octagon(draw, width, c1, c2, c3, c4):
    segment_length = c3[0] - c2[0]
    across_length = c4[0] - c1[0]
    c5, c6, c7, c8 = (c4[0], c4[1] + segment_length), (c3[0], c3[1] + across_length), (c2[0], c2[1] + across_length), \
                     (c1[0], c1[1] + segment_length)
    draw.line((c1, c2, c3, c4, c5, c6, c7, c8, c1), fill="white", width=width)
    return c1, c2, c3, c4, c5, c6, c7, c8


def wizard(img, img_size, corners, corner_images, no_corner_images):
    sliding_window = 8

    for i in range(img_size - sliding_window):
        for j in range(img_size - sliding_window):
            sub_img = img.crop((i, j, i + sliding_window, j + sliding_window))
            is_corner = False
            for corner in corners:
                if i + (sliding_window / 4) <= corner[0] < i + sliding_window - (sliding_window / 4) and + \
                        j + (sliding_window / 4) <= corner[1] < j + sliding_window - (sliding_window / 4):
                    is_corner = True
                    break

            img_array = np.clip(np.array(sub_img), 0, 1).flatten()

            if is_corner:
                # corner_images_count += 1
                if not any(np.array_equal(img_array, x) for x in corner_images):
                    corner_images.append(img_array)
            else:
                # no_corner_images_count += 1
                if not any(np.array_equal(img_array, x) for x in no_corner_images):
                    no_corner_images.append(img_array)


def generate_data():
    corner_images, no_corner_images = [], []

    if  path.exists("corner_images.pickle") and path.exists("no_corner_images.pickle"):
        corner_images = pickle.load(open("corner_images.pickle", "rb"))
        no_corner_images = pickle.load(open("no_corner_images.pickle", "rb"))
    else:
        print("Draw shapes data generation")
        img_size = 128
        width = 1
        img = Image.new("L", (img_size, img_size))
        draw = ImageDraw.Draw(img)
        corners = draw_shapes(draw, width)
        img.save(f"test_{width}.png")
        wizard(img, img_size, corners, corner_images, no_corner_images)

        print(f'Amount of corner images: {len(corner_images)}')
        print(f'Amount of no corner images: {len(no_corner_images)}')

        img = Image.new("L", (img_size, img_size))
        draw = ImageDraw.Draw(img)
        corners = draw_shapes2(draw, width)
        img.save(f"test_{width}_2.png")
        wizard(img, img_size, corners, corner_images, no_corner_images)

        print(f'Amount of corner images: {len(corner_images)}')
        print(f'Amount of no corner images: {len(no_corner_images)}')

        corner_images += get_images(path='images/corners/')
        no_corner_images += get_images(path='images/no_corners/')
        
        print(f'Amount of corner images: {len(corner_images)}')
        print(f'Amount of no corner images: {len(no_corner_images)}')
    
        with open('corner_images.pickle', 'wb') as handle:
            pickle.dump(corner_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('no_corner_images.pickle', 'wb') as handle:
            pickle.dump(no_corner_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    x = [image for image in no_corner_images] + [image for image in corner_images]
    x = torch.FloatTensor(x)
    y = [0 for _ in range(len(no_corner_images))] + [1 for _ in range(len(corner_images))]
    y = torch.FloatTensor(y)
    return x, y

if __name__ == "__main__":
    generate_data()
