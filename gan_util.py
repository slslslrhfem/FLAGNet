import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import subprocess
from hyperparameter import Hyperparams as hp

"""
Original dcGAN Codes from https://github.com/gaborvecsei/CDCGAN-Keras. 
"""
def combine_images(generated_images):
    num_images = generated_images.shape[0]
    new_width = int(math.sqrt(num_images))
    new_height = int(math.ceil(float(num_images) / new_width))
    grid_shape = generated_images.shape[1:3]
    grid_image = np.zeros((new_height * grid_shape[0], new_width * grid_shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / new_width)
        j = index % new_width
        grid_image[i * grid_shape[0]:(i + 1) * grid_shape[0], j * grid_shape[1]:(j + 1) * grid_shape[1]] = \
            img[:, :, 0]
    return grid_image


def generate_noise(shape: tuple):
    noise = np.random.uniform(0, 1, size=shape)
    return noise


def generate_condition_embedding(label: int, nb_of_label_embeddings: int):
    label_embeddings = np.zeros((nb_of_label_embeddings, hp.Label_num))
    label_embeddings[:, label] = 1
    return label_embeddings

def generate_condition(condition, nb_of_image_embeddings: int):
    image_embeddings = []
    for nb in range(nb_of_image_embeddings):
      image_embeddings.append(condition)
    image_embeddings = np.array(image_embeddings)
    return image_embeddings

def generate_condition_with_batch(condition, nb_of_image_embeddings: int):
    image_embeddings = []
    for nb in range(nb_of_image_embeddings):
      image_embeddings.append(condition[nb])
    image_embeddings = np.array(image_embeddings)
    return image_embeddings

def generate_images(generator, nb_images: int, label: int, time_condition,pitch_condition):
    noise = generate_noise((nb_images, hp.Label_num))
    label_batch = generate_condition_embedding(label, nb_images)
    time_condition_batch = generate_condition(time_condition, nb_images)
    pitch_condition_batch = generate_condition(pitch_condition, nb_images)
    generated_images = generator.predict([noise, label_batch,time_condition_batch, pitch_condition_batch], verbose=0)
    return generated_images


def generate_mnist_image_grid(generator,time_condition, pitch_condition, title: str = "Generated images"):
    generated_images = []
    for i in range(hp.Label_num):
        noise = generate_noise((hp.Label_num, hp.Label_num))
        label_input = generate_condition_embedding(i, hp.Label_num)
        time_condition_batch = generate_condition_with_batch(time_condition, hp.Label_num)
        pitch_condition_batch = generate_condition_with_batch(pitch_condition, hp.Label_num)
        gen_images = generator.predict([noise, label_input ,time_condition_batch, pitch_condition_batch], verbose=0)
        generated_images.extend(gen_images)

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images)
    image_grid = inverse_transform_images(image_grid)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(image_grid, cmap="gray")
    ax.set_title(title)
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image


def save_generated_image(image, epoch, iteration, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    cv2.imwrite(file_path, image.astype(np.uint8))


def transform_images(images: np.ndarray):
    """
    [0,1]Transform images to [-1, 1]
    """
    max_value=images.max()

    images = (images.astype(np.float32) - (max_value/2)) / (max_value/2)
    return images


def inverse_transform_images(images: np.ndarray):
    """
    From the [-1, 1] range transform the images back to [0, 255]
    """

    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images


def convert_video_to_gif(input_video_path, output_gif_path, fps=24):
    palette_image_path = "palette.png"
    command_palette = 'ffmpeg -y -t 0 -i {0} -vf fps={1},scale=320:-1:flags=lanczos,palettegen {2}'.format(input_video_path,
                                                                                                           fps,
                                                                                                           palette_image_path)
    command_convert = 'ffmpeg -y -t 0 -i {0} -i {1} -filter_complex "fps={2},scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" {3}'.format(input_video_path,palette_image_path, fps, output_gif_path)
    
    try:
        subprocess.check_call(command_palette)
        subprocess.check_call(command_convert)
    except subprocess.CalledProcessError as exc:
        print(exc.output)
        raise
    finally:
        os.remove(palette_image_path)