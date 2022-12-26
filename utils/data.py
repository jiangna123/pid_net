from __future__ import print_function

from cv2 import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    assert img.size != 0
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def pidadjustData(img, mask, edge, flag_multi_class, num_class):
    assert img.size != 0
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
            new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        edge = edge / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        edge[edge > 0.5] = 1
        edge[edge <= 0.5] = 0
    return img, mask, edge


def pidtrainGenerator(batch_size, train_path, image_folder, mask_folder, edge_folder, aug_dict,
                      image_color_mode="grayscale",
                      mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                      flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    edge_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    edge_generator = edge_datagen.flow_from_directory(
        train_path,
        classes=[edge_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, mask_generator, edge_generator)
    for (img, mask, edge) in train_generator:
        img, mask, edge = pidadjustData(img, mask, edge, flag_multi_class, num_class)
        yield (img, [mask, edge])


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for image in test_path:
        img = io.imread(image, as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        if as_gray:
            img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


def PidsaveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for index, ary in enumerate(npyfile):
        for i, item in enumerate(ary):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            io.imsave(os.path.join(save_path, f"{index}/%d_predict.png" % i), img)


index_dict = {0: 0, 1: 1,
              2: 12, 3: 16, 4: 17, 5: 18,
              6: 19, 7: 20, 8: 21, 9: 22,
              10: 2, 11: 3, 12: 4, 13: 5,
              14: 6, 15: 7, 16: 8, 17: 9,
              18: 10, 19: 11, 20: 13, 21: 14,
              22: 15}


def save_hstack_img(inputs, preputs, outputs):
    label_path = os.path.join(inputs, 'label/')
    test_path = os.path.join(preputs, '0/')
    edge_path = os.path.join(preputs, '1/')
    for index in os.listdir(inputs + "/image"):
        ex = os.path.splitext(index)[0]
        sou_img = cv2.imread(f"{inputs}/image/{index}")
        label_img = cv2.imread(f"{label_path}/{index}")
        test_img = cv2.imread(f"{test_path}/{index_dict.get(int(ex))}_predict.png")
        edge_img = cv2.imread(f"{edge_path}/{index_dict.get(int(ex))}_predict.png")

        sou_img = cv2.resize(sou_img, (256, 256))
        label_img = cv2.resize(label_img, (256, 256))
        test_img = cv2.resize(test_img, (256, 256))
        edge_img = cv2.resize(edge_img, (256, 256))

        # 把数据压到栈中
        inputss = np.hstack((sou_img, label_img, test_img, edge_img))
        cv2.imwrite(outputs + f"/{index}", inputss)


def save_hstacks_img(inputs, outputs):
    label_path = os.path.join(inputs, 'label/')
    test_path = os.path.join(inputs, 'plabel/')
    edge_path = os.path.join(inputs, 'pedge/')
    for index in os.listdir(inputs + "/image"):
        sou_img = cv2.imread(f"{inputs}/image/{index}")
        label_img = cv2.imread(f"{label_path}/{index}")
        test_img = cv2.imread(f"{test_path}/{index}")
        edge_img = cv2.imread(f"{edge_path}/{index}")

        sou_img = cv2.resize(sou_img, (256, 256))
        label_img = cv2.resize(label_img, (256, 256))
        test_img = cv2.resize(test_img, (256, 256))
        edge_img = cv2.resize(edge_img, (256, 256))

        # 把数据压到栈中
        inputss = np.hstack((sou_img, label_img, test_img, edge_img))
        cv2.imwrite(outputs + f"/{index}", inputss)


def rename_dir(inputs):
    test_path = os.path.join(inputs, 'plabel/')
    edge_path = os.path.join(inputs, 'pedge/')
    for index in os.listdir(inputs + "/image"):
        ex = os.path.splitext(index)[0]
        os.rename(f"{test_path}/{index_dict.get(int(ex))}_predict.png", f"{test_path}/{index}")
        os.rename(f"{edge_path}/{index_dict.get(int(ex))}_predict.png", f"{edge_path}/{index}")


def rgb2gray(inputs, output):
    for index in os.listdir(inputs + "/image"):
        sou_img = cv2.imread(f"{inputs}/image/{index}", cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(f"{inputs}/label/{index}", cv2.IMREAD_GRAYSCALE)
        # edge_img = cv2.imread(f"{inputs}/edge/{index}", cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"{output}/image/{index}", sou_img)
        cv2.imwrite(f"{output}/label/{index}", label_img)
        # cv2.imwrite(f"{output}/edge/{index}", edge_img)

    # for index in os.listdir(inputs + "/val_image"):
    #     val_label_img = cv2.imread(f"{inputs}/val_label/{index}", cv2.IMREAD_GRAYSCALE)
    #     val_image_img = cv2.imread(f"{inputs}/val_image/{index}", cv2.IMREAD_GRAYSCALE)
    #     val_edge_img = cv2.imread(f"{inputs}/val_edge/{index}", cv2.IMREAD_GRAYSCALE)
    #
    #     cv2.imwrite(f"{output}/val_label/{index}", val_label_img)
    #     cv2.imwrite(f"{output}/val_image/{index}", val_image_img)
    #     cv2.imwrite(f"{output}/val_edge/{index}", val_edge_img)


if __name__ == '__main__':
    # save_hstack_img('data/mszs_test256_11/', 'data/output/aspp-att-pid/mszs_test256_1', 'data/output/hstack')
    rename_dir('data/output/deeplabv3+/mszs_test256_1')
