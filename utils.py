import torch
import numpy as np
import h5py
import foolbox
import copy
import os
import h5py
from scipy import misc

IMG_EXTENSIONS = ['.JPEG', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def majority_vote(pred_y, num_class=200, branch_size=1):
    pred_y_list = []
    batch_size_y = pred_y.shape[1]
    for idx_pred in range(branch_size):
        pred_y_i = torch.zeros(batch_size_y, num_class).cuda().scatter_(1, pred_y[idx_pred].view(batch_size_y, 1), 1)
        pred_y_list.append(pred_y_i)
    pred_y_sum = torch.zeros(batch_size_y, num_class).cuda()
    for item in pred_y_list:
        pred_y_sum += item
    _, pred_y_vote = pred_y_sum.data.max(dim=1)
    return pred_y_vote


def reload_adv_data(dir_h5=None, batch_size=50000):
    assert dir_h5 is not None
    with h5py.File(dir_h5, "r") as f:
        X_train_h5 = f["X_train"]
        num_train = len(X_train_h5)
        X_val_h5 = f["X_val"]
        num_val = len(X_val_h5)
        print('num_train: ', num_train)
        print('num_val: ', num_val)
    with h5py.File(dir_h5) as f:
        # reload training adv image
        for idx in range(0, num_train, batch_size):
            f["X_train_adv"][idx:idx + batch_size] = f["X_train"][idx:idx + batch_size]
            print("Process Training Data: {0:.2f}%".format(idx / num_train * 100.0))
        # reload val adv image
        for idx in range(0, num_val, batch_size):
            f["X_val_adv"][idx:idx + batch_size] = f["X_val"][idx:idx + batch_size]
            print("Process Val Data: {0:.2f}%".format(idx / num_val * 100.0))
    print('finish reload adv img')
    return num_train, num_val


def load_sp_data(net, dir_h5=None, batch_size=500):
    assert dir_h5 is not None
    with h5py.File(dir_h5, "r") as f:
        X_train_h5 = f["X_train"]
        num_train = len(X_train_h5)
        X_val_h5 = f["X_val"]
        num_val = len(X_val_h5)
        print('num_train: ', num_train)
        print('num_val: ', num_val)

    with h5py.File(dir_h5) as f:
        # reload training adv image
        for idx in range(0, num_train, batch_size):
            x_train = torch.from_numpy(f["X_train"][idx:idx + batch_size])
            y_train = torch.from_numpy(f["Y_train"][idx:idx + batch_size])
            x_perturb_sp = sp_attack(net, x_train.cuda(), y_train.cuda())
            f["X_train_sp"][idx:idx + batch_size] = x_perturb_sp
            print("Process Training Data: {0:.2f}%".format(idx / num_train * 100.0))
    print('finish reload adv img')
    return num_train, num_val


def gaussian_attack(net, x_batch, y_batch, branch_size=1):
    epsilon = 1e-4
    x_batch = x_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()
    batch_size = x_batch.shape[0]
    x_shape = x_batch[0].shape
    x_perturb = x_batch.copy()
    x_type = x_batch[0].dtype
    for idx in range(batch_size):
        for iter in range(0, 20):
            # draw noise pattern
            noise = np.random.uniform(-1.0, 1.0, size=x_shape)
            noise = noise.astype(x_type)

            # overlay noise pattern on image
            x_perturb[idx] = x_batch[idx] + epsilon * noise

            # clip pixel values to valid range [0, 1]
            x_perturb[idx] = np.clip(x_perturb[idx], 0, 1).astype(np.float32)
            logits = net(torch.from_numpy(x_perturb[idx]).cuda().view(1, 3, 64, 64))
            _, pred_y = logits.data.max(dim=2)
            pred_y_vote = majority_vote(pred_y, num_class=200, branch_size=branch_size).cpu().numpy()
            if pred_y_vote != y_batch[idx]:
                break
            else:
                epsilon *= 2.0
    return x_perturb


def sp_attack(net, x_batch, y_batch, branch_size=1, epsilons=100):
    x_batch = x_batch.cpu().numpy() * 255.0
    y_batch = y_batch.cpu().numpy()
    batch_size = x_batch.shape[0]
    x_type = x_batch[0].dtype

    x_perturb = x_batch.copy()

    axis = 0
    channels = 3
    shape = list(x_batch[0].shape)
    shape[axis] = 1
    pixels = np.prod(shape)

    min_ = 0.0
    max_ = 255.0
    r = max_ - min_

    epsilons = min(epsilons, pixels)
    max_epsilon = 5

    for idx in range(batch_size):
        for epsilon in np.linspace(0, max_epsilon, num=epsilons + 1)[1:]:
            p = epsilon

            u = np.random.uniform(size=shape)
            u = u.repeat(channels, axis=axis)

            salt = (u >= 1 - p / 2).astype(x_type) * r
            pepper = -(u < p / 2).astype(x_type) * r

            perturbed = x_batch[idx] + salt + pepper
            perturbed = np.clip(perturbed, min_, max_)
            # save perturbed to x_perturb[idx] with normalization
            x_perturb[idx] = perturbed.copy() / 255.0

            logits = net(torch.from_numpy(x_perturb[idx]).cuda().view(1, 3, 64, 64))
            _, pred_y = logits.data.max(dim=2)
            pred_y_vote = majority_vote(pred_y, num_class=200, branch_size=branch_size).cpu().numpy()
            if pred_y_vote != y_batch[idx]:
                break
    return x_perturb


def ClassMapping(map_file):
    """
      Create a dict for mapping the class name nxxxxxxx to int from 1 to 200
      the mapping is created using shell script. the purpose of this function is
      to read and save the maping in dict

      Args:
            map_file: map file path

      return:
            class_map: a dict. key is class name(nXXXXXXXX) and value is the
            corresponding number
      """
    class_map = dict()
    class_id = 0
    with open(map_file) as fin:
        for line in fin:
            list_t = line
            list_key = list_t[0:9]
            class_map[list_key] = int(class_id)
            class_id = class_id + 1
    return class_map


def read_train_images(input_dir):
    # data_list - save image
    data_list = list()
    # label_list - save label
    label_list = list()
    # class_map - map from class name(nXXXXXXXX) to value (0-199)
    class_map = ClassMapping("./wnids.txt")

    num_images = 0
    # load image from tiny-ImageNet
    for key in class_map.keys():
        # load original image
        path = input_dir + "/" + key + "/" + "images"
        for fi in os.listdir(path):
            if not (any(fi.endswith(ext) for ext in IMG_EXTENSIONS)):
                continue
            img = misc.imread(path + "/" + fi, mode='RGB')
            data_list.append(img)
            label_list.append(class_map[key])
            num_images += 1
            if num_images > 1000:
                break

    return np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.float32)




if __name__ == "__main__":
    print("Test")