#!/usr/bin/env python3
import foolbox
import numpy as np
# from model import create_model
from fmodel import create_fmodel
# from adversarial_vision_challenge import store_adversarial
# from adversarial_vision_challenge import attack_complete
# from utils import read_train_images


def run_attack(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    # attack = foolbox.attacks.IterativeGradientAttack(model, criterion)
    # attack = foolbox.attacks.IterativeGradientSignAttack(model, criterion)
    attack = foolbox.attacks.CarliniWagnerL2Attack(model, criterion)
    return attack(image, label)


def main():
    # forward_model = load_model()
    forward_model = create_fmodel()
    backward_model = create_fmodel()

    model = foolbox.models.CompositeModel(
        forward_model=forward_model,
        backward_model=backward_model)

    # input_dir = '/home/hongyang/data/tiny-imagenet-200-aug/tiny-imagenet-200/train'
    # Images, Labels = read_train_images(input_dir)
    # print("Images.shape: ", Images.shape)
    image, _ = foolbox.utils.imagenet_example((64, 64))
    label = np.argmax(model.predictions(image))  # just for debugging

    for idx in range(1):
        # image is a numpy array with shape (64, 64, 3)
        # and dtype float32, with values between 0 and 255;
        # label is the original label (for untargeted
        # attacks) or the target label (for targeted attacks)
        # adversarial = run_attack(model, image, label)
        # store_adversarial(file_name, adversarial)
        adversarial = run_attack(model=model, image=image, label=label)
        if adversarial is None:
            print('attack failed')
        else:
            print('attack found adversarial')
        # store_adversarial(file_name, adversarial)

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    # attack_complete()


if __name__ == '__main__':
    main()
