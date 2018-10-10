import foolbox
# from model import create_model
from fmodel import *
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete
import numpy as np
from utils import *


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

    input_dir = '/home/hongyang/data/tiny-imagenet-200-aug/tiny-imagenet-200/train'
    Images, Labels = read_train_images(input_dir)
    print("Images.shape: ", Images.shape)
    for idx in range(100):
        # image is a numpy array with shape (64, 64, 3)
        # and dtype float32, with values between 0 and 255;
        # label is the original label (for untargeted
        # attacks) or the target label (for targeted attacks)
        # adversarial = run_attack(model, image, label)
        # store_adversarial(file_name, adversarial)
        adversarial = run_attack(model=model, image=Images[idx], label=Labels[idx].reshape([1, ]))
        store_adversarial(file_name, adversarial)

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    attack_complete()


if __name__ == '__main__':
    main()
