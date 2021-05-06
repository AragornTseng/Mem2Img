import numpy
import os
from PIL import Image
from torchvision import transforms as tr
import sys
import random
import argparse
random.seed(42)


def main(argv=[]):

    parser = argparse.ArgumentParser(description="argumentation of the png")
    parser.add_argument('input directory', metavar='input_dir', help='path of input directory')
    args = parser.parse_args()
    input_dir = sys.argv[1]
    transforms_pipeline = tr.Compose([tr.RandomHorizontalFlip(p = 0.4),tr.RandomVerticalFlip(p = 0.8)])
    transforms_pipeline1 = tr.RandomRotation((90,270))
    transforms_pipeline2 = tr.RandomResizedCrop(size=224,scale=(0.5,1.0))

    for i, d in enumerate(os.listdir(input_dir)):    
        for f in os.listdir(os.path.join(input_dir, d)):
            filepath = os.path.join(input_dir, d, f)
            img = Image.open(os.path.join(input_dir,d,f))
            flip_image = transforms_pipeline(img = img)
            pipeline_name = os.path.join(os.path.splitext(filepath)[0]+'_flip.png')
            flip_image.save(pipeline_name)
            rotate_image = transforms_pipeline1(img = img)
            pipeline1_name = os.path.join(os.path.splitext(filepath)[0]+'_rotate.png')
            rotate_image.save(pipeline1_name)
            scale_image = transforms_pipeline2(img = img)
            pipeline2_name = os.path.join(os.path.splitext(filepath)[0]+'_scale.png')
            scale_image.save(pipeline2_name)


if __name__ == '__main__':
    main(sys.argv)

