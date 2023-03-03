import albumentations as alb
import os
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

# image dimensions
height = 1024
width = 768

augmentor = alb.Compose([alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))



def main():
    for partition in ['train','test']: 
      for file in os.listdir(os.path.join('images',partition)):
          if ".jpg" in file:
              img = cv2.imread(os.path.join('images', partition, file))
              coords = [0,0,0,0]
              label_path = os.path.join('images', partition, f'{file.split(".")[0]}.json')
              if os.path.exists(label_path):
                  with open(label_path, 'r') as f:
                      label = json.load(f)
              cls_labels = [label['shapes'][i]['label'] for i in range (0,len(label['shapes']))]
              bboxes_all = []
              for i in range(0,len(label['shapes'])):
                  coords = [0,0,0,0]
                  coords[0] = label['shapes'][i]['points'][0][0] / width
                  coords[1] = label['shapes'][i]['points'][0][1] / height
                  coords[2] = label['shapes'][i]['points'][1][0] / width
                  coords[3] = label['shapes'][i]['points'][1][1] / height
                  ordered = [min(coords[0],coords[2]),min(coords[1],coords[3]),max(coords[0],coords[2]),max(coords[1],coords[3])]
                  bboxes_all.append(ordered)

              for x in range(5):
                  augmented = augmentor(image=img, bboxes=bboxes_all, class_labels=cls_labels)
                  cv2.imwrite(os.path.join('aug_images', partition, f'{file.split(".")[0]}.{x}.jpg'), augmented['image'])
                  annotations = {}
                  annotations['image'] = file
                  annotations['shapes'] = []

                  for i in range(0,len(label['shapes'])):
                      label_dict = {}
                      label_dict['label'] = cls_labels[i]
                      label_dict['bbox'] = bboxes_all[i]
                      annotations['shapes'].append(label_dict)
                      
                  with open(os.path.join('aug_images', partition, f'{file.split(".")[0]}.{x}.json'), 'w') as f:
                      json.dump(annotations, f)

if __name__ == "__main__":
    main()