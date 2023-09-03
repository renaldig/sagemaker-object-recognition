import os
import boto3
import json
import numpy as np
from io import BytesIO
from PIL import Image
import mxnet as mx
from mxnet import recordio

# Initialize S3 client
s3 = boto3.client('s3')

# Specify the temporary directory for storing images and annotation files
temp_dir = '/home/ec2-user'

# Download annotations files
annotations_train_file_path = os.path.join(temp_dir, 'annotations_train.json')
annotations_val_file_path = os.path.join(temp_dir, 'annotations_val.json')
s3.download_file('newobjectdetectionbucket2', 'unzipped/annotations_trainval2017/instances_train2017.json', annotations_train_file_path)
s3.download_file('newobjectdetectionbucket2', 'unzipped/annotations_trainval2017/instances_val2017.json', annotations_val_file_path)

# Load annotations files
with open(annotations_train_file_path) as f:
    annotations_train = json.load(f)
with open(annotations_val_file_path) as f:
    annotations_val = json.load(f)

# Create a map from image id to annotation id
def create_image_to_annotation_map(annotations):
    image_to_annotation = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_to_annotation:
            image_to_annotation[image_id] = []
        image_to_annotation[image_id].append(annotation)
    return image_to_annotation

image_to_annotation_train = create_image_to_annotation_map(annotations_train)
image_to_annotation_val = create_image_to_annotation_map(annotations_val)

# Create a map from category id to class id
category_to_class = {}
for i, category in enumerate(annotations_train['categories']):
    category_id = category['id']
    category_to_class[category_id] = i

# Function to process images and annotations and write them to the recordio file
def process_images_and_annotations(annotations, image_to_annotation, recordio_writer, img_folder):
    for image in annotations['images']:
        try:
            image_id = image['id']
            file_name = image['file_name']
            width = image['width']
            height = image['height']

            # Download the image from S3
            image_file = BytesIO()
            s3.download_fileobj('newobjectdetectionbucket2', f'unzipped/{img_folder}/' + file_name, image_file)
            image_file.seek(0)
            img = np.array(Image.open(image_file))

            # Get the annotations for this image
            annotations = image_to_annotation.get(image_id, [])

            # Convert the annotations to the format expected by the model
            header = mx.recordio.IRHeader(0, [annotation['category_id'] for annotation in annotations], image_id, 0)
            s = mx.recordio.pack_img(header, img, quality=100, img_fmt='.jpg')

            # Write the recordio file
            recordio_writer.write(s)
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

# Create recordio writers
recordio_writer_train = recordio.MXRecordIO(os.path.join(temp_dir, 'train.rec'), 'w')
recordio_writer_val = recordio.MXRecordIO(os.path.join(temp_dir, 'val.rec'), 'w')

# Process the training and validation images and annotations
process_images_and_annotations(annotations_train, image_to_annotation_train, recordio_writer_train, 'train2017')
process_images_and_annotations(annotations_val, image_to_annotation_val, recordio_writer_val, 'val2017')

# Close the recordio writers
recordio_writer_train.close()
recordio_writer_val.close()

# Upload the .rec files to S3
s3.upload_file(os.path.join(temp_dir, 'train.rec'), 'newobjectdetectionbucket2', 'recordio/train2/train.rec')
s3.upload_file(os.path.join(temp_dir, 'val.rec'), 'newobjectdetectionbucket2', 'recordio/validation2/val.rec')

# Remove the temporary files
os.remove(annotations_train_file_path)
os.remove(annotations_val_file_path)
os.remove(os.path.join(temp_dir, 'train.rec'))
os.remove(os.path.join(temp_dir, 'val.rec'))

print('Data conversion complete.')
