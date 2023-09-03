import boto3
import zipfile
import os

# Initialize the S3 client
s3 = boto3.client('s3')

# Specify your bucket names
source_bucket = 'newobjectdetectionbucket'
target_bucket = 'newobjectdetectionbucket2'

# Specify your file keys
file_keys = ['train2017.zip', 'val2017.zip', 'annotations_trainval2017.zip']

for file_key in file_keys:
    # Download the file from S3
    s3.download_file(source_bucket, file_key, '/home/ec2-user/' + file_key)

    # Create a folder to unzip the files to
    folder_name = file_key.replace('.zip', '')
    os.makedirs('/home/ec2-user/unzipped/' + folder_name, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile('/home/ec2-user/' + file_key, 'r') as zip_ref:
        zip_ref.extractall('/home/ec2-user/unzipped/' + folder_name)

    # Upload the unzipped files back to S3
    for subdir, _, files in os.walk('/home/ec2-user/unzipped/' + folder_name):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                s3.upload_fileobj(data, target_bucket, 'unzipped/' + folder_name + '/' + file)

    # Remove the unzipped files from the EC2 instance
    for subdir, _, files in os.walk('/home/ec2-user/unzipped/' + folder_name):
        for file in files:
            os.remove(os.path.join(subdir, file))

    # Remove the downloaded zip file
    os.remove('/home/ec2-user/' + file_key)

print('Files unzipped successfully')
