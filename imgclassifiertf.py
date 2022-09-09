import sys,os
import logging

import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info('that should work :-)')

# An S3 Bucket Name
data_bucket_name='sagemakertestwat-dev'

# A prefix name inside the S3 bucket containing sub-folders of images (one per label class)
dataset_name = 'data' 

logger.info("data bucket name is: "+data_bucket_name);
logger.info("dataset name is: "+dataset_name);

role = get_execution_role()
sess = sagemaker.Session()

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")

logger.info(training_image)

# Find im2rec in our environment and set up some other vars in our environemnt

base_dir='/tmp'

%env BASE_DIR=$base_dir
%env S3_DATA_BUCKET_NAME = $data_bucket_name
%env DATASET_NAME = $dataset_name

suffix='/mxnet/tools/im2rec.py'
im2rec = list(filter( (lambda x: os.path.isfile(x + suffix )), sys.path))[0] + suffix
logger.info("im2rec is: "+im2rec)
%env IM2REC=$im2rec

# Pull our images from S3
!aws s3 sync s3://$S3_DATA_BUCKET_NAME/public/$DATASET_NAME $BASE_DIR/$DATASET_NAME --quiet
  
 %%bash
# Use the IM2REC script to convert our images into RecordIO files

# Clean up our working dir of existing LST and REC files
#cd $BASE_DIR
#rm *.rec
#rm *.lst

# First we need to create two LST files (training and test lists), noting the correct label class for each image
# We'll also save the output of the LST files command, since it includes a list of all of our label classes
#echo "Creating LST files"
python $IM2REC --list --recursive --pass-through --test-ratio=0.3 --train-ratio=0.7 $DATASET_NAME $DATASET_NAME > ${DATASET_NAME}_classes

echo "Label classes:"
cat ${DATASET_NAME}_classes

# Then we create RecordIO files from the LST files

echo "Creating RecordIO files"
python $IM2REC --num-thread=4 ${DATASET_NAME}_train.lst $DATASET_NAME
python $IM2REC --num-thread=4 ${DATASET_NAME}_test.lst $DATASET_NAME
ls -lh *.rec
  
# Upload our train and test RecordIO files to S3 in the bucket that our sagemaker session is using
bucket = sess.default_bucket()
#bucket = 'rekognitionbucket1'

s3train_path = 's3://{}/{}/train/'.format(bucket, dataset_name)
s3validation_path = 's3://{}/{}/validation/'.format(bucket, dataset_name)

logger.info("s3 file training path is: "+s3train_path)
logger.info("s3 file validation path is: "+s3validation_path)

# Clean up any existing data
#!aws s3 rm s3://{bucket}/{dataset_name}/train --recursive
#!aws s3 rm s3://{bucket}/{dataset_name}/validation --recursive

# Upload the rec files to the train and validation channels
#!aws s3 cp /tmp/{dataset_name}_train.rec $s3train_path
#!aws s3 cp /tmp/{dataset_name}_test.rec $s3validation_path

train_data = sagemaker.session.s3_input(
    s3train_path, 
    distribution='FullyReplicated', 
    content_type='application/x-recordio', 
    s3_data_type='S3Prefix'
)

validation_data = sagemaker.session.s3_input(
    s3validation_path, 
    distribution='FullyReplicated', 
    content_type='application/x-recordio', 
    s3_data_type='S3Prefix'
)

data_channels = {'train': train_data, 'validation': validation_data}

s3_output_location = 's3://{}/{}/output'.format(bucket, dataset_name)

image_classifier = sagemaker.estimator.Estimator(
    training_image,
    role, 
    train_instance_count=1, 
    train_instance_type='ml.p3.2xlarge',
    output_path=s3_output_location,
    sagemaker_session=sess
)

logger.info("image classifier params are: "+image_classifier)

#num_classes=! ls -l {base_dir}/{dataset_name} | wc -l
#num_classes=! ls -l {base_dir}/{dataset_name} | wc -l

#num_training_samples=! cat {base_dir}/{dataset_name}_train.lst | wc -l
#num_training_samples = 3504

num_classes=! ls -l {base_dir}/{dataset_name} | wc -l
num_classes=int(num_classes[0]) - 1

logger.info("number of classes is: "+num_classes)

num_training_samples=! cat {base_dir}/{dataset_name}_train.lst | wc -l
num_training_samples = int(num_training_samples[0])

logger.info("number of training samples is: "+num_training_samples)

# Learn more about the Sagemaker built-in Image Classifier hyperparameters here: https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html

# These hyperparameters we won't want to change, as they define things like
# the size of the images we'll be sending for input, the number of training classes we have, etc.
base_hyperparameters=dict(
    use_pretrained_model=1,
    num_layers=200,
    image_shape='3,150,150',
    epochs=1000,
    augmentation_type='crop',
    optimizer='nag',
    num_classes=num_classes,
    num_training_samples=num_training_samples,
)

logger.info("base parameters are: "+base_hyperparameters)

# These are hyperparameters we may want to tune, as they can affect the model training success:
hyperparameters={
    **base_hyperparameters, 
    **dict(
        learning_rate=0.01,
        mini_batch_size=42,
    )
}


image_classifier.set_hyperparameters(**hyperparameters)

logger.info("hyperparameters params are: "+hyperparameters)
hyperparameters

%%time

import time
now = str(int(time.time()))
training_job_name = 'IC-' + dataset_name.replace('_', '-') + '-' + now

image_classifier.fit(inputs=data_channels, job_name=training_job_name, logs=True)

job = image_classifier.latest_training_job
model_path = f"/tmp/{job.name}"

logger.info(f"\n\n Finished training! The model is available for download at: {image_classifier.output_path}/{job.name}/output/model.tar.gz")
