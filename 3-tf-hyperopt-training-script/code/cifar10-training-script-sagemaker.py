#%%
import tensorflow as tf
import argparse
import os
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam, SGD

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

# Copy inference pre/post-processing script so that it'll be included in the model package
os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')

#%%
def train_preprocess_fn(image):

    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


#%%
def make_batch(filenames, batch_size):
    """Read the images and labels from 'filenames'."""
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(single_example_parser, num_parallel_calls=1)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()

    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


#%%
def single_example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    image = train_preprocess_fn(image)
    label = tf.one_hot(label, NUM_CLASSES)
    
    return image, label


#%%
def cifar10_model(input_shape):

    input_tensor = Input(shape=input_shape)
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights='imagenet',
                                                      input_tensor=input_tensor,
                                                      input_shape=input_shape,
                                                      classes=None)

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(10, activation='softmax')(x)
    mdl = Model(inputs=base_model.input, outputs=predictions)
    return mdl


#%%
def main(args):
    # Hyper-parameters
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    momentum = args.momentum
    weight_decay = args.weight_decay
    optimizer = args.optimizer

    # SageMaker options
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    train_dir = args.train
    validation_dir = args.validation
    eval_dir = args.eval

    train_dataset = make_batch(train_dir+'/train.tfrecords',  batch_size)
    val_dataset = make_batch(validation_dir+'/validation.tfrecords', batch_size)
    eval_dataset = make_batch(eval_dir+'/eval.tfrecords', batch_size)

    input_shape = (HEIGHT, WIDTH, DEPTH)
    model = cifar10_model(input_shape)

    # Multi-GPU training
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    # Optimizer
    if optimizer.lower() == 'sgd':
        opt = SGD(lr=lr, decay=weight_decay, momentum=momentum)
    else:
        opt = Adam(lr=lr, decay=weight_decay)

    # Compile model
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(x=train_dataset[0], y=train_dataset[1],
                        steps_per_epoch=40000 // batch_size,
                        validation_data=val_dataset,
                        validation_steps=10000 // batch_size,
                        epochs=epochs)

    # Evaluate model performance
    score = model.evaluate(eval_dataset[0],
                           eval_dataset[1],
                           steps=10000 // args.batch_size,
                           verbose=0)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    # Save model to model directory
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,   default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size',    type=int,   default=128)
    parser.add_argument('--weight-decay',  type=float, default=2e-4)
    parser.add_argument('--momentum',      type=float, default='0.9')
    parser.add_argument('--optimizer',     type=str,   default='sgd')

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str,   default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--gpu-count',        type=int,   default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--train',         type=str,   default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)
