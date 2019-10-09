
import os
import random
import cv2
import pathlib
import tensorflow as tf
tf.enable_eager_execution()

DEBUG = 0

def divide_set_into_subsets(image_path, label_path, sample_percentage):

    image_path.shuffle()
    n = len(image_path)

    subsets = []
    iterator = 0
    for i in range(sample_percentage):
        percent = iterator+(sample_percentage[i]*n)
        if percent > n:
            percent = n
        subsets.add(image_path[iterator:percent])
        iterator = percent
    return subsets


def load_and_preprocess_image(path,img_size):

    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size[0], img_size[1]])
    image /= 255.0  # normalize to [0,1] range
    image = 2 * image - 1

    return image

def augmentation(image):

    image = tf.image.random_flip_left_right(image)
    return image

def get_data_path_labels(dataset_path):

    data_root = pathlib.Path(dataset_path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    if DEBUG:
        print(str(image_count) + ' images found.')

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    if DEBUG:
        print('Label names:')
        print(label_names)

    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    if DEBUG:
        for n in range(5):
            image_path = all_image_paths[n]
            im = cv2.imread(image_path)
            im = cv2.resize(im,(224,224))
            cv2.imshow('CLASS: ' + str(all_image_labels[n]), im)
            cv2.waitKey(1)


    return all_image_paths, all_image_labels, label_names, label_to_index

# ----------- PARAMS ------------- #

BATCH_SIZE = 16
IMAGE_SIZE = [224, 224]
IMG_DIR = '/home/socialab/Downloads/PetImages/'
IMG_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

# ---------- LOAD PATHS AND LABELS

all_image_paths, all_image_labels, label_names, label_to_index = get_data_path_labels(IMG_DIR)
num_images = len(all_image_paths)



# ---------- CREATE GENERATORS

path_gen = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_gen = path_gen.map(lambda x: load_and_preprocess_image(x,IMAGE_SIZE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
image_gen = image_gen.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_gen = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_gen = tf.data.Dataset.zip((image_gen, label_gen))


train_size = int(0.7*num_images)
val_size = int(0.15*num_images)
test_size = int(0.15*num_images)

train_gen = image_label_gen.take(train_size)
test_gen = image_label_gen.skip(train_size)
val_gen = test_gen.skip(val_size)
test_gen = test_gen.take(test_size)

print(image_label_gen)

# for n,data in enumerate(image_label_gen.take(4)):
#   image = data[0]
#   label = data[1]
#   imagea =  tf.image.convert_image_dtype(image, dtype=tf.uint8).numpy()
#   cv2.imshow(str(label.numpy()),imagea)
#   cv2.waitKey(1)

# for label in label_ds.take(10):
#   print(label_names[label.numpy()])




# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
# ds = image_label_ds.shuffle(buffer_size=num_images)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
# # `prefetch` lets the dataset fetch batches, in the background while the model is training.
# ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# print(ds)

ds_train = train_gen.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
print(ds_train)

ds_val = val_gen.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=num_images))
ds_val = ds_val.batch(BATCH_SIZE)
ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
print(ds_val)





# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch_train, label_batch_train = next(iter(ds_train))
image_batch_val, label_batch_val = next(iter(ds_val))

net = tf.keras.applications.ResNet50(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False)
net.trainable = True

feature_map_batch = net(image_batch_train)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))])


logit_batch = model(image_batch_train).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

len(model.trainable_variables)

model.summary()

steps_per_epoch = tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
print(steps_per_epoch)

model.fit(ds_train, validation_data=ds_val, validation_steps=10, epochs=10, steps_per_epoch=steps_per_epoch)