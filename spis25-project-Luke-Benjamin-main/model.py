from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


images_dir = "/home/linnarelli/spis25-project-Luke-Benjamin/ASL_Alphabet_Train"
images_dir2 = "/home/linnarelli/spis25-project-Luke-Benjamin/close_up_letters/close_up_letters"


training_dataset, testing_dataset = tf.keras.utils.image_dataset_from_directory(
    images_dir,
    labels="inferred",
    label_mode="int",
    image_size= (128,128),
    batch_size=4,
    seed=123,
    validation_split=0.2,
    shuffle=True,
    subset="both",
)

class_names = training_dataset.class_names
training_dataset = training_dataset.map(preprocess)
testing_dataset = testing_dataset.map(preprocess)

training_dataset2, testing_dataset2 = tf.keras.utils.image_dataset_from_directory(
    images_dir2,
    labels="inferred",
    label_mode="int",
    image_size= (128,128),
    batch_size=4,
    seed=42,
    validation_split=0.2,
    shuffle=True,
    subset="both",
)


training_dataset = training_dataset.concatenate(training_dataset2)
testing_dataset = testing_dataset.concatenate(testing_dataset2)

num_classes = 29


# all_labels = []

# for _, labels in training_dataset.unbatch():
#     all_labels.append(int(labels.numpy()))

# class_weights_array = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(all_labels),
#     y=all_labels
# )
# class_weights = dict(enumerate(class_weights_array))

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size= (3,3), padding= 'same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, kernel_size= (3,3), padding= 'same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, kernel_size= (3,3), padding= 'same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(256, kernel_size= (3,3), padding= 'same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(512, kernel_size= (3,3), padding= 'same', activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(num_classes, activation = 'softmax')
])

'''
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)

])
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
training_dataset = training_dataset.shuffle(500).prefetch(tf.data.AUTOTUNE)
testing_dataset = testing_dataset.prefetch(tf.data.AUTOTUNE)
model.fit(training_dataset, validation_data = testing_dataset, epochs = 2)

model.save("Testing_model_with_3photos7.keras")
