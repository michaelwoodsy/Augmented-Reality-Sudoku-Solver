import tensorflow as tf

batch_size = 16
nb_epochs = 5 # number of iterations to run

num_classes = 10 # 10 classes because we include zero
input_shape = (64, 64, 1) # 64 because images are 64 x 64 pixels

# Our CNN Model
model = tf.keras.models.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=input_shape),
  tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Loads the training data from our custom dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./dataset",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(64, 64),
  color_mode="grayscale",
  batch_size=32
)

# Loads the validation data from our custom dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./dataset",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(64, 64),
  color_mode="grayscale",
  batch_size=32
)

# Fits the model based on our custom dataset
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=nb_epochs,
)

# Saves model to JSON
model_json = model.to_json()
with open("./model/model.json", "w") as json_file:
    json_file.write(model_json)

# Saves weights to H5
model.save_weights("./model/weights.h5")
print("Saved model to disk")