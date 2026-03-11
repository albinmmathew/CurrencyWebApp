import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ----------------------------
# PARAMETERS
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

BASE_PATH = "dataset/Indian currency dataset v1"
TRAIN_PATH = os.path.join(BASE_PATH, "training")
VAL_PATH = os.path.join(BASE_PATH, "validation")

# ----------------------------
# DATA GENERATORS
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save class indices
with open("class_indices.txt", "w") as f:
    f.write(str(train_generator.class_indices))

# ----------------------------
# LOAD PRETRAINED MODEL
# ----------------------------
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False

# ----------------------------
# CUSTOM CLASSIFIER
# ----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes,
                    activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("currency_model.h5")

print("Training complete. Model saved as currency_model.h5")