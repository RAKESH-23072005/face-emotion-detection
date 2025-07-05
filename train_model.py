import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from load_data import train_generator, test_generator

#models CNN
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 = émotions
])

#compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate scheduler callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)
lr_reduce = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

model.fit(
    train_generator,
    epochs=40,
    validation_data=test_generator,
    callbacks=[early_stop, lr_reduce]
)

os.makedirs('model', exist_ok=True)
model.save('model/emotion_model.keras')
print("Model trained and saved as emotion_model.keras")