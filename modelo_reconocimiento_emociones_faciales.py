#=========================1: Imports =========================
from keras.preprocessing.image import ImageDataGenerator
# Nos permite importar imagenes.

from keras.models import Sequential
# Sequential sirve para hacer una red neuronal sequencial.

from keras.layers import Conv2D
# Estamos trabajando con imagenes que son 2D

from keras.layers import MaxPooling2D
# Nos permite usar Max Pooling.

from keras.layers import Flatten
# Flatten nos deja convertir la imagen 2D en un array.

from keras.layers import Dense
# Dense nos permite crear una red neuronal donde todas las neuronas estan conectadas entre si. 

from keras.layers import Dropout
# Droput nos permite de manera aleatoria seleccionar un subset de neuronas y removerlo durante el entrenamiento.

from keras.optimizers import Adam
# Adam es un optimizador que nos permite el uso de gradientes durante el entrenamiento.

from keras.callbacks import EarlyStopping
# Early Stopping es un optimizador que detecta cuando una metrica deja de mejorar durante el entrenamiento y 
# reduce el tiempo de entrenamiento para no sobre entrenar la IA en base a ese parametro.

from keras.models import save_model

#=========================2: Definimos la data de entrenamiento =========================
# Definimos los directorios
train_dir = './training_set'
test_dir = './test_set'

# Creamos el generador de imagenes para recibir la data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Importamos la data de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

#=========================3: Creamos el modelo =========================

# Definimos el modelo CNN
modelo = Sequential()

# Agregamos las neuronas convolucionales
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Conv2D(128, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(6, activation='softmax'))

# Compilamos el modelo.
modelo.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Imprimimos un resumen del modelo
modelo.summary()

# Agregamos los callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamos el modelo
modelo.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=500,
    callbacks=[early_stopping]
)

# Guardamos el modelo
modelo.save("Reconocimiento_Emociones_Faciales.h5")

# Evaluamos el modelo
loss, accuracy = modelo.evaluate(test_generator)
print(f'Test accuracy: {accuracy * 100:.2f}%')