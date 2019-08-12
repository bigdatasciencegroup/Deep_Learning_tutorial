"""
해당 코드는 Data 개수가 적어서 overfitting이 발생.
이 문제를 해결하기 위해서 DataAugmentation을 실행해주는데.
이 적용 코드는 train_augmentation에서 실시.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import make_dir as md
from model import model

"""Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR 해결 책."""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#gpu 메모리 할당 문제....
"""=============================="""

#print(len(os.listdir(md.train_cats_dir)))

print(type(md.train_cats_dir))




train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_fr
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)om_directory(
    md.train_dir,
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
    target_size = (150, 150), #--> 모든
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50) 이미지를 150x150 으로 변경
    batch_size = 10,
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
    class_mode = 'binary' #--> oprimize
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)r 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
validation_generator = train_datagen.fl
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)ow_from_directory(
    md.validation_dir,
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
    target_size=(150, 150),
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
    batch_size=10,
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
    class_mode = 'binary'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    md.train_dir,
    target_size = (150, 150), #--> 모든 이미지를 150x150 으로 변경
    batch_size = 10,
    class_mode = 'binary' #--> oprimizer 부분에서 binary crossentropy loss를 사용해서 이진 레이블이 필요하다.
)

validation_generator = train_datagen.flow_from_directory(
    md.validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

#########그래프 그리기#############

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


