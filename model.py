import csv
import matplotlib.image as mpimg
import numpy as np

def read_in_data(file,lines):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

def image_measurement(list_name, path,image_list, measure_list):
    for line in list_name:
        camera = np.random.choice(['center', 'left', 'right'])
        measurement = float(line[3])
        if camera == 'center':
            source_path = line[0]
        elif camera == 'left':
            source_path = line[1]
            measurement += 0.2
        else:
            source_path = line[2]
            measurement -= 0.2
        file_name = source_path.split('/')[-1]
        current_path = path + file_name
        image = mpimg.imread(current_path)
        image_list.append(image)    
        measure_list.append(measurement)
        
lines_manual = []
lines_Udacity = []
file1 = 'Train_data_1/driving_log.csv'  #readin dataset focusing on recovery manuever
read_in_data(file1,lines_manual)
file2 = 'data/driving_log.csv'       #readin default dataset
read_in_data(file2,lines_Udacity)
lines_Udacity = lines_Udacity[1:]

images = []
measurements =[]
path_manual = 'Train_data_1/IMG/'
path_Udacity = 'data/IMG/'
image_measurement(lines_manual, path_manual, images, measurements)
image_measurement(lines_Udacity, path_Udacity, images, measurements)


augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Convolution2D,Flatten, Dropout, Dense, Lambda, MaxPooling2D, Cropping2D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5 ,input_shape =  (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

#Nvidia Model

model.add(Convolution2D(24,(5,5),strides = (2,2),activation = "relu"))
model.add(Convolution2D(36,(5,5),strides = (2,2),activation = "relu"))
model.add(Convolution2D(48,(5,5),strides = (2,2),activation = "relu"))
model.add(Convolution2D(64,(3,3),strides = (1,1),activation = "relu"))
model.add(Convolution2D(64,(3,3),strides = (1,1),activation = "relu"))

model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.0005, verbose = 0, mode='min')   

history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 7,callbacks=[early_stopping] )

model.save('model.h5')   #save model
print("Model Saved")


#python drive.py model.h5  #use this line for autonomous driving

    