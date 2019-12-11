import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D
from scipy import ndimage

# Enabling/Disabling switches
ENABLE_SIDE_IMAGES = 1
ENABLE_FLIPPING_IMAGES = 1
DISPLAY_ONE_IMAGE = 1
ENABLE_FOR_PC=1

# Hyper parameters to fine tune
input_shape = (160, 320, 3)
num_of_epochs = 14
conv_filters = 6
conv_kernel_size = (5, 5)
dropout_rate = 0.01
validation_split = 0.2  # 20% Validation and 80% Training
correction_factor = 0.2
correction = [0.0, correction_factor, -correction_factor]  # [center, left, right]

# Global lists
lines = []
images = []
measurements = []


def display_image(ori_img, new_img, new_img_name):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(ori_img)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(new_img)
    ax2.set_title(new_img_name, fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('flipped_image.jpg')


print('Reading driving_log.csv file')
if ENABLE_FOR_PC:
    comp_path = 'C:\Self_Driving_Course\Behavioural_Cloning\Training_Data\\'
else:
    comp_path = 'Training_Data/'

with open(comp_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print('Parsed the csv file')

for line in lines:
    for index in range(0, 3):
        path = line[index]
        filename = path.split('\\')[-1]
        image_path = comp_path + '/IMG/' + filename
        # filename = path.split('\\')[-1]
        # image_path = 'C:\Self_Driving_Course\Behavioural_Cloning\Training_Data\IMG\\' + filename
        # Using ndimage as the cv2.imread reads out in BGR format while the drive.py loads in RGB format
        image = ndimage.imread(image_path)
        images.append(image)
        measurement = float(line[3]) + correction[index]
        measurements.append(measurement)
        # Flip the image and append its image as well so that training is not left biased
        if ENABLE_FLIPPING_IMAGES:
            images.append(np.fliplr(image))
            measurements.append(measurement * (-1.0))
            if DISPLAY_ONE_IMAGE:
                display_image(image, np.fliplr(image), 'Flipped Image')
                DISPLAY_ONE_IMAGE = 0

        if not ENABLE_SIDE_IMAGES:
            break

print('Images and its Measurements added to the list')

# Create a numpy array of features (X_train) and labels (Y_train)
X_train = np.array(images)
Y_train = np.array(measurements)

# Create the model here
model = Sequential()

# Data PreProcessing
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

model.add(Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.add(Dropout(rate=dropout_rate))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=validation_split, shuffle=True, epochs=num_of_epochs)
model.save('my_model_mod.h5')

print('model saved')
