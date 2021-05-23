"""
CNG 483
Project 1
Developed by PIM-Tech

Pınar Dilbaz 2243392
İbrahim Aydın 2151835
Muhammed Didin 2243384
"""

import cv2
from math import sqrt


# this function takes the img data send them to a prediction algorith that uses KNN Clasification method
# it takes both valid set and test set
# You can change the desired set to run in the for loop
# Returns a list of scores that contains prediction value and real value
def evaluate_algorithm(dataset, valid, test, algorithm, *args):
    scores = list()
    for valid_point in test:
        predicted = algorithm(dataset, valid_point, *args)
        actual = valid_point[-1]
        predicted.append(actual)
        scores.append(predicted)
    return scores


# calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1) - 1):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)


# takes the training set , a point and number of neighbors
# Calculate distance between current point and other points in the training list
# Sort the distance list and takes closest k neighbors
def get_neighbors(train, test_point, num_neighbors):
    distances = list()
    for train_point in train:
        dist = euclidean_distance(test_point, train_point)
        distances.append((train_point, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# look the k closest neighbors class value  of  the test point
# count them and and choose the most repeated class as prediction
def predict_classification(train, test_point, num_neighbors):
    neighbors = get_neighbors(train, test_point, num_neighbors)
    output_values = [point[-1] for point in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Start of the KNN algorithm
# returns the class prediction
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    output = predict_classification(train, test, num_neighbors)
    predictions.append(output)
    return (predictions)


# Splits  image to 4 part
def fold_image(img):
    height = img.shape[0] // 2
    weight = img.shape[1] // 2

    imgList = list()

    imgList.append(img[:height, :weight])
    imgList.append(img[:height, weight:])
    imgList.append(img[height:, :weight])
    imgList.append(img[height:, weight:])

    return imgList


# Splits  image to 16 part
def fold_image_sixteen(img):
    height = img.shape[0] // 2
    weight = img.shape[1] // 2

    img_temp_list = fold_image(img)
    img_list = list()
    for item in img_temp_list:
        img_list.append(fold_image(item))

    return img_list


# Read Dataset
# Transform image to grayscale
# Take gray scale histogram of each image to store as a list of histogram
def read_gray_scale_data_level1(trainArr, validArr, testArr):
    bins = [13]
    mode = [0]
    str_cloudy = "cloudy"
    str_sunrise = "sunrise"
    str_shine = "shine"

    ####
    #### Cloudy
    ####
    ####################################################################
    ## Train Array
    ################
    count = 1
    while (count != 151):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(0)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ####################
    while (count != 226):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(0)
        validArr.append(hist)
        count += 1
    ## Test Array
    ####################
    while (count != 301):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(0)
        testArr.append(hist)
        count += 1

    ####
    #### Shine
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 127):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(1)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ######################
    while (count != 190):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        if (img is None):
            count += 1
            continue
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(1)
        validArr.append(hist)
        count += 1
    ## Test Array
    ######################
    while (count != 253):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(1)
        testArr.append(hist)
        count += 1

    ####
    #### Sunrise
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 179):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(2)
        trainArr.append(hist)
        count += 1

    ## Validate Array
    ######################
    while (count != 269):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(2)
        validArr.append(hist)
        count += 1

    ## Test Array
    ######################
    while (count != 357):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, mode, None, bins, [0, 256]).flatten().tolist()
        hist.append(2)
        testArr.append(hist)
        count += 1

    ################################################################################


# Read Dataset
# Transform image to grayscale
# Split each image to 4 part
# Take gray scale histogram of each part and concatenate them
# Store the concatenated histograms as a list
def read_gray_scale_data_level2(trainArr, validArr, testArr):
    bins = [13]
    mode = [0]
    str_cloudy = "cloudy"
    str_sunrise = "sunrise"
    str_shine = "shine"

    ####
    #### Cloudy
    ####
    ####################################################################
    ## Train Array
    ################
    count = 1
    while (count != 151):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(0)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ####################
    while (count != 226):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(0)
        validArr.append(hist)
        count += 1
    ## Test Array
    ####################
    while (count != 301):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(0)
        testArr.append(hist)
        count += 1

    ####
    #### Shine
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 127):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(1)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ######################
    while (count != 190):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        if (img is None):
            count += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(1)
        validArr.append(hist)
        count += 1
    ## Test Array
    ######################
    while (count != 253):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(1)
        testArr.append(hist)
        count += 1

    ####
    #### Sunrise
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 179):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(2)
        trainArr.append(hist)
        count += 1

    ## Validate Array
    ######################
    while (count != 269):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(2)
        validArr.append(hist)
        count += 1

    ## Test Array
    ######################
    while (count != 357):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(2)
        testArr.append(hist)
        count += 1

    ################################################################################


# Read Dataset
# Transform image to grayscale
# Split each image to 16 part
# Take gray scale histogram of each part and concatenate them
# Store the concatenated histograms as a list
def read_gray_scale_data_level3(trainArr, validArr, testArr):
    bins = [14]
    mode = [0]
    str_cloudy = "cloudy"
    str_sunrise = "sunrise"
    str_shine = "shine"

    ####
    #### Cloudy
    ####
    ####################################################################
    ## Train Array
    ################
    count = 1
    while (count != 151):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(0)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ####################
    while (count != 226):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(0)
        validArr.append(hist)
        count += 1
    ## Test Array
    ####################
    while (count != 301):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(0)
        testArr.append(hist)
        count += 1

    ####
    #### Shine
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 127):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(1)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ######################
    while (count != 190):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        if (img is None):
            count += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(1)
        validArr.append(hist)
        count += 1
    ## Test Array
    ######################
    while (count != 253):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(1)
        testArr.append(hist)
        count += 1

    ####
    #### Sunrise
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 179):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(2)
        trainArr.append(hist)
        count += 1

    ## Validate Array
    ######################
    while (count != 269):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(2)
        validArr.append(hist)
        count += 1

    ## Test Array
    ######################
    while (count != 357):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list = fold_image_sixteen(gray)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, mode, None, bins, [0, 256]).flatten().tolist())
        hist.append(2)
        testArr.append(hist)
        count += 1

    ################################################################################


# Read Dataset
# Take the histogram of combination of r,g,b channels histogram
# Store the histograms as a list
def read_color_scale_data_level1(trainArr, validArr, testArr):
    str_cloudy = "cloudy"
    str_sunrise = "sunrise"
    str_shine = "shine"

    ####
    #### Cloudy
    ####
    ####################################################################
    ## Train Array
    ################
    count = 1
    while (count != 151):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(0)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ####################
    while (count != 226):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(0)
        validArr.append(hist)
        count += 1
    ## Test Array
    ####################
    while (count != 301):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(0)
        testArr.append(hist)
        count += 1

    ####
    #### Shine
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 127):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(1)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ######################
    while (count != 190):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        if (img is None):
            count += 1
            continue
        # print(img)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(1)
        validArr.append(hist)
        count += 1
    ## Test Array
    ######################
    while (count != 253):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(1)
        testArr.append(hist)
        count += 1

    ####
    #### Sunrise
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 179):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(2)
        trainArr.append(hist)
        count += 1

    ## Validate Array
    ######################
    while (count != 269):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(2)
        validArr.append(hist)
        count += 1

    ## Test Array
    ######################
    while (count != 357):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        hist = cv2.calcHist([img], [0, 1, 2],
                            None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        hist.append(2)
        testArr.append(hist)
        count += 1

    ################################################################################


# Read Dataset
# Split each image to 4 part
# Take the histogram of combination of r,g,b channels histogram for each part and concatenate them
# Store the concatenated histograms as a list
def read_color_scale_data_level2(trainArr, validArr, testArr):
    str_cloudy = "cloudy"
    str_sunrise = "sunrise"
    str_shine = "shine"

    ####
    #### Cloudy
    ####
    ####################################################################
    ## Train Array
    ################
    count = 1
    while (count != 151):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(0)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ####################
    while (count != 226):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(0)
        validArr.append(hist)
        count += 1
    ## Test Array
    ####################
    while (count != 301):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(0)
        testArr.append(hist)
        count += 1

    ####
    #### Shine
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 127):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(1)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ######################
    while (count != 190):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        if (img is None):
            count += 1
            continue
        # print(img)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(1)
        validArr.append(hist)
        count += 1
    ## Test Array
    ######################
    while (count != 253):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(1)
        testArr.append(hist)
        count += 1

    ####
    #### Sunrise
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 179):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(2)
        trainArr.append(hist)
        count += 1

    ## Validate Array
    ######################
    while (count != 269):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(2)
        validArr.append(hist)
        count += 1

    ## Test Array
    ######################
    while (count != 357):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(2)
        testArr.append(hist)
        count += 1

    ################################################################################


# Read Dataset
# Split each image to 16 part
# Take the histogram of combination of r,g,b channels histogram for each part and concatenate them
# Store the concatenated histograms as a list
def read_color_scale_data_level3(trainArr, validArr, testArr):
    str_cloudy = "cloudy"
    str_sunrise = "sunrise"
    str_shine = "shine"

    ####
    #### Cloudy
    ####
    ####################################################################
    ## Train Array
    ################
    count = 1
    while (count != 151):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(0)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ####################
    while (count != 226):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(0)
        validArr.append(hist)
        count += 1
    ## Test Array
    ####################
    while (count != 301):
        image_name = str_cloudy + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_cloudy + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(0)
        testArr.append(hist)
        count += 1

    ####
    #### Shine
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 127):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(1)
        trainArr.append(hist)
        count += 1
    ## Validate Array
    ######################
    while (count != 190):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        if (img is None):
            count += 1
            continue
        # print(img)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(1)
        validArr.append(hist)
        count += 1
    ## Test Array
    ######################
    while (count != 253):
        image_name = str_shine + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        if (img is None):
            image_name = str_shine + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(1)
        testArr.append(hist)
        count += 1

    ####
    #### Sunrise
    ####
    ############################################################################

    ## Train Array
    ####################
    count = 1
    while (count != 179):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(2)
        trainArr.append(hist)
        count += 1

    ## Validate Array
    ######################
    while (count != 269):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(2)
        validArr.append(hist)
        count += 1

    ## Test Array
    ######################
    while (count != 357):
        image_name = str_sunrise + str(count)
        image_name = "Dataset/" + image_name + ".jpg"
        # print(image_name)
        img = cv2.imread(image_name)
        # print(type(img))
        if (img is None):
            image_name = str_sunrise + str(count)
            image_name = "Dataset/" + image_name + ".jpeg"
            img = cv2.imread(image_name)
        # print(img)
        img_list = fold_image_sixteen(img)
        hist = list()
        for image in img_list:
            hist.extend(cv2.calcHist(image, [0, 1, 2],
                                     None, [10, 10, 10], [0, 256, 0, 256, 0, 256]).flatten().tolist())
        hist.append(2)
        testArr.append(hist)
        count += 1

    ################################################################################


# Main contains a menu
# You can choose which operation you want to run
# At the end of each operation control the results by compering predictions class and actual class
# Calculates the percentage of the true predictions and print them
def main():
    print("Welcome to the PIM-Tech Image Classifier!")
    print("Choose Your Data Source: ")
    print("\t1) Grayscale Intensity Histogram")
    print("\t2) Color Histogram")
    selection = int(input("Your selection: "))

    trainArr = list()  # List for Trainig Set
    validArr = list()  # List for Validation Set
    testArr = list()  # List for Test Set
    count = 0
    if (selection == 1):
        numLevels = int(input("Enter your spatial level (1,2 or 3): "))
        if (numLevels == 1):
            read_gray_scale_data_level1(trainArr, validArr, testArr)
            num_neighbors = 9
            result_set = evaluate_algorithm(trainArr, validArr, testArr, k_nearest_neighbors, num_neighbors)
            count = 0
            for item in result_set:
                if (item[0] == item[1]):
                    count += 1
        elif (numLevels == 2):
            read_gray_scale_data_level2(trainArr, validArr, testArr)
            num_neighbors = 9
            result_set = evaluate_algorithm(trainArr, validArr, testArr, k_nearest_neighbors, num_neighbors)
            count = 0
            for item in result_set:
                if (item[0] == item[1]):
                    count += 1
        else:
            read_gray_scale_data_level3(trainArr, validArr, testArr)
            num_neighbors = 9
            result_set = evaluate_algorithm(trainArr, validArr, testArr, k_nearest_neighbors, num_neighbors)
            count = 0
            for item in result_set:
                if (item[0] == item[1]):
                    count += 1
        print("Correctnes = %{:.2f}".format((count / len(result_set)) * 100))
    elif (selection == 2):
        numLevels = int(input("Enter your spatial level (1,2 or 3): "))
        if (numLevels == 1):
            read_color_scale_data_level1(trainArr, validArr, testArr)
            num_neighbors = 5
            result_set = evaluate_algorithm(trainArr, validArr, testArr, k_nearest_neighbors, num_neighbors)
            count = 0
            for item in result_set:
                if (item[0] == item[1]):
                    count += 1
        elif (numLevels == 2):
            read_color_scale_data_level2(trainArr, validArr, testArr)
            num_neighbors = 9
            result_set = evaluate_algorithm(trainArr, validArr, testArr, k_nearest_neighbors, num_neighbors)
            count = 0
            for item in result_set:
                if (item[0] == item[1]):
                    count += 1
        else:
            read_color_scale_data_level3(trainArr, validArr, testArr)
            num_neighbors = 9
            result_set = evaluate_algorithm(trainArr, validArr, testArr, k_nearest_neighbors, num_neighbors)
            count = 0
            for item in result_set:
                if (item[0] == item[1]):
                    count += 1

        print("Correctnes = %{:.2f}".format((count / len(result_set)) * 100))


if __name__ == "__main__":
    main()
