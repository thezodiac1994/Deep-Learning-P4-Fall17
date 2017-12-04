import pandas as pd
import scipy as sp
import cv2
import numpy as np

labelFile = pd.read_csv('attributes.csv');
outputLabel = 'Eyeglasses'
nameLabel = 'image_name'
labels = labelFile[:][outputLabel]
paths = labelFile[:][nameLabel]
dataFraction = int(.01 * len(labels))  # fraction of data to be used -> set to .01 when using on local system (1%)
trainFraction = int(0.8 * dataFraction)
validationFraction = int(0.9 * dataFraction)


def extractLabels():

    # labels for different sets
    trainingLabels = labels[0 : trainFraction]
    #trainingLabels.to_csv('trainingLabels.csv', sep=',',header='None')
    np.save('trainingLabels', trainingLabels.values)

    validationLabels = labels[trainFraction : validationFraction]
    #validationLabels.to_csv('validationLabels.csv', sep=',',header='None')
    np.save('validationLabels', validationLabels.values)

    testingLabels = labels[validationFraction : dataFraction]
    #testingLabels.to_csv('testingLabels.csv', sep=',',header='None')
    np.save('testingLabels',testingLabels.values)


def readImages():
    #relative path append to paths -> '../img/'
    pathsList = paths.tolist()
    imagePaths = ['../img/' + s for s in pathsList]
    trainPaths = imagePaths[0 : trainFraction]
    validationPaths = imagePaths[trainFraction : validationFraction]
    testingPaths = imagePaths[validationFraction : dataFraction]

    trainVectors = np.empty((0, 59 * 73), float);
    for ipath in trainPaths:
        img = cv2.imread(ipath);
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (59, 73))
        flattenedImg = img.flatten()
        trainVectors = np.vstack([trainVectors, flattenedImg]);
        print(ipath)

    np.save('trainingVectors', trainVectors)
    #np.savetxt('trainingVectors.csv', trainVectors, delimiter = ',')


    validationVectors = np.empty((0, 59 * 73), float);
    for ipath in validationPaths:
        img = cv2.imread(ipath);
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (59, 73))
        flattenedImg = img.flatten()
        validationVectors = np.vstack([validationVectors, flattenedImg]);
        print(ipath)

    np.save('validationVectors', validationVectors)
    #np.savetxt('validationVectors.csv', validationVectors, delimiter = ',')

    testingVectors = np.empty((0, 59 * 73), float);
    for ipath in testingPaths:
        img = cv2.imread(ipath);
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (59, 73))
        flattenedImg = img.flatten()
        testingVectors = np.vstack([testingVectors, flattenedImg]);
        print(ipath)

    np.save('testingVectors', testingVectors)
    #np.savetxt('testingVectors.csv', testingVectors, delimiter=',')


def main():
    extractLabels();
    readImages();

main()

