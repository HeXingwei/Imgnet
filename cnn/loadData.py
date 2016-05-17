import cv2
import numpy as np

def loadData():
    testFile="../food-10/meta/test.txt"
    trainFile="../food-10/meta/train.txt"
    classFile="../food-10/meta/classes.txt"
    with open(testFile) as f:
        testImgFiles=f.readlines()
    with open(trainFile) as f:
        trainImgFiles=f.readlines()
    classDict={}
    i=0
    with open(classFile) as f:
        for food in f.readlines():
            food=food.split("\n")[0]
            classDict[food]=i
            i+=1
    trainDataLength=len(trainImgFiles)
    testDataLength=len(testImgFiles)
    print "train data length is %d"%trainDataLength
    print "test data length is %d"%testDataLength
    X_train=np.zeros((trainDataLength,3,227,227))
    Y_train=np.zeros((trainDataLength),dtype="int32")
    X_test=np.zeros((testDataLength,3,227,227))
    Y_test=np.zeros((testDataLength),dtype="int32")
    #load the trainData from image files
    i=0
    for file in trainImgFiles:
        file=file.split("\n")[0]
        food=file.split("/")[0]
        Y_train[i]=classDict[food]
        file="../food-10/images/"+file+".jpg"
        #read the img (BGR),and scaling the image to 227*227
        img = cv2.imread(file,1)
        img=cv2.resize(img,(227, 227), interpolation = cv2.INTER_CUBIC)
        #translate BGR into RGB
        X_train[i,0]=img[:,:,2]
        X_train[i,1]=img[:,:,1]
        X_train[i,2]=img[:,:,0]
        i+=1
    #load the testData from image files
    i=0
    for file in testImgFiles:
        file=file.split("\n")[0]
        file="../food-10/images/"+file+".jpg"
        #read the img (BGR),and scaling the image to 227*227
        img = cv2.imread(file,1)
        img=cv2.resize(img,(227, 227), interpolation = cv2.INTER_CUBIC)
        #translate BGR into RGB
        X_test[i,0]=img[:,:,2]
        X_test[i,1]=img[:,:,1]
        X_test[i,2]=img[:,:,0]
        i+=1
        #cv2.imshow('image',img)
        #print img.shape
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        return X_train,Y_train,X_test,Y_test
