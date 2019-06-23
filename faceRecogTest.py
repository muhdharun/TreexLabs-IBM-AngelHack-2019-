#!/usr/bin/env python3
import numpy as np
import requests

## This code was taken from the IBM/MAX-Facial-Recognizer readme page
## with a few minor edits due to trouble with getting some of the libraries
## such as matplotlib and scipy.
## Below are also some test cases

url = 'http://192.168.99.100:5000/model/predict'

def facenet(input_img):
    """
    Takes in input image file path and generates face embeddings/bboxes.
    """
    files = {'image': ('image.jpg',open(input_img,'rb'), 'images/jpeg')}
    r = requests.post(url, files=files).json()
    
    return r

def same_person(img1, img2, threshold=0.95):
    """
    Determines if two images belong to the same face/person.
    
    This function is a simple example to show how to use the 
    model to detect faces in images and generate the embedding 
    vectors to identify people. The hard-coded `threshold` 
    may not work for all images. With more label data, users
    could train a classifier (e.g., SVM) to perform the 
    classification with higer accuracy.
    """
    emb1 = facenet(img1)['predictions'][0]['embedding']
    emb2 = facenet(img2)['predictions'][0]['embedding']
    
    return np.linalg.norm(np.asarray(emb1)-np.asarray(emb2)) <= threshold

##Sanity check that it works
preds = facenet('assets/Lenna.jpg')
print(preds['status'])


paul1 = 'assets/Paul.jpeg'
paul2 = 'assets/Paul2.jpeg'
chris = 'assets/Chris.jpeg'
bob1 = 'assets/bob1.jpg'
bob2 = 'assets/bob2.png'
o1 = 'assets/ObamaID.jpg'
o2 = 'assets/ObamaActual.jpg'
tom1 = 'assets/TomCruiseID.jpg'
tom2 = 'assets/TomCruiseActual.jpg'
d1 = 'assets/DonaldTrumpID.jpg'
d2 = 'assets/DonaldTrumpActual.jpg'

#Positive test cases:
print("===Positive test cases start===")
people1 = [(paul1,paul2),(d1,d2),(o1,o2),
           (tom1,tom2)]

##print(same_person(paul1,paul2))
##print(same_person(d1,d2))
##print(same_person(o1,o2))
##print(same_person(tom1,tom2))

pass1 = 0
fail1 = 0

for i in people1:
    result = same_person(i[0],i[1])
    print(result)
    if (result):
        pass1 += 1
    else:
        fail1 += 1
print("Test cases passed: ", str(pass1) + "/" + str(fail1+pass1))
print("===Positive test cases end===")

print("===Negative test cases start===")
people2 = [(paul1,tom1),(d1,tom1),(o1,bob2),
           (tom1,d2)]

pass2 = 0
fail2 = 0

for i in people2:
    result = same_person(i[0],i[1])
    print(result)
    if (result):
        pass2 += 1
    else:
        fail2 += 1      

print("Test cases passed: ", str(fail2) + "/" + str(fail2+pass2))
print("===Negative test cases end===")
