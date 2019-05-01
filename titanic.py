import numpy as np
from sklearn.naive_bayes import GaussianNB
import xlrd
import xlwt
import csv

#load the train sheet values into a normal array
trainbook = xlrd.open_workbook('train.xlsx')
trainsheet = trainbook.sheet_by_name('train')
train_normal = []
for i in range (892):
    train_normal.append([])
    for j in range (12):
        train_normal[i].append(trainsheet.cell(i, j).value)

#put the values in numpy array
train_numpy = np.array(train_normal)


#preprocessing of training data
for i in range (len(train_numpy)):


    if train_numpy[i,5] == '':
        train_numpy[i,5] = 0.0
    elif len(train_numpy[i,5]) == 1:
        train_numpy[i,5] = 1.0
    elif train_numpy[i,5][0] == '1':
        train_numpy[i,5] = 2.0
    elif train_numpy[i,5][0] == '2':
        train_numpy[i,5] = 3.0
    elif train_numpy[i,5][0] == '3':
        train_numpy[i,5] = 4.0
    elif train_numpy[i,5][0] == '4':
        train_numpy[i,5] = 5.0
    elif train_numpy[i,5][0] == '5':
        train_numpy[i,5] = 6.0
    elif train_numpy[i,5][0] == '6':
        train_numpy[i,5] = 7.0
    else:
        train_numpy[i,5] = 8.0


    if train_numpy[i,4] == 'male':
        train_numpy[i,4] = 0.0
    else:
        train_numpy[i,4] = 1.0


    if train_numpy[i,-3].find('.') < 2:
        train_numpy[i,-3] = 0.0
    elif train_numpy[i,-3].find('.') < 3:
        train_numpy[i,-3] = 1.0
    elif train_numpy[i,-3].find('.') < 4:
        train_numpy[i,-3] = 2.0
    else:
        train_numpy[i,-3] = 3.0





    if train_numpy[i,-2] == '':
        train_numpy[i,-2] = '7.0'

    elif train_numpy[i,-2][0] == 'A':
        train_numpy[i,-2] = '0.0'
    elif train_numpy[i,-2][0] == 'B':
        train_numpy[i,-2] = '1.0'
    elif train_numpy[i,-2][0] == 'C':
        train_numpy[i,-2] = '2.0'
    elif train_numpy[i,-2][0] == 'D':
        train_numpy[i,-2] = '3.0'
    elif train_numpy[i,-2][0] == 'E':
        train_numpy[i,-2] = '4.0'
    elif train_numpy[i,-2][0] == 'F':
        train_numpy[i,-2] = '5.0'
    elif train_numpy[i,-2][0] == 'G':
        train_numpy[i,-2] = '6.0'
    else:
        train_numpy[i,-2] = '8.0'


    if train_numpy[i, -1] == 'C':
        train_numpy[i, -1] = '0.0'
    elif train_numpy[i, -1] == 'S':
        train_numpy[i, -1] = '1.0'
    else:
        train_numpy[i, -1] = '2.0'



#take the coloumns needed only
x_set = np.array(train_numpy[1:,[2,4,5,6,7,9,10,11]]).astype(np.float)
y_set = np.array(train_numpy[1:,1])

#fit the data into NB
clf = GaussianNB()
clf.fit(x_set,y_set)



#load the test sheet values into a normal array


testbook = xlrd.open_workbook('test.xlsx')
testsheet = testbook.sheet_by_name('test')


test_normal = []
for i in range (419):
    test_normal.append([])
    for j in range (11):
        test_normal[i].append(testsheet.cell(i, j).value)

#put the values in numpy array


test_numpy = np.array(test_normal)

#preprocessing of test data

for i in range (len(test_numpy)):


    if test_numpy[i,4] == '':
        test_numpy[i,4] = 0.0
    elif len(test_numpy[i,4]) == 1:
        test_numpy[i,4] = 1.0
    elif test_numpy[i,4][0] == '1':
        test_numpy[i,4] = 2.0
    elif test_numpy[i,4][0] == '2':
        test_numpy[i,4] = 3.0
    elif test_numpy[i,4][0] == '3':
        test_numpy[i,4] = 4.0
    elif test_numpy[i,4][0] == '4':
        test_numpy[i,4] = 5.0
    elif test_numpy[i,4][0] == '5':
        test_numpy[i,4] = 6.0
    elif test_numpy[i,4][0] == '6':
        test_numpy[i,4] = 7.0
    else:
        test_numpy[i,4] = 8.0


    if test_numpy[i,-3].find('.') < 2:
        test_numpy[i,-3] = 0.0
    elif test_numpy[i,-3].find('.') < 3:
        test_numpy[i,-3] = 1.0
    elif test_numpy[i,-3].find('.') < 4:
        test_numpy[i,-3] = 2.0
    else:
        test_numpy[i,-3] = 3.0


    if test_numpy[i,3] == 'male':
        test_numpy[i,3] = 0.0
    else:
        test_numpy[i,3] = 1.0

    if test_numpy[i,-2] == '':
        test_numpy[i,-2] = '7.0'

    elif test_numpy[i,-2][0] == 'A':
        test_numpy[i,-2] = '0.0'
    elif test_numpy[i,-2][0] == 'B':
        test_numpy[i,-2] = '1.0'
    elif test_numpy[i,-2][0] == 'C':
        test_numpy[i,-2] = '2.0'
    elif test_numpy[i,-2][0] == 'D':
        test_numpy[i,-2] = '3.0'
    elif test_numpy[i,-2][0] == 'E':
        test_numpy[i,-2] = '4.0'
    elif test_numpy[i,-2][0] == 'F':
        test_numpy[i,-2] = '5.0'
    elif test_numpy[i,-2][0] == 'G':
        test_numpy[i,-2] = '6.0'
    else:
        test_numpy[i,-2] = '8.0'


    if test_numpy[i, -1] == 'C':
        test_numpy[i, -1] = '0.0'
    elif test_numpy[i, -1] == 'S':
        test_numpy[i, -1] = '1.0'
    else:
        test_numpy[i, -1] = '2.0'


#take the coloumns needed only

test_set = np.array(test_numpy[1:,[1,3,4,5,6,8,9,10]]).astype(np.float)
output = np.array(test_numpy[0:,[0,1]])

output[0][1] = 'Survived'


#predict
prediction = clf.predict(test_set)

#put the prediction in the output array

for i in range (len(output)):
    if i == 0 :
        continue
    else:
        output[i][1] = prediction[i-1]

#save the output array into an excel sheet

titanic = open('titanic.csv', 'w')
wr = csv.writer(titanic, quoting=csv.QUOTE_ALL)

for i in range (len(output)):
    if i == 0:
        wr.writerow(output[i])
    else:
        wr.writerow([int(float(output[i][0])),int(float(output[i][1]))])


titanic.close()
