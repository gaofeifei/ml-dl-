import csv
import pandas as pd
import numpy as np
#第一行就是数据，没有题目栏 要用header=None
test_data = pd.read_csv('./data/test.csv', encoding='big5', header=None)
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
# test_raw_data = test_data.to_numpy()
test_raw_data = np.array(test_data)

test_x = np.empty([240, 18 * 9], dtype=float)
# print(test_raw_data)
for i in range(240):
    test_x[i, :] = test_raw_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
# print(test_x[0])

test_mean = np.mean(test_x, axis=0)
test_std = np.std(test_x, axis=0)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if test_std[j] != 0:
            test_x[i][j] = (test_x[i][j] - test_mean[j]) / test_std[j]
# print(test_x[0])
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
test_y = np.empty([240, 1], dtype=float)
w = np.load('weight.npy')
test_y = np.dot(test_x, w)
for i in range(240):
    if int(test_y[i][0]) > 0:
        # print('id:', i, ' PM2.5:', int(test_y[i][0]))
        print( i, ' :', int(test_y[i][0]))
    else:
        # print('id:', i, ' PM2.5:', 0)
        print(i, ' :', 0)

with open('predict.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    header = ['id', 'PM2.5']
    csv_writer.writerow(header)
    for i in range(240):
        if int(test_y[i][0]) > 0:
            row = ['id_'+str(i), str(int(test_y[i][0]))]
        else:
            row = ['id_'+str(i), '0']
        csv_writer.writerow(row)