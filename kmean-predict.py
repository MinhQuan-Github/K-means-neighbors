import numpy as np
from scipy.spatial.distance import cdist


class KMeanPredict:
    def __init__(self):
        # init data
        self.centers = np.load('models/kmean_thetrang_center.npy')
        self.min_max_scaler = np.load('models/scaler_info.npy')
        self.labels_meaning = ['Underweight', 'overweight', 'standard']

    def input(self):
        height_input = int(input("Enter height of person: "))
        weight_input = int(input("Enter weight of person: "))
        return height_input, weight_input

    def predict(self, height=175, weight=70):
        height_norm = (height - self.min_max_scaler[0][0]) / (self.min_max_scaler[0][1] - self.min_max_scaler[0][0])
        weight_norm = (weight - self.min_max_scaler[1][0]) / (self.min_max_scaler[1][1] - self.min_max_scaler[1][0])

        D = cdist(np.array([[height_norm, weight_norm]]), self.centers)
        print(D)
        label = np.argmin(D, axis=1)[0]
        return self.labels_meaning[label]


if __name__ == '__main__':
    k_mean_predict = KMeanPredict()
    height, weight = k_mean_predict.input()
    label_mean = k_mean_predict.predict(height=height, weight=weight)
    print("=> Result: ", label_mean)
