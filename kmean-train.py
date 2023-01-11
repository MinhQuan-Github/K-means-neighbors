import preprocessing
import numpy as np
from scipy.spatial.distance import cdist
import time


class KmeanTrain:
    def __init__(self,):
        self.prep = preprocessing.Preprocessing()
        self.k = self.getCluster()

    def getCluster(self):
        return 3

    def training(self):
        data = self.prep.get_data_train()
        centers = data[np.random.choice(data.shape[0], self.k, replace=False)]
        while True:
            # filter labels depend on distance
            labels = np.argmin(cdist(data, centers), axis=1)

            self.prep.draw_data(labels=labels)
            self.prep.draw_points(points=centers, is_show=True)

            new_centers = []
            for i in range(self.k):
                center = np.mean(data[labels == i], axis=0)
                new_centers.append(center)
            new_centers = np.array(new_centers)

            if self.check_center_before_and_after_is_same(centers, new_centers):
                break

            centers = new_centers
            time.sleep(1)
        np.save('models/kmean_thetrang_center.npy', centers)
        print("Training done!!!")

    def check_center_before_and_after_is_same(self, before, alter):
        return True if set([tuple(c) for c in before]) == set([tuple(c) for c in alter]) else False


if __name__ == '__main__':
    k_mean_train = KmeanTrain()
    k_mean_train.training()
