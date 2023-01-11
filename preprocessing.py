import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self):
        self.data_path = "datasets/the_trang_kmeans.csv"
        self.data = pd.read_csv(self.data_path).values
        self.data = np.array(self.data, dtype=np.float32)

    def draw_data(self, is_show=False, labels=None):
        plt.title("The trang K-means")
        plt.xlabel("Height")
        plt.ylabel("Weight")
        if labels is None:
            plt.scatter(self.data[:, 0], self.data[:, 1])
        else:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=labels)
        if is_show:
            plt.show()

    def draw_points(self, points, is_show=False, cs='rx'):
        plt.plot(points[:, 0], points[:, 1], cs)
        if is_show:
            plt.show()

    def normalization(self):
        scaler = [1.0, 1.0]
        scaler_save = []
        for i in range(self.data.shape[1]):
            min_cols = np.min(self.data[:, i])
            max_cols = np.max(self.data[:, i])
            self.data[:, i] = scaler[i] * (self.data[:, i] - min_cols) / (max_cols - min_cols)
            scaler_save.append([min_cols, max_cols])
        scaler_save = np.array(scaler_save)
        np.save('models/scaler_info.npy', scaler_save)
        print("normalize succeeded")

    def get_data_train(self):
        self.normalization()
        return self.data


if __name__ == '__main__':
    prep_obj = Preprocessing()
    prep_obj.draw_data(is_show=True)
    print(prep_obj.get_data_train())
