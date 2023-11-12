import pandas as pd
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size):
        self.x, self.y = x_data, y_data
        self.batch_size = batch_size
        self.num_batches = np.ceil(len(x_data) / batch_size)
        self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

    def __len__(self):
        return len(self.batch_idx)//4

    def __getitem__(self, idx):
        # This line is shit:
        actual_id = np.random.randint(0, len(self.batch_idx))

        batch_x = self.x[self.batch_idx[actual_id]]
        batch_y = self.y[self.batch_idx[actual_id]]
        return batch_x, batch_y


def get_data_generator_and_tests(l, valid_idx):
    # Create a filename based on 'l'
    filename = './HNU/pp_data/data-1d-{}-pp.csv'.format(l)

    # Read data from the CSV file using pandas
    df = pd.read_csv(filename, sep=';')
    final_array = []

    # Extract the data from the DataFrame and convert it to a NumPy array
    for row in df.iloc[:, 0]:
        data = row.split(',')
        numpy_array = np.array(data, dtype=float)
        final_array.append(numpy_array)
    X = np.array(final_array)

    Y = np.array(df.iloc[:, 2])
    from sklearn.model_selection import train_test_split

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=42)

    print("Train data: ", train_x.shape, train_y.shape)
    print("Test data: ", test_x.shape, test_y.shape)

    train_generator = DataGenerator(train_x, train_y, batch_size = 64)

    return train_generator, test_x, test_y