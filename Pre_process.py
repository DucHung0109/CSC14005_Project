import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def load_cifar10_data(data_path):
    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch = unpickle(f"{data_path}/data_batch_{i}")
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)

    test_batch = unpickle(f"{data_path}/test_batch")
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_batch[b'labels'])

    return train_data, train_labels, test_data, test_labels
data_path = "cifar-10-batches-py"
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_path)
# Each row is one sample
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]

train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Each row is one sample
train_data_final = train_data.reshape(train_data.shape[0], -1)
test_data_final = test_data.reshape(test_data.shape[0], -1)