import numpy as np
import os

def create_data(sample_path, label_path, num_samples):
    samples = []
    labels = []
    random_sizes = np.random.randint(low=1, high=20, size=[num_samples])
    for size in random_sizes:
        current_sample = np.random.rand(size)  # create sample
        current_label = [x + 2 for x in current_sample]  # create label for current sample
        samples.append(current_sample)
        labels.append(current_label)
    np.save(sample_path, samples)
    np.save(label_path, labels)
    return samples, labels


def load_data(sample_path, label_path):
    samples = np.load(sample_path)
    labels = np.load(label_path)
    return samples, labels


def main():
    samples_path = 'samples.npy'
    labels_path = 'labels.npy'
    num_samples = 10000
    try:
        return load_data(samples_path, labels_path)
    except:
        return create_data(samples_path, labels_path, num_samples)
