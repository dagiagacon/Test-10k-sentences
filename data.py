import numpy as np


eos_id = 0
sos_id = 1
unk_id = 2
src_vocab_size = 7000
tgt_vocab_size = 6000


def create_data(sample_path, label_path, num_samples):
    samples = []
    labels = []
    random_sizes = np.random.randint(low=1, high=20, size=[num_samples])
    for size in random_sizes:
        current_sample = np.random.randint(unk_id, src_vocab_size, [size])  # create sample
        current_label = [x + 1 for x in current_sample]  # create label for current sample
        samples.append(current_sample)
        labels.append(current_label)
    np.save(sample_path, samples)
    np.save(label_path, labels)
    return samples, labels


def load_data(sample_path, label_path):
    samples = np.load(sample_path)
    labels = np.load(label_path)
    return samples, labels


def get_train_set():
    samples_path = 'samples.npy'
    labels_path = 'labels.npy'
    num_samples = 10000
    try:
        return load_data(samples_path, labels_path)
    except:
        return create_data(samples_path, labels_path, num_samples)


def get_test_set():
    samples_path = 'test_samples.npy'
    labels_path = 'test_labels.npy'
    num_samples = 500
    try:
        return load_data(samples_path, labels_path)
    except:
        return create_data(samples_path, labels_path, num_samples)


def get_vocabs():
    vocab_src = np.arange(src_vocab_size)
    vacab_tgt = np.arange(tgt_vocab_size)
    return vocab_src, vacab_tgt