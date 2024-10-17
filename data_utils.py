import numpy as np
import tensorflow as tf
import h5py

import os, sys

class Dataset:
    def __init__(self, data_path, split, input_length, data_desync=0, start_idx=0, desync=0, seed=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync
        self.start_idx = start_idx
        self.desync = desync
        self.seed = seed

        corpus = h5py.File(data_path, 'r')
        if split == 'train':
            split_key = 'Profiling_traces'
        elif split == 'test':
            split_key = 'Attack_traces'

        window_length = self.input_length + self.desync + self.data_desync
        self.traces = corpus[split_key]['traces'][:, self.start_idx: self.start_idx + window_length]
        self.labels = np.reshape(corpus[split_key]['labels'], [-1, 1])

        self.num_all_samples = self.traces.shape[0]
        self.trace_length = self.traces.shape[1]

        #np.random.seed(self.seed)
        rng = np.random.RandomState(self.seed)
        self.desyncs = rng.randint(low=0, high=self.desync+1, size=(self.num_all_samples))
        self.ApplyDesync(self.traces, self.desyncs, self.input_length+self.data_desync)

        self.plaintexts = self.GetPlaintexts(corpus[split_key]['metadata'])
        self.masks = self.GetMasks(corpus[split_key]['metadata'])
        self.keys = self.GetKeys(corpus[split_key]['metadata'])

    
    def GetPlaintexts(self, metadata):
        plaintexts = []
        for i in range(len(metadata)):
            plaintexts.append(metadata[i]['plaintext'][2])
        return np.array(plaintexts)


    def GetKeys(self, metadata):
        keys = []
        for i in range(len(metadata)):
            keys.append(metadata[i]['key'][2])
        return np.array(keys)


    def GetMasks(self, metadata):
        masks = []
        for i in range(len(metadata)):
            masks.append(np.array(metadata[i]['masks']))
        masks = np.stack(masks, axis=0)
        return masks


    def ApplyDesync(self, traces, desyncs, trace_length):
        nsamples, _ = traces.shape
        for i in range(nsamples):
            traces[i, :trace_length] = traces[i, desyncs[i]: desyncs[i]+trace_length]


    def GetMaxSupportedNumBatches(self, batch_size):
        return min(1000000000//(self.input_length*batch_size), self.num_all_samples//batch_size)


    def GetSampledDataset(self, num_batches, batch_size, training=False):
        sample_size = num_batches * batch_size
        assert sample_size*self.input_length <= 1000000000

        if not hasattr(self, 'sample_start_idx'):
            self.sample_start_idx = self.num_all_samples

        if self.sample_start_idx > self.num_all_samples - sample_size:
            if training:
                rng_state = np.random.get_state()
                np.random.shuffle(self.traces)
                np.random.set_state(rng_state)
                np.random.shuffle(self.labels)
            self.sample_start_idx = 0

        trace_samples = np.zeros(shape=(sample_size, self.input_length), dtype=self.traces.dtype)

        ds = np.random.randint(low=0, high=self.data_desync+1, size=(sample_size))
        for i in range(sample_size): 
            trace_samples[i, :] = self.traces[self.sample_start_idx + i, ds[i]: ds[i] + self.input_length]
        label_samples = self.labels[self.sample_start_idx: self.sample_start_idx + sample_size]

        self.sample_start_idx = self.sample_start_idx + sample_size

        sampled_dataset = tf.data.Dataset.from_tensor_slices((trace_samples, label_samples))
        del trace_samples, label_samples

        return sampled_dataset.batch(batch_size, drop_remainder=True) \
                              .map(lambda x, y: (tf.cast(x, tf.float32), y)) \
                              .prefetch(10)


    def GetDataset(self):
        return self.traces, self.labels

    
if __name__ == '__main__':
    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    split = sys.argv[3]

    dataset = Dataset(data_path, split, 100)

    '''
    print("traces    : "+str(dataset.traces.shape))
    print("labels    : "+str(dataset.labels.shape))
    print("plaintext : "+str(dataset.plaintexts.shape))
    print("keys      : "+str(dataset.keys.shape))
    print("traces ty : "+str(dataset.traces.dtype))
    print("")
    print("")
    '''

    print(dataset.GetMaxSupportedNumBatches(batch_size))
    print('')
    print('')

    for it in range(10):
        sampled_dataset = dataset.GetSampledDataset(10, batch_size, training=True)
        iterator = iter(sampled_dataset)
        for i in range(10):
            tr, lbl = iterator.get_next()
            #print(str(tr.shape)+' '+str(lbl.shape))
            #print(str(tr.dtype)+' '+str(lbl.dtype))
            print(str(lbl.numpy().reshape([-1])))
            #print('')
        print('')

