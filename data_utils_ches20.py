import numpy as np
import tensorflow as tf
import h5py

import os, sys


def sbox_layer(x):
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return np.stack([y0, y1, y2, y3], axis=1)


class Dataset:
    def __init__(self, data_path, split, input_length, data_desync=0, start_idx=0, desync=0, seed=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync
        self.start_idx = start_idx
        self.desync = desync
        self.seed = seed

        data = np.load(data_path)
        self.traces = data['traces']
        self.nonces = data['nonces']
        self.umsk_keys = data['umsk_keys']

        shift = 17
        self.nonces = self.nonces >> shift
        self.umsk_keys = self.umsk_keys >> shift
        if len(self.umsk_keys.shape) == 1:
            self.umsk_keys = np.reshape(self.umsk_keys, [1, -1])
        
        sbox_in = np.bitwise_xor(self.nonces, self.umsk_keys)
        sbox_in = sbox_in.T
        sbox_out = sbox_layer(sbox_in)
        self.labels = (sbox_out & 0x1)
        self.labels = self.labels.astype(np.float32)

        window_length = self.input_length + self.desync + self.data_desync
        assert self.start_idx + window_length <= self.traces.shape[1]
        if self.start_idx > 0:
            self.traces = self.traces[:, self.start_idx: self.start_idx + window_length]
        elif self.traces.shape[0] < 2*window_length:
            self.traces = self.traces[:, self.start_idx: self.start_idx + window_length]
        self.trace_length = self.traces.shape[1]

        #np.random.seed(self.seed)
        rng = np.random.RandomState(self.seed)
        self.desyncs = rng.randint(low=0, high=self.desync+1, size=(self.traces.shape[0]))
        self.ApplyDesync(self.traces, self.desyncs, self.input_length+self.data_desync)

        self.num_all_samples = self.traces.shape[0]


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
                              .prefetch(2)


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

