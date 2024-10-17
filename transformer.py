import tensorflow as tf
from fast_attention import FastSelfAttention
from self_attention import FullSelfAttention
from normalization import LayerScaling, LayerCentering
from tensorflow.keras.layers.experimental import SyncBatchNormalization

import math


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class PositionalFeature(tf.keras.layers.Layer):
    def __init__(self, d_feature, beta_hat_2, **kwargs):
        super().__init__(**kwargs)

        self.slopes = tf.range(d_feature, 0, -4.0, dtype=tf.float32) / d_feature
        self.slopes = self.slopes * beta_hat_2

    def call(self, slen, bsz=None):
        pos_seq = tf.range(0, slen, 1.0, dtype=tf.float32)/float(slen-1)
        forward = tf.einsum("i,j->ij", pos_seq, self.slopes)
        backward = tf.reverse(forward, axis=[0])
        neg_forward = -tf.identity(forward)
        neg_backward = -tf.identity(backward)
        pos_feature = tf.concat([forward, backward, neg_forward, neg_backward], -1)

        pos_feature_slopes = tf.concat(
                              [tf.identity(self.slopes),
                               -tf.identity(self.slopes),
                               -tf.identity(self.slopes),
                               tf.identity(self.slopes)], axis=0)
        pos_feature_slopes = tf.reshape(pos_feature_slopes, [1, -1])

        if bsz is not None:
            pos_feature = tf.tile(pos_feature[None, :, :], [bsz, 1, 1])
            pos_feature_slopes = tf.tile(pos_feature_slopes[None, :, :], [bsz, 1, 1])
        else:
            pos_feature = pos_feature[None, :, :]
            pos_feature_slopes = pos_feature_slopes[None, :, :]
        return pos_feature, pos_feature_slopes


class PositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.layer_1 = tf.keras.layers.Dense(
            d_inner, activation=tf.nn.relu, name='layer_1'
        )
        self.drop_1 = tf.keras.layers.Dropout(dropout, name='drop_1')
        self.layer_2 = tf.keras.layers.Dense(d_model, name='layer_2')
        self.drop_2 = tf.keras.layers.Dropout(dropout, name='drop_2')


    def call(self, inp, training=False):
        core_out = inp
        core_out = self.layer_1(core_out)
        core_out = self.drop_1(core_out, training=training)
        core_out = self.layer_2(core_out)
        core_out = self.drop_2(core_out, training=training)

        output = [core_out]
        return output


class FirstLevelTransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_head,
        d_model,
        d_inner,
        dropout,
        feature_map_type,
        normalize_attn,
        d_kernel_map,
        model_normalization,
        head_init_range,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.feature_map_type = feature_map_type
        self.normalize_attn = normalize_attn
        self.d_kernel_map = d_kernel_map
        self.model_normalization = model_normalization
        self.head_init_range = head_init_range

        self.self_attn = FastSelfAttention(
            d_model=self.d_model,
            d_head=self.d_head,
            n_head=self.n_head,
            attention_dropout=self.dropout,
            feature_map_type=self.feature_map_type,
            normalize_attn=self.normalize_attn,
            d_kernel_map=self.d_kernel_map,
            head_init_range = self.head_init_range,
            name="tran_attn",
        )
        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
            name="pos_ff",
        )

        assert self.model_normalization in ['preLC', 'postLC', 'none'], "model_normalization must be one of 'preLC', 'postLC' or 'none'"
        if self.model_normalization in ['preLC', 'postLC']:
            self.lc1 = LayerCentering()
            self.lc2 = LayerCentering()


    def call(self, inputs, training=False):
        inp, pos_ft, pos_ft_slopes = inputs
        if self.model_normalization == 'preLC':
            attn_in = self.lc1(inp)
        else:
            attn_in = inp
        attn_outputs = self.self_attn(attn_in, pos_ft, pos_ft_slopes,
                                      training=training)
        attn_outputs[0] = attn_outputs[0] + inp
        if self.model_normalization == 'postLC':
            attn_outputs[0] = self.lc1(attn_outputs[0])

        if self.model_normalization == 'preLC':
            ff_in = self.lc2(attn_outputs[0])
        else:
            ff_in = attn_outputs[0]
        ff_output = self.pos_ff(ff_in, training=training)
        ff_output[0] = ff_output[0] + attn_outputs[0]
        if self.model_normalization == 'postLC':
            ff_output[0] = self.lc2(ff_output[0])

        outputs = [ff_output[0]] + attn_outputs[1:]

        return outputs


class SecondLevelTransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_head,
        d_model,
        d_inner,
        dropout,
        model_normalization,
        gamma,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.model_normalization = model_normalization
        self.gamma = gamma

        self.self_attn = FullSelfAttention(
            n_head=self.n_head,
            d_model=self.d_model,
            d_head=self.d_head,
            dropout=self.dropout,
            gamma=self.gamma,
            name="full_self_attn",
        )
        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
            name="pos_ff",
        )

        assert self.model_normalization in ['preLC', 'postLC', 'none'], "model_normalization must be one of 'preLC', 'postLC' or 'none'"
        if self.model_normalization in ['preLC', 'postLC']:
            self.lc1 = LayerCentering()
            self.lc2 = LayerCentering()


    def call(self, inputs, softmax_attn_smoothing, training=False):
        inp, pos_emb = inputs[:2]
        if self.model_normalization == 'preLC':
            attn_in = self.lc1(inp)
        else:
            attn_in = inp
        attn_outputs = self.self_attn([attn_in, pos_emb], softmax_attn_smoothing, training=training)
        attn_outputs[0] = attn_outputs[0] + inp
        if self.model_normalization == 'postLC':
            attn_outputs[0] = self.lc1(attn_outputs[0])

        if self.model_normalization == 'preLC':
            ff_in = self.lc2(attn_outputs[0])
        else:
            ff_in = attn_outputs[0]
        ff_output = self.pos_ff(ff_in, training=training)
        ff_output[0] = ff_output[0] + attn_outputs[0]
        if self.model_normalization == 'postLC':
            ff_output[0] = self.lc2(ff_output[0])

        outputs = [ff_output[0]] + attn_outputs[1:]

        return outputs


class SoftmaxAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_head, dropout, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout

        self.q_heads = self.add_weight(
            shape=(self.d_head, self.n_head), name="q_heads"
        )
        self.k_net = tf.keras.layers.Dense(
            self.d_head * self.n_head, use_bias=False, name="k_net"
        )
        self.v_net = tf.keras.layers.Dense(
            self.d_head * self.n_head, use_bias=False, name="v_net"
        )
        self.o_net = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="v_net"
        )

        self.scale = 1. / (self.d_head ** 0.5)

        self.drop_q = tf.keras.layers.Dropout(self.dropout)
        self.drop_k = tf.keras.layers.Dropout(self.dropout)
        self.drop_v = tf.keras.layers.Dropout(self.dropout)
        self.drop_o = tf.keras.layers.Dropout(self.dropout)


    def build(self, input_shape):
        self.softmax_attn_smoothing = self.add_weight(
            "softmax_attn_smoothing",
            shape=(),
            initializer=tf.keras.initializers.Constant(0),
            dtype=tf.float32,
            trainable=False)


    def call(self, inp, softmax_attn_smoothing, training=False):
        bsz, slen = inp.shape[:2]
        if training:
            self.softmax_attn_smoothing.assign(softmax_attn_smoothing)

        k_head = self.k_net(inp)
        v_head = self.v_net(inp)

        q_head = self.drop_q(self.q_heads)
        k_head = self.drop_k(k_head, training=training)
        v_head = self.drop_v(v_head, training=training)

        k_head = tf.reshape(k_head, [-1, slen, self.d_head, self.n_head])
        v_head = tf.reshape(v_head, [-1, slen, self.d_head, self.n_head])

        attn_score = tf.einsum("bndh,dh->bnh", k_head, q_head)
        attn_score = attn_score * self.scale * self.softmax_attn_smoothing

        attn_prob = tf.nn.softmax(attn_score, axis=1)

        attn_out = tf.einsum("bndh,bnh->bnhd", v_head, attn_prob)
        attn_out = tf.reshape(attn_out, [-1, slen, self.n_head*self.d_head])

        attn_out = self.o_net(attn_out)
        attn_out = self.drop_o(attn_out, training=training)

        return attn_out, attn_score


class EstraNet(tf.keras.Model):
    def __init__(self, n_layer, d_model, d_head, n_head, d_inner, dropout, 
                 n_classes, conv_kernel_size, n_conv_layer, pool_size, d_kernel_map, 
                 beta_hat_2, model_normalization, head_initialization='forward', 
                 output_attn=False):

        super(EstraNet, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.d_inner = d_inner
        self.feature_map_type = 'fourier'
        self.normalize_attn = False
        self.d_kernel_map = d_kernel_map
        self.beta_hat_2 = beta_hat_2
        self.model_normalization = model_normalization
        self.head_initialization = head_initialization

        self.dropout = dropout 

        self.n_classes = n_classes

        self.conv_kernel_size = conv_kernel_size
        self.n_conv_layer = n_conv_layer
        self.pool_size = pool_size

        self.output_attn = output_attn

        conv_filters = [min(8*2**i, self.d_model) for i in range(self.n_conv_layer-1)] + [self.d_model]

        self.conv_layers = []
        self.norm_layers = []
        self.relu_layers = []
        self.pool_layers = []

        for l in range(self.n_conv_layer):
            ks = 11 if l == 0 else self.conv_kernel_size
            self.conv_layers.append(tf.keras.layers.Conv1D(conv_filters[l], ks, padding='same'))
            self.relu_layers.append(tf.keras.layers.ReLU())
            self.pool_layers.append(tf.keras.layers.AveragePooling1D(self.pool_size, self.pool_size, padding='same'))

        self.pos_feature = PositionalFeature(self.d_model, self.beta_hat_2)

        head_init_ranges = []
        if self.head_initialization == 'forward':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((0., 0.5))
                else:
                    head_init_ranges.append((0., 1.0))
        elif self.head_initialization == 'backward':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((-0.5, 0.0))
                else:
                    head_init_ranges.append((-1.0, 0.0))
        elif self.head_initialization == 'symmetric':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((-0.5, 0.5))
                else:
                    head_init_ranges.append((-1.0, 1.0))
        else:
            assert False, "head_initialization can be one of ['forward', 'backward', 'symmetric']"

        self.fast_tran_layers = []
        for i in range(self.n_layer):
            self.fast_tran_layers.append(
                FirstLevelTransformerLayer(
                    n_head=self.n_head,
                    d_head=self.d_head,
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    feature_map_type=self.feature_map_type,
                    normalize_attn=self.normalize_attn,
                    d_kernel_map=self.d_kernel_map,
                    model_normalization=self.model_normalization,
                    head_init_range = head_init_ranges[i],
                    name='layers_._{}'.format(i)
                )
            )

        self.out_dropout = tf.keras.layers.Dropout(dropout, name='out_drop')


    def call(self, inp, training=False):
        # apply the convolution blocks
        for l in range(self.n_conv_layer):
            inp = self.conv_layers[l](inp)
            inp = self.relu_layers[l](inp)
            inp = self.pool_layers[l](inp)

        bsz, slen = shape_list(inp)[:2]

        pos_ft, pos_ft_slopes = self.pos_feature(slen, bsz)

        core_out = inp
        out_list = []
        for i, layer in enumerate(self.fast_tran_layers):
            all_out = layer([core_out, pos_ft, pos_ft_slopes], training=training)
            core_out = all_out[0]
            out_list.append(all_out[1:])
        core_out = self.out_dropout(core_out, training=training)

        for i in range(len(out_list)-1):
            for j in range(len(out_list[i])):
                out_list[i][j] = tf.transpose(out_list[i][j], [1, 0, 2, 3])

        if self.output_attn:
            return [core_out, out_list]
        else:
            return [core_out]
 

class HierTransformer(tf.keras.Model):
    def __init__(self, n_layer, d_model, d_head, n_head, d_inner, 
                 d_head_softmax, n_head_softmax, dropout, n_classes, 
                 conv_kernel_size, n_conv_layer, pool_size, d_kernel_map, beta_hat_2, 
                 gamma, model_normalization, head_initialization='forward', seg_len=5000, 
                 seg_stride=3000, input_len=10000, softmax_attn=True, output_attn=False):

        super(HierTransformer, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_head_softmax = d_head_softmax
        self.n_head_softmax = n_head_softmax
        self.feature_map_type = 'fourier'
        self.normalize_attn = False
        self.d_kernel_map = d_kernel_map
        self.beta_hat_2 = beta_hat_2
        self.gamma = gamma
        self.model_normalization = model_normalization
        self.head_initialization = head_initialization
        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.input_len = input_len
        self.softmax_attn = softmax_attn

        self.dropout = dropout 

        self.n_classes = n_classes

        self.conv_kernel_size = conv_kernel_size
        self.n_conv_layer = n_conv_layer
        self.pool_size = pool_size

        self.output_attn = output_attn

        self.inner_trans = EstraNet(n_layer=self.n_layer, 
                                    d_model=self.d_model, 
                                    d_head=self.d_head, 
                                    n_head=self.n_head, 
                                    d_inner=self.d_inner, 
                                    dropout=self.dropout, 
                                    n_classes=self.n_classes, 
                                    conv_kernel_size=self.conv_kernel_size, 
                                    n_conv_layer=self.n_conv_layer, 
                                    pool_size=self.pool_size, 
                                    d_kernel_map=self.d_kernel_map, 
                                    beta_hat_2=self.beta_hat_2, 
                                    model_normalization=self.model_normalization, 
                                    head_initialization=self.head_initialization, 
                                    output_attn=self.output_attn)

        if self.softmax_attn:
            self.n_segment = int(math.ceil(self.input_len/self.seg_stride))
            self.sm_attns = []
            for i in range(self.n_segment):
                self.sm_attns.append(
                    SoftmaxAttention(d_model=self.d_model, n_head=self.n_head_softmax, 
                                     d_head=self.d_head_softmax, dropout=self.dropout)
                )

        self.abs_pos_embedding = tf.keras.layers.Embedding(self.n_segment, self.d_model)

        self.full_trans_layer = SecondLevelTransformerLayer(
                    n_head=self.n_head,
                    d_head=self.d_head,
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    model_normalization=self.model_normalization,
                    gamma=self.gamma,
                    name='full_trans_layer'
                )

        self.fc_output = tf.keras.layers.Dense(self.n_classes)


    def call(self, inp, softmax_attn_smoothing=1, training=False):
        # convert the input dimension from [bsz, len] to [bsz, len, 1]
        if len(inp.shape.as_list()) == 2:
            inp = tf.expand_dims(inp, axis=-1)
        bsz, slen = shape_list(inp)[:2]

        out_list = []
        sm_attn_score_list = []
        core_out_list = []
        i = 0
        for s_start in range(0, slen - self.seg_len + self.seg_stride, self.seg_stride):
            s_end = min(s_start + self.seg_len, slen)
            all_out = self.inner_trans(inp[:, s_start: s_end, :], training=training)
            core_out = all_out[0]
            out_list.append(all_out[1:])
            if self.softmax_attn:
                core_out, sm_attn_score = self.sm_attns[i](core_out, softmax_attn_smoothing, training=training)
                sm_attn_score_list.append(sm_attn_score)
            core_out = tf.reduce_mean(core_out, axis=1)
            core_out_list.append(core_out)
            i += 1
        core_out = tf.stack(core_out_list, axis=1)

        # Add absolute positional encoding to the input
        nseg = shape_list(core_out)[1]
        pos_seq = tf.range(0, nseg, 1)
        pos_emb = self.abs_pos_embedding(pos_seq)

        # Put the last transformer layer
        all_out = self.full_trans_layer([core_out, pos_emb], softmax_attn_smoothing, training=training)
        output = all_out[0][:, 0, :]
        out_list.append(all_out[1:])

        # Get the final scores for all classes
        scores = self.fc_output(output)

        if self.output_attn:
            return [scores, out_list, sm_attn_score_list]
        else:
            return [scores]
        

