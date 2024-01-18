import tensorflow as tf


class FullSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="q_net"
        )
        self.k_net = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="k_net"
        )
        self.v_net = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="v_net"
        )
        self.o_net = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="o_net"
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


    def call(self, inputs, softmax_attn_smoothing, training=False):
        inp, pos_emb = inputs[:2]

        bsz, slen = inp.shape[:2]
        if training:
            self.softmax_attn_smoothing.assign(softmax_attn_smoothing)

        q_head = self.q_net(pos_emb)
        k_head = self.k_net(pos_emb)
        v_head = self.v_net(inp)

        q_head = self.drop_q(q_head, training=training)
        k_head = self.drop_k(k_head, training=training)
        v_head = self.drop_v(v_head, training=training)

        q_head = tf.reshape(q_head, [slen, self.d_head, self.n_head])
        k_head = tf.reshape(k_head, [slen, self.d_head, self.n_head])
        v_head = tf.reshape(v_head, [-1, slen, self.d_head, self.n_head])

        attn_score = tf.einsum("idh,jdh->ijh", q_head, k_head)
        attn_score = attn_score * self.scale * self.softmax_attn_smoothing

        attn_prob = tf.nn.softmax(attn_score, axis=1)

        attn_out = tf.einsum("ijh,bjdh->bihd", attn_prob, v_head)
        attn_out = tf.reshape(attn_out, [bsz, slen, -1])

        attn_out = self.o_net(attn_out)
        attn_out = self.drop_o(attn_out, training=training)

        return [attn_out, attn_score]


