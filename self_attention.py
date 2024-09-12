import tensorflow as tf


class FullSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        gamma,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.gamma = gamma

        self.q_net_inp = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="q_net_inp"
        )
        self.k_net_inp = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="k_net_inp"
        )

        self.q_net_pos = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="q_net_pos"
        )
        self.k_net_pos = tf.keras.layers.Dense(
            self.n_head * self.d_head, use_bias=False, name="k_net_pos"
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

        q_head_inp = self.q_net_inp(inp)
        k_head_inp = self.k_net_inp(inp)

        q_head_pos = self.q_net_pos(pos_emb)
        k_head_pos = self.k_net_pos(pos_emb)

        v_head = self.v_net(inp)

        q_head_inp = self.drop_q(q_head_inp, training=training)
        k_head_inp = self.drop_k(k_head_inp, training=training)

        q_head_pos = self.drop_q(q_head_pos, training=training)
        k_head_pos = self.drop_k(k_head_pos, training=training)

        v_head = self.drop_v(v_head, training=training)

        q_head_inp = tf.reshape(q_head_inp, [-1, slen, self.d_head, self.n_head])
        k_head_inp = tf.reshape(k_head_inp, [-1, slen, self.d_head, self.n_head])

        q_head_pos = tf.reshape(q_head_pos, [slen, self.d_head, self.n_head])
        k_head_pos = tf.reshape(k_head_pos, [slen, self.d_head, self.n_head])

        v_head = tf.reshape(v_head, [-1, slen, self.d_head, self.n_head])

        attn_score_inp = tf.einsum("bidh,bjdh->bijh", q_head_inp, k_head_inp)
        attn_score_inp = attn_score_inp * self.scale * self.softmax_attn_smoothing
        attn_prob_inp = tf.nn.softmax(attn_score_inp, axis=2)

        attn_score_pos = tf.einsum("idh,jdh->ijh", q_head_pos, k_head_pos)
        attn_score_pos = attn_score_pos * self.scale * self.softmax_attn_smoothing
        attn_prob_pos = tf.nn.softmax(attn_score_pos, axis=1)
        attn_prob_pos = tf.reshape(attn_prob_pos, [1, slen, slen, self.n_head])

        attn_prob = (1-self.gamma)*attn_prob_inp + self.gamma*attn_prob_pos

        attn_out = tf.einsum("bijh,bjdh->bihd", attn_prob, v_head)
        attn_out = tf.reshape(attn_out, [bsz, slen, -1])

        attn_out = self.o_net(attn_out)
        attn_out = self.drop_o(attn_out, training=training)

        return [attn_out, attn_prob]


