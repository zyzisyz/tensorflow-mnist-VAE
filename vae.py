import tensorflow as tf


# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
    '''
    x是输入的数据
    n_hidden是hidden layer的数目
    n_output是MLP输出的数目，即latent z
    keep_prob是
    '''
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable(
            'w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable(
            'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        w_out = tf.get_variable(
            'w_out', [h1.get_shape()[1], n_output * 2], initializer=w_init)
        b_out = tf.get_variable('b_out', [n_output * 2], initializer=b_init)

        # tf.matmul矩阵与矩阵相乘 禁止与标量相乘
        gaussian_params = tf.matmul(h1, w_out) + b_out

        # The mean parameter is unconstrained
        # 这步没看懂 mean平均值为啥可以这样求

        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability

        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])  # 标准差

    return mean, stddev


# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable(
            'w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)

        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable(
            'w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        w_out = tf.get_variable(
            'w_out', [h1.get_shape()[1], n_output], initializer=w_init)
        b_out = tf.get_variable('b_out', [n_output], initializer=b_init)

        # 为啥sigmod函数
        y = tf.sigmoid(tf.matmul(h1, w_out) + b_out)

    return y


# 算loss的函数
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):

    # encoding
    mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # decoding
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)

    # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    # loss

    # reduce_sum降维， 1 级进行仇和，[1+1+1, 1+1+1] = [3, 3]
    marginal_likelihood = tf.reduce_sum(
        x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
    
    # KL散度
    KL_divergence = 0.5 * \
        tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                      tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, -marginal_likelihood, KL_divergence


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y
