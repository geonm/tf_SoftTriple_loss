import os
import sys
import numpy as np
import tensorflow as tf

def softtriple(gt, embeddings, dim_features, num_class, num_centers=2, p_lambda=20.0, p_tau=0.2, p_gamma=0.1, p_delta=0.01, with_reg=True):
    large_centers = tf.get_variable(name='feature_extractor/large_centers', shape=[num_class * num_centers, dim_features],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                    trainable=True)
    large_centers = tf.nn.l2_normalize(large_centers, axis=-1)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    large_logits = tf.matmul(embeddings, large_centers, transpose_b=True) # [batch_size, num_class * num_centers]

    batch_size = tf.shape(large_logits)[0]

    rs_large_logits = tf.reshape(large_logits, [batch_size, num_centers, num_class])

    exp_rs_large_logits = tf.exp((1.0 / p_gamma) * rs_large_logits)

    sum_rs_large_logits = tf.reduce_sum(exp_rs_large_logits, axis=1, keepdims=True)

    coeff_large_logits = exp_rs_large_logits / sum_rs_large_logits

    rs_large_logits = tf.multiply(rs_large_logits, coeff_large_logits)

    logits = tf.reduce_sum(rs_large_logits, axis=1, keepdims=False)

    # get labels_map
    gt = tf.reshape(gt, [-1]) # e.g., [0, 7, 3, 22, 39, ...]

    gt_int = tf.cast(gt, tf.int32)

    labels_map = tf.one_hot(gt_int, depth=num_class, dtype=tf.float32)

    # subtract p_delta
    delta_map = p_delta * labels_map

    logits_delta = logits - delta_map
    scaled_logits_delta = p_lambda * (logits_delta)

    # get xentropy loss
    loss_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits_delta, labels=labels_map)
    loss_xentropy = tf.reduce_mean(loss_xentropy, name='loss_xentropy')

    # get regularizer terms
    loss_reg = 0.0
    if with_reg:
        # get R function
        # large_centers = [num_class * num_centers, dim_features]
        sim_large_centers = tf.abs(tf.matmul(large_centers, large_centers, transpose_b=True)) # [num_class * num_centers, num_class * num_centers]

        # check error
        #sim_large_centers = tf.where(sim_large_centers > 1.0, tf.ones_like(sim_large_centers, dtype=tf.float32), sim_large_centers)

        dist_large_centers = tf.sqrt(tf.abs(2.0 - 2.0 * sim_large_centers) + 1e-10)
        checkerboard = tf.range(num_class, dtype=tf.int32)
        checkerboard = tf.one_hot(checkerboard, depth=num_class, dtype=tf.float32)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, num_centers, axis=0)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, num_centers, axis=1)

        dist_large_centers = tf.multiply(dist_large_centers, checkerboard)

        mask = tf.ones_like(dist_large_centers, dtype=tf.float32) - tf.eye(num_class * num_centers, dtype=tf.float32)

        dist_large_centers = p_tau * tf.multiply(dist_large_centers, mask)

        reg_numer = tf.reduce_sum(dist_large_centers) / 2.0

        reg_denumer = num_class * num_centers * (num_centers - 1.0)

        loss_reg = reg_numer / reg_denumer

    # l2 reg loss
    #reg_embeddings = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings), 1))
    #reg_centers = tf.reduce_mean(tf.reduce_sum(tf.square(large_centers), 1))
    #loss_l2_reg = tf.multiply(0.25 * 0.002, reg_embeddings + reg_centers, name='loss_l2_reg')

    total_loss = loss_xentropy

    return total_loss  
