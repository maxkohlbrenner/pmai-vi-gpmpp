import tensorflow as tf 
import numpy as np

def tf_tril_indices(N, k=0):
    '''
    Helper function to construct a triangular matrix from a vector.
    (Later used to construct a valid covariance matrix)

    Code found on github:
    https://github.com/GPflow/GPflow/issues/439
    '''
    M1 = tf.tile(tf.expand_dims(tf.range(N), axis=0), [N,1])
    M2 = tf.tile(tf.expand_dims(tf.range(N), axis=1), [1,N])
    mask = (M1-M2) >= -k
    ix1 = tf.boolean_mask(M2, tf.transpose(mask))
    ix2 = tf.boolean_mask(M1, tf.transpose(mask))
    return ix1, ix2 

def load_lookup_table(file = 'g_lookup_table.npy'):
    # load lookup table of precomputed values for the g function
    return tf.convert_to_tensor(np.load(file), dtype=tf.float32)

def table_lookup_op_parallel(table, keys):
    '''
    Return a tensorflow op that approximates a function by linear interpolation from a precomputed lookup table

    TODO: handle edge cases
    '''
    
    table_keys = table[0]
    table_vals = table[1]
    
    num_keys = tf.shape(keys)[0]
    
    # index from table value with closest table_key to given key
    table_ind = tf.argmin( tf.abs(tf.expand_dims(table_keys, 0) - tf.expand_dims(keys, 1) ) , output_type=tf.int32, axis=1)
    
    top_keys = tf.gather(table_keys, table_ind)
    
    # difference from closest table_key to given key
    shift     = keys - top_keys
    
    # -1 if table_ind == 0, 1 if table_ind > 0 (table ind always >= 0)
    ti_zero_indicator = - tf.sign( tf.cast(tf.subtract(tf.ones([num_keys], dtype=tf.int32), tf.sign(table_ind)), dtype=tf.float32) - tf.constant(.5))
    
    # shift to next table entry (used for gradient computation)
    # if table_key == key:
    # if key != 0 : next smaller table_key is used
    # if key == 0 : next greater table_key is used
    nonzero_shift = (1 - tf.sign(tf.abs(shift))) * (-1. * ti_zero_indicator) + shift
    
    shift_step        = tf.cast(tf.sign(nonzero_shift), tf.int32) 
    table_ind_shifted = table_ind + shift_step
    
    table_val      = tf.gather(table_vals, table_ind)
    next_table_val = tf.gather(table_vals, table_ind_shifted)
    
    table_key      = tf.gather(table_keys, table_ind)
    next_table_key = tf.gather(table_keys, table_ind_shifted)
    
    dx = (next_table_key - table_key)
    dy = (next_table_val - table_val)
    
    gradient               = dy / dx
    interpolated_fun_value = table_val + shift * gradient
    
    return tf.stop_gradient(gradient) * keys + tf.stop_gradient(interpolated_fun_value - gradient * keys)