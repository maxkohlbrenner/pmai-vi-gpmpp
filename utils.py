import tensorflow as tf 
import numpy as np
import math

def build_graph(num_inducing_points = 11):

    ## ######### ##
    # PLACEHOLDER # 
    ## ######### ##
    Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')
    u_ph = tf.placeholder(tf.float32, [],           name='inducing_point_mean')
    X_ph =tf.placeholder(tf.float32, [None, None],  name='input_data')

    # TODO: set constants as variables and create two optimizers with var_lists to optimize with/without hyperparams
    a_const = tf.ones([1]) # dimension = tf.shape(Z_ph)[1]
    g_const = tf.ones([1]) # later we have to define gamma as variable
    C = tf.constant(0.57721566)

    #Tlims
    Tmins = tf.reduce_min(Z_ph, axis=0)
    Tmaxs = tf.reduce_max(Z_ph, axis=0)

    # TODO: use shape of Z_ph instead? Right now, the number is defined twice (once here, one above in the definition of Z)
    # num_inducing_points = 11 # tf.shape(Z_ph)[0] 

    ## ####### ##
    # VARIABLES # 
    ## ####### ##
    # mean
    m_init = tf.ones([num_inducing_points])
    m = tf.Variable(m_init)

    # vectorized version of covariance matrix S (ensure valid covariance matrix)
    vech_size   = (num_inducing_points * (num_inducing_points+1)) / 2 
    vech_indices= tf.transpose(tf_tril_indices(num_inducing_points))
    L_vech_init = tf.ones([vech_size])
    L_vech = tf.Variable(L_vech_init)
    L_shape = tf.constant([num_inducing_points, num_inducing_points])
    L_st = tf.SparseTensor(tf.to_int64(vech_indices), L_vech, tf.to_int64(L_shape))
    L = tf.sparse_add(tf.zeros(L_shape), L_st)
    S = tf.matmul(L, tf.transpose(L))

    # kernel calls
    K_zz  = ard_kernel(Z_ph, Z_ph, alphas=a_const)
    K_zz_inv = tf.matrix_inverse(K_zz)

    with tf.name_scope('intergration-over-region-T'):
        psi_matrix = psi_term(Z_ph,Z_ph,a_const,g_const,Tmins,Tmaxs)
        integral_over_T = T_Integral(m,S,K_zz_inv,psi_matrix,g_const,Tmins,Tmaxs)

    with tf.name_scope('expectation_at_datapoints'):
        mu_t_sqr, sig_t_sqr = mu_tilde_square(X_ph,Z_ph,S,m,K_zz_inv, a_const)
        exp_term = exp_at_datapoints(mu_t_sqr,sig_t_sqr,C)

    with tf.name_scope('KL-divergence'):
        kl_term_op = kl_term(m, S, K_zz, K_zz_inv, u_ph)
        tf.summary.scalar('kl_div', kl_term_op)

    with tf.name_scope('calculate_bound'):
        lower_bound = -integral_over_T + exp_term - kl_term_op

    m_grad = tf.gradients(kl_term_op, [m])[0]  
    L_vech_grad = tf.gradients(kl_term_op, [L_vech])[0]


    merged = tf.summary.merge_all()
    
    return lower_bound, merged, Z_ph, u_ph, X_ph, m, S


def ard_kernel(X1, X2, gamma=1., alphas=None):
    # X1:  (n1 x d)
    # X2:  (n2 x d)
    # out: (n1 x n2
    with tf.name_scope('ard_kernel'):
        if alphas is None:
            alphas = tf.ones([tf.shape(X1)[1]])
        return gamma * tf.reduce_prod(tf.exp(- (tf.expand_dims(X1, 1) - tf.expand_dims(X2, 0))**2 / (2 * tf.expand_dims(tf.expand_dims(alphas, 0), 0))), axis=2) 


def mu_tilde_square(X_data, Z, S, m, Kzz_inv, a_const):
    k_zx = ard_kernel( Z,X_data, alphas=a_const)
    k_xz = tf.transpose(k_zx)
    K_xx = ard_kernel(X_data, X_data, alphas=a_const)
    mu_sqr = tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(m,1)),Kzz_inv)
                                                     ,k_zx)**2
 
    sig_sqr = K_xx - tf.matmul(tf.matmul(k_xz,Kzz_inv),k_zx) + tf.matmul(tf.matmul(tf.matmul(tf.matmul(k_xz,Kzz_inv),S),Kzz_inv),k_zx)

    return mu_sqr,sig_sqr

def kl_term(m, S, K_zz, K_zz_inv, u_ovln):
    # mean_diff = (u_ovln * tf.ones([tf.shape(Z_ph)[0]]) - m)
    mean_diff = tf.expand_dims(u_ovln * tf.ones([tf.shape(m)[0]]) - m, 1)
    first  = tf.trace(tf.matmul(K_zz_inv, S))
    second = tf.log(tf.matrix_determinant(K_zz) / tf.matrix_determinant(S))
    third  = tf.to_float(tf.shape(m)[0])
    # fourth = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(mean_diff, tf.transpose(K_zz_inv)), axis=1) , mean_diff))
    
    fourth = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(mean_diff), K_zz_inv), mean_diff))
    
    return 0.5 * (first  + second - third + fourth)

def psi_term(Z1, Z2,a,g,Tmin,Tmax):
    z_ovln = (tf.expand_dims(Z1,1)+tf.expand_dims(Z2,0))/2
    a_r = tf.expand_dims(tf.expand_dims(a,0),1)
    
    pi = tf.constant(math.pi)
    
    return (g**2) * tf.reduce_prod(-(tf.sqrt(pi * a_r)/2
                   ) * tf.exp(-tf.pow(tf.expand_dims(Z1,1) - tf.expand_dims(Z2,0),2) / (4 * a_r)
                             ) * (tf.erf((z_ovln-tf.expand_dims(tf.expand_dims(Tmax,0),1))/tf.sqrt(a_r)
                                     ) - tf.erf((z_ovln-tf.expand_dims(tf.expand_dims(Tmin,0),1))/tf.sqrt(a_r))),2)

def T_Integral(m, S, Kzz_inv,psi, g,Tmin, Tmax):

    e_qf = tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(m,1)),Kzz_inv),psi),Kzz_inv),tf.expand_dims(m,1))
    T = tf.reduce_prod(Tmax-Tmin)
    var_qf = g * T - tf.trace(tf.matmul(Kzz_inv,psi)) + tf.trace(tf.matmul(tf.matmul(tf.matmul(Kzz_inv,S),Kzz_inv),psi))
    return (e_qf + var_qf)

def G(mu_sqr,sig_sqr_matrix):
    
    sig_sqr = tf.diag_part(sig_sqr_matrix)
    lookup_x = - tf.squeeze(mu_sqr) / (2*sig_sqr)
    
    lookup_table = load_lookup_table()
    return table_lookup_op_parallel(lookup_table, lookup_x)
    
    
def exp_at_datapoints(mu_sqr,sig_sqr,C):
    return tf.reduce_sum(-G(mu_sqr,sig_sqr)+tf.log(mu_sqr/2)-C,axis=1)


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


def get_scp_samples(rate_function, region_lims, upper_bound):
    
    # region lims: np.array of shape (D x 2), D dimension of input space
    D = region_lims.shape[0]
    
    assert(np.alltrue(region_lims[:,0] <= region_lims[:,1])) # , 'First entries of regional limits need to be smaller or equal to the second entries')
    
    # 1. calc measure
    V = np.prod(np.absolute(region_lims[:,0] - region_lims[:,1]), axis=0)
    # 2. sample from poisson 
    J = np.random.poisson(V * upper_bound)
    # 3. sample locations uniformly
    low  = region_lims[:,0]
    high = region_lims[:,1]
    sample_candidates = np.random.uniform(low=low, high=high, size=(J, D))
    
    vals = rate_function(sample_candidates)
    
    # 5. iterate over points and accept/reject
    R = np.random.uniform(size=J) * upper_bound
    accept = R < vals # R < logistic(vals) * upper_bound
    
    return sample_candidates[accept], R[accept], sample_candidates, R