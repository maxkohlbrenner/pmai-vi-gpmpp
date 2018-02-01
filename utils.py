import tensorflow as tf 
import numpy as np
import math

def get_test_log_likelihood():
    X_test_ph = tf.placeholder(tf.float32, [None, None],  name='evaluation_points')
    Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')
    
    K_zz_inv_ph = tf.placeholder(tf.float32, [None, None], name='Kzz_inverse')
    
    S_ph = tf.placeholder(tf.float32, [None, None], name='final_S')
    m_ph = tf.placeholder(tf.float32, [None],           name='final_mean')
    
    a_ph = tf.placeholder(tf.float32, [None],name='final_alphas')
    g_ph = tf.placeholder(tf.float32,None,name='final_gamma')
    
    Tmins = tf.reduce_min(Z_ph, axis=0)
    Tmaxs = tf.reduce_max(Z_ph, axis=0)
    
    with tf.name_scope('intergration-over-region-T (loglikelike testdata)':
        psi_matrix = psi_term(Z_ph,Z_ph,a_ph,g_gh,Tmins,Tmaxs)
        integral_over_T = T_Integral(m_ph,S_ph,K_zz_inv_ph,psi_matrix,g_ph,Tmins,Tmaxs)

    with tf.name_scope('expectation_at_datapoints (loglikelike testdata)'):
        mu_t, sig_t_sqr = mu_tilde_square(X_test_ph,Z_ph,S,m,K_zz_inv, a,g)
        exp_term = exp_at_datapoints(mu_t**2,sig_t_sqr,C)

    with tf.name_scope('KL-divergence'):
        kl_term_op = kl_term(m, S, K_zz, K_zz_inv, u_ph, L)

    with tf.name_scope('calculate_bound'):
        lower_bound = -integral_over_T + exp_term - kl_term_op
    

def build_eval_graph():
    X_eval_ph = tf.placeholder(tf.float32, [None, None],  name='evaluation_points')
    Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')
    
    K_zz_inv_ph = tf.placeholder(tf.float32, [None, None], name='Kzz_inverse')
    
    S_ph = tf.placeholder(tf.float32, [None, None], name='final_S')
    m_ph = tf.placeholder(tf.float32, [None],           name='final_mean')
    
    a_ph = tf.placeholder(tf.float32, [None],name='final_alphas')
    g_ph = tf.placeholder(tf.float32,None,name='final_gamma')
    
    with tf.name_scope('evaluation'):
        mu_t_eval, sig_t_sqr_eval = mu_tilde_square(X_eval_ph,Z_ph,S_ph,m_ph,K_zz_inv_ph, a_ph,g_ph)
        lam = mu_t_eval**2
        lam_var = sig_t_sqr_eval #TODO: lam_var = sig_t_sqr_eval**2 ???

    return lam, lam_var, Z_ph,X_eval_ph,K_zz_inv_ph, S_ph, m_ph, a_ph,g_ph


def build_graph(num_inducing_points = 11,dim = 1,a_init_val=1, g_init_val=1.):

    ## ######### ##
    # PLACEHOLDER # 
    ## ######### ##
    Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')
    u_ph = tf.placeholder(tf.float32, [],           name='inducing_point_mean')
    X_ph = tf.placeholder(tf.float32, [None, None],  name='input_data')
    #a_ph = tf.placeholder(tf.float32, [None] ,name='alphas')

    # TODO: set constants as variables and create two optimizers with var_lists to optimize with/without hyperparams
    #a_const = 1 * tf.ones([1]) # dimension = tf.shape(Z_ph)[1]
    #g_const = tf.ones([1]) # later we have to define gamma as variable
    C = tf.constant(0.57721566)

    #Tlims
    Tmins = tf.reduce_min(Z_ph, axis=0)
    Tmaxs = tf.reduce_max(Z_ph, axis=0)

    # TODO: use shape of Z_ph instead? Right now, the number is defined twice (once here, one above in the definition of Z)
    # num_inducing_points = 11 # tf.shape(Z_ph)[0] 

    ## ####### ##
    # VARIABLES # 
    ## ####### ##

    with tf.name_scope('variational_distribution_parameters'):
        #alphas
        a_init = tf.ones([dim])*a_init_val
        a = tf.Variable(a_init, name = 'variational_alphas')
        
        #gamma
        g_base = tf.Variable(g_init_val, name = 'variational_gamma')
        g = tf.abs(g_base)
        
        # mean
        m_init = tf.ones([num_inducing_points])
        m = tf.Variable(m_init, name='variational_mean')

        # vectorized version of covariance matrix S (ensure valid covariance matrix)
        vech_size   = tf.cast( (num_inducing_points * (num_inducing_points+1)) / 2, tf.int32)
        vech_indices= tf.transpose(tf_tril_indices(num_inducing_points))
        
        # L_vech_init = tf.ones([vech_size]) 
        L_vech_init = tf.random_normal([vech_size], stddev=0.35)

        L_vech = tf.Variable(L_vech_init)
        L_shape = tf.constant([num_inducing_points, num_inducing_points])
        L_st = tf.SparseTensor(tf.to_int64(vech_indices), L_vech, tf.to_int64(L_shape))
        L = tf.sparse_add(tf.zeros(L_shape), L_st)
        # L = tf.sparse_add(tf.eye(L_shape[0], num_columns=L_shape[1]), L_st)
        S = tf.matmul(L, tf.transpose(L), name='variational_covariance') 

    # kernel calls
    K_zz  = ard_kernel(Z_ph, Z_ph, gamma=g, alphas=a)
    K_zz_inv = tf.matrix_inverse(K_zz)

    with tf.name_scope('intergration-over-region-T'):
        psi_matrix = psi_term(Z_ph,Z_ph,a,g,Tmins,Tmaxs)
        integral_over_T = T_Integral(m,S,K_zz_inv,psi_matrix,g,Tmins,Tmaxs)

    with tf.name_scope('expectation_at_datapoints'):
        mu_t, sig_t_sqr = mu_tilde_square(X_ph,Z_ph,S,m,K_zz_inv, a,g)
        exp_term = exp_at_datapoints(mu_t**2,sig_t_sqr,C)

    with tf.name_scope('KL-divergence'):
        kl_term_op = kl_term(m, S, K_zz, K_zz_inv, u_ph, L)

    with tf.name_scope('calculate_bound'):
        lower_bound = -integral_over_T + exp_term - kl_term_op

    tf.summary.scalar('variational_lower_bound',    tf.squeeze(lower_bound)     )
    tf.summary.scalar('integral_over_T',            tf.squeeze(integral_over_T) )
    tf.summary.scalar('exp_term',                   tf.squeeze(exp_term)        )
    tf.summary.scalar('kl_div',                     kl_term_op                  )

    # m_grad = tf.gradients(kl_term_op, [m])[0]  
    # L_vech_grad = tf.gradients(kl_term_op, [L_vech])[0]

    interesting_gradient = tf.gradients(lower_bound, [exp_term])[0]

    merged = tf.summary.merge_all()
    
    return lower_bound, merged, Z_ph, u_ph, X_ph, m, S,L_vech, interesting_gradient,K_zz_inv,a,g_base,K_zz


def ard_kernel(X1, X2, gamma=1., alphas=None):
    # X1:  (n1 x d)
    # X2:  (n2 x d)
    # out: (n1 x n2
    with tf.name_scope('ard_kernel'):
        if alphas is None:
            alphas = tf.ones([tf.shape(X1)[1]])
        return gamma * tf.reduce_prod(tf.exp(- (tf.expand_dims(X1, 1) - tf.expand_dims(X2, 0))**2 / (2 * tf.expand_dims(tf.expand_dims(alphas, 0), 0))), axis=2) 


def mu_tilde_square(X_data, Z, S, m, Kzz_inv, a, g):
    k_zx = ard_kernel( Z,X_data, gamma = g, alphas=a)
    k_xz = tf.transpose(k_zx)
    K_xx = ard_kernel(X_data, X_data, gamma = g, alphas=a)
    mu = tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(m,1)),Kzz_inv)
                                                     ,k_zx)
 
    sig_sqr = K_xx - tf.matmul(tf.matmul(k_xz,Kzz_inv),k_zx) + tf.matmul(tf.matmul(tf.matmul(tf.matmul(k_xz,Kzz_inv),S),Kzz_inv),k_zx)

    return mu,sig_sqr

def kl_term(m, S, K_zz, K_zz_inv, u_ovln, L):
    # mean_diff = (u_ovln * tf.ones([tf.shape(Z_ph)[0]]) - m)
    mean_diff = tf.expand_dims(u_ovln * tf.ones([tf.shape(m)[0]]) - m, 1)
    first  = tf.trace(tf.matmul(K_zz_inv, S), name='kl_first')

    # #########################################
    # TODO: solve matrix determinant Problem
    # Approaches:

    # 1. naive impl of determinants 
    # -> Problem: NaN as Determimants get very large for big matrices
    # Code:
    # kzz_det = tf.matrix_determinant(K_zz) 
    # S_det   = tf.matrix_determinant(S)
    # second = tf.log(kzz_det / S_det, name='kl_second')

    # 2. Logdet and Cholesky decomp
    # -> Problem: Cholesky decomp not always possible (only pos semidefinite by our constr?)
    # -> Adding Eye to S might be a possible solution
    K_zz_logdet = tf.linalg.logdet(K_zz)
    posdef_stabilizer = tf.eye(tf.shape(S)[0]) * 0.0001
    S_logdet =  tf.linalg.logdet(S + posdef_stabilizer)
    # S_logdet = 2 * tf.reduce_sum(tf.log(tf.diag_part(L)))
    # posdef_stabilizer = tf.eye(L_shape[0]) * lambda
    second = tf.subtract(K_zz_logdet, S_logdet, name='kl_second')

    # 3. Using tf.slogdet
    # -> Problem: slogdet doesn't seem to have a gradient defined
    #kzz_lds, kzz_ldav = tf.linalg.slogdet(tf.expand_dims(K_zz, 0))
    #K_zz_logdet = kzz_lds[0] * kzz_ldav[0]
    #S_lds, S_ldav = tf.linalg.slogdet(tf.expand_dims(S, 0))
    #S_logdet = S_lds[0] * S_ldav[0]
    #second = tf.subtract(K_zz_logdet, S_logdet, name='kl_second')
    # #########################################

    

    third  = tf.to_float(tf.shape(m)[0], name='kl_third')
    # fourth = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(mean_diff, tf.transpose(K_zz_inv)), axis=1) , mean_diff))
    
    fourth = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(mean_diff), K_zz_inv), mean_diff), name='kl_fourth')
    
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
    Remark: this one handles edge cases correctly
    '''
    
    table_keys = table[0]
    table_vals = table[1]
    
    num_keys = tf.shape(keys)[0]
    
    # index from table value with closest table_key to given key
    table_ind = tf.argmin( tf.abs(tf.expand_dims(table_keys, 0) - tf.expand_dims(keys, 1) ) , output_type=tf.int32, axis=1)
    
    top_keys = tf.gather(table_keys, table_ind)
    
    # difference from closest table_key to given key
    shift     = keys - top_keys

    # out of bounds switch on the left
    table_min_key = table_keys[0]
    oob_l_switch  = tf.sign(tf.sign( keys - table_min_key) - 0.5)
    # out of bounds switch on the right
    table_max_key = table_keys[tf.shape(table)[1] - 1]
    oob_r_switch  = -1 * tf.sign(tf.sign( keys - table_max_key ) - 0.5)
    
    # real shift or shift to the smaller key if shift == 0
    nonzero_shift = (tf.sign(tf.abs(shift)) - 1)  + shift
    # shift to the right if table_ind is 0
    # adapted_shift = nonzero_shift * (-1 * ti_zero_indicator) 
    adapted_shift = nonzero_shift * oob_l_switch
    # shift to the left if key > max_key
    adapted_shift = adapted_shift * oob_r_switch
    
    # either -1 or 1, direction to the second table entry used for gradient calculation
    next_entry_shift = tf.cast(tf.sign(adapted_shift), tf.int32) 
    
    table_ind_shifted = table_ind + next_entry_shift
    
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