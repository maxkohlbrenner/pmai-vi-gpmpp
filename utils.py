import tensorflow as tf 
import numpy as np
import math

na = np.newaxis

def train_parameters(data, ind_point_number, Tlims, optimize_inducing_points = True, train_hyperparameters = False, learning_rate=0.0001, max_iterations = 1000, gamma_init = 0.3, alphas_init = 1, log_dir=None, run_prefix=None):
    ## ######## ##
    # PARAMETERS #
    ## ######## ##

    print('Begin training')

    # init path if not specified
    if log_dir == None:
        log_dir        = 'logs'
    if run_prefix == None:
        if optimize_inducing_points:
            ip_part = '_ipopt'
        else:
            ip_part=''
        if not train_hyperparameters:
            hp_part = '_hpfix'
        else:
            hp_part = ''

        run_prefix = 'vipp{}{}_ipn{}_lr{}_{}iterations'.format(ip_part, hp_part, ind_point_number, learning_rate, max_iterations)
    
    # dimensionality of the space
    D = data.shape[1]

    if not optimize_inducing_points:
        # Tlims is of shape (D,2),  [[min, max] for each dimension]
        ranges = [np.linspace(lims[0], lims[1], ind_point_number) for lims in Tlims]
        grid   = np.array(np.meshgrid(*ranges))
        
        Z = np.stack(grid, len(grid)).reshape(ind_point_number ** D, D)

        print('Fixed inducing points on a grid')
        print('Z.shape: {}'.format(Z.shape))

        num_inducing_points = ind_point_number ** D

    else:
        # optimize inducing point locs, variable is initialized in build_graph
        Z = None 
        num_inducing_points = ind_point_number


    ## ######### ##
    # BUILD GRAPH #
    ## ######### ##
    tf.reset_default_graph()
    lower_bound, merged, Z_ph, u_ph, X_ph, m, S,L_vech, interesting_gradient, K_zz_inv, alphas, gamma, Kzz, omegas = build_graph(Tlims, num_inducing_points, D,alphas_init,gamma_init, optimize_inducing_points)

    variables = [m,L_vech]
    
    if train_hyperparameters:
        variables = variables + [alphas, gamma]

    if optimize_inducing_points:
        variables = variables + [omegas]
    
    with tf.name_scope('optimization'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(-lower_bound,var_list=variables)


    # inspected_op = tf.get_default_graph().get_tensor_by_name("KL-divergence/truediv:0")
    #interesting_gradient = tf.gradients(lower_bound, [inspected_op])[0]

    with tf.name_scope('nanchecks'):
        check = tf.add_check_numerics_ops()

    ## ########## ##
    # OPTIMIZATION #f
    ## ########## ##
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir + '/' + run_prefix, sess.graph)

        # S_init_val = sess.run([S])
        # print(S_init_val)
        # print(np.all(np.linalg.eigvals(S_init_val) >= 0))

        # print(S_init_val)


        feed_dict = {u_ph:0. ,X_ph:data}
        if not optimize_inducing_points:
            feed_dict[Z_ph] = Z

        init_state = sess.run([merged, lower_bound, m, S, Kzz], feed_dict=feed_dict)
        writer.add_summary(init_state[0], 0)
        # print(init_state[1:])

        for i in range(max_iterations):

            _, lower_bound_val, m_val, S_val, Z_locs, grad_val, summary, Kzz_inv, _, alphas_vals, gamma_val, Kzz_val = sess.run([train_step, lower_bound, m, S, Z_ph, interesting_gradient, merged, K_zz_inv, check, alphas, gamma, Kzz], feed_dict=feed_dict)
            writer.add_summary(summary, i+1)

            # print(Kzz_val)
            # print('------------')
            #print(lower_bound_val)
            #print(Kzz_val)
            #print(g_val)
            # print(np.min(S_val))
            # print(np.max(S_val))
            # print(np.allclose(S_val, S_val.T))
            # print(np.all(np.linalg.eigvals(S_val) >= 0))

            #print(sess.run([S]))
            #print(sess.run([L_vech_grad]))
            
    
    return m_val, S_val, Kzz_inv, alphas_vals, Z_locs, gamma_val


def evaluation(m_val,S_val,Kzz_inv,alphas_vals,gamma_val,Z, eval_grid):

    #build graph
    lam, lam_var, Z_ph,X_eval_ph, K_zz_inv_ph, S_ph, m_ph,alphas_ph,gamma_ph  = build_eval_graph()

    #run session
    with tf.Session() as sess:
        lam_vals,lam_var_vals = sess.run([lam,lam_var], feed_dict={Z_ph:Z, X_eval_ph:eval_grid, K_zz_inv_ph: Kzz_inv, S_ph:S_val, m_ph:m_val, alphas_ph:alphas_vals, gamma_ph:gamma_val})

    return lam_vals,lam_var_vals

def build_2d_grid(lims, resolution):

    x = np.linspace(lims[0,0], lims[0,1], resolution)[:,na]
    y = np.linspace(lims[1,0], lims[1,1], resolution)[:,na]
    xx, yy = np.meshgrid(x, y)
    grid = np.array([xx, yy]).transpose(1,2,0).reshape(resolution**2, 2)

    return grid


def get_test_log_likelihood():
    X_test_ph = tf.placeholder(tf.float32, [None, None],  name='evaluation_points')
    Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')
    
    K_zz_inv_ph = tf.placeholder(tf.float32, [None, None], name='Kzz_inverse')
    
    S_ph = tf.placeholder(tf.float32, [None, None], name='final_S')
    m_ph = tf.placeholder(tf.float32, [None],       name='final_mean')
    
    alphas_ph = tf.placeholder(tf.float32, [None],name='final_alphas')
    gamma_ph = tf.placeholder(tf.float32,None,name='final_gamma')
    
    # TODO: replace by the actual limits
    Tmins = tf.reduce_min(Z_ph, axis=0)
    Tmaxs = tf.reduce_max(Z_ph, axis=0)
    
    C = tf.constant(0.57721566)

    with tf.name_scope('intergration-over-region-T_test_data'):
        psi_matrix = psi_term(Z_ph, Z_ph, alphas_ph, gamma_ph, Tmins, Tmaxs)
        integral_over_T = T_Integral(m_ph,S_ph,K_zz_inv_ph,psi_matrix,g_ph,Tmins,Tmaxs)

    with tf.name_scope('expectation_at_datapoints_test_data'):
        mu_t, sig_t_sqr = mu_tilde_square(X_test_ph,Z_ph,S_ph,m_ph,K_zz_inv_ph, alphas_ph, gamma_ph)
        exp_term = exp_at_datapoints(tf.square(mu_t),sig_t_sqr,C)

    with tf.name_scope('calculate_bound'):
        lower_bound = -integral_over_T + exp_term
        
    return lower_bound, Z_ph, X_test_ph, m_ph, S_ph,K_zz_inv_ph,alpha_ph,gamma_ph
    

def build_eval_graph():
    X_eval_ph = tf.placeholder(tf.float32, [None, None],  name='evaluation_points')
    Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')
    
    K_zz_inv_ph = tf.placeholder(tf.float32, [None, None], name='Kzz_inverse')
    
    S_ph = tf.placeholder(tf.float32, [None, None], name='final_S')
    m_ph = tf.placeholder(tf.float32, [None],           name='final_mean')
    
    alpha_ph = tf.placeholder(tf.float32, [None],name='final_alphas')
    gamma_ph = tf.placeholder(tf.float32,None,name='final_gamma')
    
    with tf.name_scope('evaluation'):
        mu_t_eval, sig_t_sqr_eval = mu_tilde_square(X_eval_ph,Z_ph,S_ph,m_ph,K_zz_inv_ph, alpha_ph,gamma_ph)
        lam = mu_t_eval**2
        lam_var = sig_t_sqr_eval #TODO: lam_var = sig_t_sqr_eval**2 ???

    return lam, lam_var, Z_ph,X_eval_ph,K_zz_inv_ph, S_ph, m_ph, alpha_ph,gamma_ph


def build_graph(Tlims, num_inducing_points = 11,dim = 1,alphas_init_val=1, gamma_init_val=1., optimize_inducing_points=False):

    ## ######### ##
    # PLACEHOLDER # 
    ## ######### ##
    if not optimize_inducing_points: # TODO change back to None
        Z_ph = tf.placeholder(tf.float32, [None, None], name='inducing_point_locations')

    u_ph = tf.placeholder(tf.float32, [],           name='inducing_point_mean')
    X_ph = tf.placeholder(tf.float32, [None, None],  name='input_data')
    #a_ph = tf.placeholder(tf.float32, [None] ,name='alphas')

    # TODO: set constants as variables and create two optimizers with var_lists to optimize with/without hyperparams
    #a_const = 1 * tf.ones([1]) # dimension = tf.shape(Z_ph)[1]
    #g_const = tf.ones([1]) # later we have to define gamma as variable
    C = tf.constant(0.57721566)

    # 
    Tlims = tf.constant(Tlims, dtype='float')
    assert(Tlims.shape == (dim,2))

    #Tlims
    Tmins = tf.reduce_min(Tlims, axis=1)
    Tmaxs = tf.reduce_max(Tlims, axis=1)

    assert(Tmins.dtype == tf.float32)
    assert(len(Tmins.shape) == 1)

    ## ####### ##
    # VARIABLES # 
    ## ####### ##

    if optimize_inducing_points:
        # optimize inducing point location
        with tf.name_scope('inducing_point_optimization'):
            omegas_init = tf.random_uniform([num_inducing_points, dim])
            omegas      = tf.Variable(omegas_init, dtype='float', name='ind_point_omegas')

            dim_mean    = tf.reduce_mean(Tlims, axis=1)
            dim_shifter = tf.subtract(Tmins, Tmaxs, name= 'ind_point_ranges') / 2

            dim_mean    = tf.expand_dims(dim_mean,    0)
            dim_shifter = tf.expand_dims(dim_shifter, 0)

            assert(dim_mean.shape    == (1, dim))
            assert(dim_shifter.shape == (1, dim))

            Z_ph = tf.subtract(dim_mean, dim_shifter * tf.sin(omegas), name='inducing_point_locations')
    else:
        omegas = None

    Tlims = tf.cast(Tlims, dtype='float32')


    with tf.name_scope('variational_distribution_parameters'):

        with tf.name_scope('kernel_hyperparameters'):

            #alphas
            alphas_init = tf.ones([dim])*alphas_init_val
            alphas = tf.Variable(alphas_init, name = 'variational_alphas')
            
            #gamma
            gamma_base = tf.Variable(gamma_init_val, name = 'variational_gamma')
            # TODO: choose how to treat gamma base
            gamma = tf.abs(gamma_base)

            tf.summary.scalar('gamma_base', gamma_base)
            tf.summary.tensor_summary('alphas', alphas)

        
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

    # kernel call
    K_zz  = ard_kernel(Z_ph, Z_ph, gamma=gamma, alphas=alphas)
    K_zz_inv = tf.matrix_inverse(K_zz)

    with tf.name_scope('integration-over-region-T'):
        
        with tf.name_scope('psi_matrix'):
            psi_matrix = psi_term(Z_ph,Z_ph,alphas,gamma,Tmins,Tmaxs)

        with tf.name_scope('T_integral'):
            integral_over_T = T_Integral(m,S,K_zz_inv,psi_matrix,gamma,Tmins,Tmaxs)

    with tf.name_scope('expectation_at_datapoints'):
        with tf.name_scope('mu_and_sig_calculation'):
            mu_t, sig_t_sqr = mu_tilde_square(X_ph,Z_ph,S,m,K_zz_inv, alphas, gamma)

        with tf.name_scope('squaring_that_mu'):
            mu_t_square = tf.square(mu_t)

        exp_term = exp_at_datapoints(mu_t_square,sig_t_sqr,C)

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
    
    return lower_bound, merged, Z_ph, u_ph, X_ph, m, S,L_vech, interesting_gradient, K_zz_inv, alphas, gamma_base,K_zz, omegas


def ard_kernel(X1, X2, gamma=1., alphas=None):
    # X1:  (n1 x d)
    # X2:  (n2 x d)
    # out: (n1 x n2
    with tf.name_scope('ard_kernel'):
        if alphas is None:
            alphas = tf.ones([tf.shape(X1)[1]])
        return gamma * tf.reduce_prod(tf.exp(- tf.square(tf.expand_dims(X1, 1) - tf.expand_dims(X2, 0)) / (2 * tf.expand_dims(tf.expand_dims(alphas, 0), 0))), axis=2) 


def mu_tilde_square(X_data, Z, S, m, Kzz_inv, a, g):
    '''
    N : num datapoints
    D : datapoint dimensionality
    M : number inducing points

    IN: 
    ---
    X_data   : (N, D)
    Z        : (M, D)
    S        : (M, M)
    m        : (M)
    K_zz_inv : (M, M)
    a        : (D)
    g        : ()

    OUT:
    ----
    mu      : (N)
    sig_sqr : (N)
    '''

    with tf.name_scope('K_ZX'):
        # k_zx : (M, N)
        k_zx = ard_kernel( Z,X_data, gamma = g, alphas=a)
    with tf.name_scope('K_XZ'):
        # k_xz : (N, M)
        k_xz = tf.transpose(k_zx, name='K_XZ')
    with tf.name_scope('K_XX'):
        # k_xx : (N, N)
        K_xx = ard_kernel(X_data, X_data, gamma = g, alphas=a)

    # mu = tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(m,1)),Kzz_inv),k_zx, name='mu')

    # mu : (N, M)dot(M, M)dot(M) = (N)
    mu = tf.squeeze( tf.matmul(tf.matmul(k_xz, Kzz_inv), tf.expand_dims(m, 1), name='mu') ) 

    # sig_sqr : (N, N) - (N, M)dot(M,M)dot(M,N)
    XX_cov = K_xx - tf.matmul(tf.matmul(k_xz,Kzz_inv),k_zx) + tf.matmul(tf.matmul(tf.matmul(tf.matmul(k_xz,Kzz_inv),S),Kzz_inv),k_zx)

    sig_sqr = tf.diag_part(XX_cov, name='sig_sqr')
    return mu, sig_sqr

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

    with tf.name_scope('log_of_determinant_ratio'):

        posdef_stabilizer = tf.eye(tf.shape(K_zz)[0]) * 0.01

        with tf.name_scope('K_zz_logdet'):
            K_zz_logdet = tf.linalg.logdet(K_zz + posdef_stabilizer)

        with tf.name_scope('S_logdet'):
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

def psi_term(Z1, Z2, alphas, gamma, Tmins, Tmaxs):
    '''
    D: dimensionality of the space
    M: number inducing points

    input shapes:
    ------------
    Z1     : (M, D)
    Z2     : (M, D)
    alphas : (D,)
    gamma  : (,)
    Tmins  : (D,)
    Tmaxs  : (D,)
    '''

    # broadcasting axes: 
    # 0: Z1 element
    # 1: Z2 element
    # 2: Dimension

    assert(Tmins.dtype == tf.float32)
    assert(Tmaxs.dtype == tf.float32)

    z_ovln = (tf.expand_dims(Z1,1)+tf.expand_dims(Z2,0))/2

    alphas_r = tf.expand_dims(tf.expand_dims(alphas,0),1)
    pi       = tf.constant(math.pi)

    factor   = - (tf.sqrt(pi * alphas_r)/2)

    with tf.name_scope('exp_part'):
        exp_part = tf.exp( - tf.square(tf.expand_dims(Z1,1) - tf.expand_dims(Z2,0)) / (4 * alphas_r), name='exp_part')

    with tf.name_scope('erf_part'):
        erf_part = tf.subtract( tf.erf((z_ovln-tf.expand_dims(tf.expand_dims(Tmaxs,0),1)) / tf.sqrt(alphas_r)), 
            tf.erf((z_ovln-tf.expand_dims(tf.expand_dims(Tmins,0),1)) / tf.sqrt(alphas_r)), name='erf_part')
    
    psi_matrix =  tf.multiply( tf.square(gamma), tf.reduce_prod(factor * exp_part * erf_part ,2), name='psi_matrix')

    return psi_matrix

def T_Integral(m, S, Kzz_inv,psi, gamma,Tmins, Tmaxs):
    '''
    D : dimensionality of space
    M : number of inducing points

    Input dims:
    m        : (M)
    S        : (M, M)
    K_zz_inf : (M, M)
    psi      : (M, M)
    gamma    : ()
    Tmins    : (D)
    Tmax s   : (D)
    '''

    e_qf = tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.expand_dims(m,0),Kzz_inv, name='firstmatdot'),psi),Kzz_inv),tf.expand_dims(m,1))
    
    T_measure = tf.reduce_prod(Tmaxs-Tmins)
    
    var_qf = gamma * T_measure - tf.trace(tf.matmul(Kzz_inv,psi)) + tf.trace(tf.matmul(tf.matmul(tf.matmul(Kzz_inv,S),Kzz_inv),psi))
    
    return (e_qf + var_qf)

def G_lookup(mu_sqr,sig_sqr):

    lookup_x = - tf.squeeze(mu_sqr) / (2*sig_sqr)
    lookup_table = load_lookup_table()

    return table_lookup_op_parallel(lookup_table, lookup_x)
    
    
def exp_at_datapoints(mu_sqr,sig_sqr,C):

    with tf.name_scope('G_lookup'):
        G_value = - G_lookup(mu_sqr,sig_sqr)

    with tf.name_scope('log_of_sig_sqr'):
        log_of_sig_sqr = tf.log(sig_sqr/2)

    return tf.reduce_sum( G_value + log_of_sig_sqr - C, axis=0, name='exp_at_datapoints')


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

def get_scp_samples(rate_function, region_lims, upper_bound, res):
    
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
    sample_candidates_training = np.random.uniform(low=low, high=high, size=(J, D))
    sample_candidates_test = np.random.uniform(low=low, high=high, size=(J, D))
    
    #grid for plot
    if(region_lims.shape[0]>1):
        

        xx, yy = np.meshgrid(np.linspace(low[0], high[0], res), np.linspace(low[1], high[1], res))
        X = np.array([xx, yy]).transpose(1,2,0).reshape(res**2, 2)

        sample_points = np.concatenate((np.concatenate((sample_candidates_training,sample_candidates_test)),X))
    else:
        sample_points = np.concatenate((sample_candidates_training,sample_candidates_test))
        X = []
    
    vals = rate_function(sample_points)
    
    # 5. iterate over points and accept/reject
    R = np.random.uniform(size=J*2) * upper_bound
    accept_training = R[:J] < vals[:J] # R < logistic(vals) * upper_bound
    accept_test = R[J:(J+J)] < vals[J:(J+J)] 

    
    return sample_candidates_training[accept_training],sample_candidates_test[accept_test], R, xx,yy, vals[(J+J):]