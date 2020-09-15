import tensorflow as tf

def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):

	# define representation

	gaussian_params = tf.matmul(h1,w) + b

	# free variable
	mean = gaussian_params[:, :n_output]
	# positive standard deviation
	stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

	return mean, stddev

def variational_MLP_autoencoder(x_hat, x, dim_z, n_hidden, keep_prob):

	# encoding
	mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)
	# sampling by reparameterization
	z = mu + sigma + tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)