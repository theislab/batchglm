import tensorflow as tf
from tensorflow.contrib.distributions import NegativeBinomial


def negative_binomial(r, mu):
    # ugly hack using tensorflow, since parametrisation with `p`
    # does not work with np.random.negative_binomial
    
    retVal = None
    with tf.Graph().as_default() as g:
        r = tf.constant(r)
        mu = tf.constant(mu)
        
        p = mu / (r + mu)
        
        dist = NegativeBinomial(total_count=r, probs=p)
        
        # run sampling session
        with tf.Session() as sess:
            retVal = sess.run(tf.squeeze(
                dist.sample(1)
            ))
    
    # random_data = np.random.negative_binomial(
    #     self.r,
    #     self.p,
    # )
    # return random_data
    
    return retVal
