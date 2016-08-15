import theano
import theano.tensor as T
import numpy as np

x = T.ivector('x')
m1 = T.ivector('m1')
m2 = T.ivector('m2')

def my_sum(x_i, s_i1, y_i1, m_1, m_2):
    s_i = m_1 * x_i + s_i1
    y_i = m_2 * x_i - y_i1
    return [s_i, y_i]

def my_sum_1(x_i, s_i1, y_i1, m_1, m_2, s, y):
    s_i = m_1 * x_i + s_i1 + theano.tensor.sum(s, axis=0)
    y_i = m_2 * x_i + y_i1 + theano.tensor.sum(y, axis=0)
    return [s_i, y_i]

[s, y], updates = theano.scan(my_sum, sequences=x, outputs_info=[dict(initial=T.zeros(4)), dict(initial=T.zeros(4))], non_sequences=[m1, m2])


#[s_f, y_f], updates = theano.scan(my_sum, sequences=x, outputs_info=[dict(initial=s[-1]), dict(initial=y[-1])], non_sequences=[s, y])

output = s[-1] + y[-1]

my_sum_built = theano.function([x, m1, m2], output)

#x_np = np.array([[1, 2, 3, 4],[1, 2, 3, 4]], dtype='int32') #[ 2.  4.  6.  8.]

x_np = np.array([1, 2, 3, 4], dtype='int32') #[ 8.  8.  8.  8.]

m1_np = np.ones(4, dtype='int32')
m2_np = -np.ones(4, dtype='int32')

print my_sum_built(x_np, m1_np, m2_np)