import tensorflow as tf

a = tf.constant(5)
b = tf.constant(6)
print(type(a))
print(type(b))
result = tf.multiply(a,b)
print(result)

sess = tf.Session()
print(sess.run(result))
