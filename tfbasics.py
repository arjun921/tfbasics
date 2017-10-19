import tensorflow as tf

a = tf.constant(5)
b = tf.constant(6)
print(type(a))
print(type(b))
result = tf.multiply(a,b)
print(result)

with tf.Session() as sess:
	print(sess.run(result))
