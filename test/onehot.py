import tensorflow as tf

# one-hot 
tf_sess = tf.Session()

target = [0, 2, -1, 1, 3, 2, 13]
target = tf.one_hot(target, 15, 5, 0)

print(tf_sess.run(target))
#[[1 0 0 0 0 0]
# [0 0 1 0 0 0]
# [0 0 0 0 0 0]
# [0 1 0 0 0 0]
# [0 0 0 1 0 0]]

#[[1 0 0 0 0]
# [0 0 1 0 0]
# [0 0 0 0 0]
# [0 1 0 0 0]
# [0 0 0 1 0]
# [0 0 1 0 0]]
