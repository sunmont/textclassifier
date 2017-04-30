import tensorflow as tf
import numpy

# one-hot 
tf_sess = tf.Session()

target_ = [
          [0, 2, -1, 1, 3, 2, 13],
          [1, 3, 4, 5, 6, 5, 10]
         ]
print(target_)

print("unstack ...")
target = tf.unstack(target_, num=7, axis = 1)
lst_ = tf_sess.run(target) 
print(lst_)
print(numpy.array(lst_).shape)

print("stack ...")
target = tf.stack(lst_)
lst_stack_ = tf_sess.run(target)
print(lst_stack_)
print(numpy.array(lst_stack_).shape)



#[array([0, 1], dtype=int32), array([2, 3], dtype=int32), array([-1,  4], dtype=int32), array([1, 5], dtype=int32), array([3, 6], dtype=int32), array([2, 5], dtype=int32), array([13, 10], dtype=int32)]
