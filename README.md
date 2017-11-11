# Structure-Tensorflow-Neural-Network
How to structure Tensorflow ANN using MNIST as an example

We are going to take a look at the basic structure of a tensorflow program for constructing a deep neural network.

We’ll consider the MNIST dataset with 60,000 training samples and 10,000 test samples. Each sample is a 28 x 28 image. The goal is to determine the digit in the image. We first read the dataset by one_hot encoding our labels. 

```
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
```

We create a directory to store paths to events in our computation graph. 

```
# Path to Computation graphs
LOGDIR = './graphs'
```

Now, Start a session (or do it later using “with ”, your choice).

```
# start session
sess = tf.Session()
```

We then chose appropriate values for our hyper parameters. 

```
# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 1000
EPOCHS = 10
```

We define the number of neurons per hidden layer in the network, and define dataset specific parameters (like the input and output size).

```
# Layers
HL_1 = 1000
HL_2 = 500

# Other Parameters
INPUT_SIZE = 28*28
N_CLASSES = 10
```

While programming in Tensorflow, it is often convenient to think of componets visually as we would observe in the computation graph. Every program can be thought of having:
- Inputs
- Layers
- Loss
- Optimizer
- Evaluation
- Training

Generally speaking, _Input_ data is passed through the _layers_ of our Neural Network. The corresponding cost or _loss_ is computed, that needs to be minimized by an _optimizer_. We can then _evaluate_ performance of _training_.

We begin by defining these sections with `tf.name_scope` and fill each out one at a time. In the case of MNIST dataset, for example, **Inputs** would be the set of images and labels. 
Such inputs are defined with placeholders, indicating a promised value.

```
with tf.name_scope('input'):
	images = tf.placeholder(tf.float32, [None, INPUT_SIZE] , name="images")
	labels = tf.placeholder(tf.float32, [None, N_CLASSES], name="labels")
```

For defining **layers**, we could define each hidden layer in its own scope manually. But its cleaner code to define a function, passing in the layer name. _fc_ stands for "fully connected". In each layer, we have a set of weights, which we initialized to random normally distributed values , and biases, initialized to -1. We compute `wx + b` and apply the `reLU` activation (rectified linear unit), flushing the negative values to 0. Note this argument is optional as we don’t apply `reLU` to the output layer. 

```
def fc_layer(x, layer, size_out, activation=None):
	with tf.name_scope(layer):
		size_in = int(x.shape[1])
		W = tf.Variable(tf.random_normal([size_in, size_out]) , name="weights") 
		b = tf.Variable(tf.constant(-1, dtype=tf.float32, shape=[size_out]), name="biases")

		wx_plus_b = tf.add(tf.matmul(x, W), b)
		if activation: 
			return activation(wx_plus_b)
		return wx_plus_b

```


Each layer is now defined, passing the input to the layer, the name of the layer, number of output nodes, and the optional reLU activation function. 

```
fc_1 = fc_layer(images, "fc_1",  HL_1, tf.nn.relu)
fc_2 = fc_layer(fc_1, "fc_2", HL_2, tf.nn.relu)
```

After the second layer, I construct a _dropout layer_ that randomly turns off neurons. This is done to enhance generalization as it forces the network to learn along new paths. It reduces the chance of overfitting the data.

```
dropped = tf.nn.dropout(fc_2, keep_prob=0.9)
y = fc_layer(dropped, "output", N_CLASSES)
```

In the **loss** section, we compute the cost, or inaccuracy of our neural network. By applying a softmax function to the output layer `y`, we get a set of values with a range 0 to 1, indicating the probability that the image is a specific digit. We then determine _cross entropy loss_ by comparing to the one hot encoded labels. 

`tf.summary.scalar` allows us to record changes in this value overtime and display it on the tensorboard.

```
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
	tf.summary.scalar('loss', loss)
```

In the **optimizer** section, we define the Adam Optimizer to minimize the loss function.  Adam combines the advantages of other modern optimizers like Adagrad and RMSProp and works for a wide varaiety of problems.

```
with tf.name_scope('optimizer'):
	train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
```

In the **evaluation** section, we determine the number of image samples predicted correctly. We’ll see how this improves over time by recording It on our tensorboard.

```
with tf.name_scope('evaluation'):
	correct = tf.equal( tf.argmax(y, 1), tf.argmax(labels, 1) )
	accuracy = tf.reduce_mean( tf.cast(correct, dtype=tf.float32) )
	tf.summary.scalar('accuracy', accuracy)
```

Now we define FileWriters which contain event writing mechanisms to log _events_ in their respective directories. Events in this case are step values of our scalars: loss and accuracy. FileWriter creates an empty directory and returns an event logger. So `train_writer` can write events to the train folder. While `test_writer` can write events to the test folder. We finish by merging them to a list of summaries.

```
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "test"), sess.graph)
summary_op = tf.summary.merge_all()
``` 

All variables like weights and biases are now initialized. 

```
init = tf.global_variables_initializer()
sess.run(init)
```

The  tensors defined until this point are powerful, but they only construct the network. And what good is a network without flow?  We execute the training in batches, defined by BATCH_SIZE. We then push the flow to train the batch samples using sess.run.  Since this is the result of training, I will log this event into the train directory, using the train_writer. This event will contain the “loss” computed for this batch.

```
summary_result, _ = sess.run( [summary_op, train], feed_dict={images: batch_xs, labels: batch_ys} )
train_writer.add_summary(summary_result, step)
```

We then determine performance against the entire test set. Since this is the result of testing, I will log this event into the test directory, using the test_writer. This event will contain the “accuracy” computed at this point.

```
summary_result, acc = sess.run( [summary_op, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels} )
test_writer.add_summary(summary_result, step)
```

We clean up by Closing the train and test writers and the session. 

```
train_writer.close()
test_writer.close()
sess.close()
```

And we’re done. 

## Usage 

Run the program, and fix any errors that occur. 

```
$ python mnist_code.py
```

We now run tensorboard, feeding in our “graph” directory.

```
$ tensorboard --logdir="graphs/"
```
Then open `localhost:6006` on a browser tab.

## The Tensorboard

In the graph section, we can view our computation graph, just like we described in our program. 

![Computation Graph](https://github.com/ajhalthor/Structure-Tensorflow-Neural-Network/blob/master/mics/screen3.png)


In the scalar section, we can monitor our loss and accuracy overtime. You can see that everythings running well as loss decreases over time and accuracy improves for the test set.

![Loss and Accuracy curves](https://github.com/ajhalthor/Structure-Tensorflow-Neural-Network/blob/master/mics/screen1.png)

For the current configuration, I can get upto a **96%** accuracy over 10 epochs. Play around with the hyper parameters to improve this number. NOTE that a typical feed forward neural net isn’t optimal for such a computer vision problem. You can use a convolutional net instead and get performance well over 99%. 

Regardless, I hope this gave you an insight on how to structure your Tensorflow code for DNNs and make use of tensorboard for visualization.

If you like the repository, check out my youtube video on [Code Emporium]() and subscribe on your way out for more amazing content! Cheers!

 

 


 


