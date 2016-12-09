My own neural network implementation in Python. My goal is to implement a standard multilayer perceptron in Python, along with a wide palette of variations and learning algorithms. I have gotten far enough to successfully train an MLP to solve some tough nonlinear classification problems with reinforcement learning.

What has been accomplished so far:
 - Implemented a perceptron and trained it to do linear classification
 - Fully capable multilayer perceptron with support for up to four hidden layers
 - Working reinforcement (evolutionary) learning algorithm.
 - In evolutionary scheme, was able to train forementioned MLP to do nonlinear classification. Successes have included:
 	- Classifying random points above or below degree 2 polynomials
	- Classifying random points inside or outside piecewise function that defines a circle
	In these cases, error was lower than 0.01.
	
About the versions:

Perceptron-1.1 to 1.6 deals with perceptrons only. From Perception-1.7 and onward, we may initialize a neural network by calling:

n = NeuralNetwork(3,8,7,2,1)

The first parameter is the number of neurons in the input layer. The next two parameters are the hidden layers. The fourth is the output. To add a bias, we use the fifth parameter.

To feedforward a neuron, we call its feedForward() method with an array of weights and inputs as parameters, respectively.
To feedforward an entire network, we call the feedForward() function with inputs and a neural network object as parameters.

Perception-2.0 and subsequent versions are rewrites that are going address several problems.

 - Instead of being hacked together for personal use only, I will be emphasizing neatness and reproducibility. Functions will have clear documentation and can be reused.
 - For matrix multiplication, I have switched to the numpy library for faster calculations. The problem of ambiguous input layers has been temporarily resolved. Users now have the choice of how many they want to add.
 - Types will be standardized. Everything a neural network gets as input or generates as output will be an array.
 - Bias nodes are confusing to use and will be changed.
 - The neural network class had to initialize an array of neuron objects. This was confusing and unnecessary. Now, each neural network will simply store a matrix for the weights of each layer. Each weight can be changed manually with set methods.

Now, initializing a neural network is simpler. For instance, to implement a two-hidden layer network, simply type:

n = NeuralNetwork(3,8,7,None,None,1,2)

We now have a neural network with 3 inputs and 1 output. There are two bias nodes on the hidden layers.

Want a single perceptron? Just intialize:

n = NeuralNetwork(3,1,None,None,None,None,None)

This will work, because n.feedForward() will just skip every layer that doesn't exist. feedForward() takes the same parameters, but a layer set to 'None' will force feedForward() to pass over that layer.

Also, feedForward() is no longer a global function but a method of the neural network class. Now, call n.feedForward() and just pass the input array as a parameter.

Up-and-coming additions for 2.0:
 - Backpropagation.
 - A better, easier to use evolutionary algorithm.
 - Finished tic-tac-toe demo. Users will be able to play TTT against a trained MLP.
 - A very simple GUI to demonstrate TTT for non-programmers.
 - A demo for backpropagation. Possibly something like a logic table.

3.0 and beyond:
 - Support for neural network variants:
	- Recurrent
	- Convolutional
	- Spiking
	- Hopfield
 - More learning algorithms:
	- Self-organized learning
	- Adaptive resonance theory
	- STDP and Hebbian learning
 - GUI support.
 - Modularity. Initialize multiple neural networks and connect them together.
 - I/O support. Get input from an Arduino microcontroller or camera. Send output to an Arduino microcontroller.
 - Addition of a computer vision library such as OpenCV
 - Addition of a neural network library such as Keras
 - Deep learning support
