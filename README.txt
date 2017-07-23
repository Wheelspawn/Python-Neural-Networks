My own neural network implementation in Python. My goal is to implement a standard multilayer perceptron in Python, along with a wide palette of variations and learning algorithms. I have gotten far enough to successfully train an MLP to solve some tough nonlinear classification problems with reinforcement learning.

What has been accomplished so far:
 - Implemented a perceptron and trained it to do linear classification
 - Fully capable multilayer perceptron with support for arbitrary hidden layers
 - Working reinforcement (evolutionary) learning algorithm.
 - In evolutionary scheme, was able to train forementioned MLP to do nonlinear classification. Successes have included:
 	- Classifying random points above or below degree 2 polynomials
	- Classifying random points inside or outside piecewise function that defines a circle
	In both of these cases, error was lower than 0.01.
 - Backpropagation is working for the final layer, but still has to be configured for the posterior layers.
 - Functioning GUI lets you initialize neural network, feedforward, and adjust weights.

About the current version:

Initialize a neural network by calling:

n = NeuralNetwork([3,8,7,2,1], act='sigmoid', bias=1)

The 0th value is the input, the 1st value is the 1st hidden layer, etc.

act is the activation function. Choices include:
 - 'step' (stepwise function)
 - 'rect' (rectilinear function)
 - 'softplus' (smooth approx. to rectilinear)
 - 'tanh' (hyperbolic function)
 - 'sigmoidSimple' (piecewise approximation of sigmoid for small gains in efficiency)
 - 'sigmoid' (default sigmoidal function).
 
bias is the input clamp on the bias nodes. Set it to 0 and the bias nodes will not activate.

To feedforward inputs through a network, call its feedForward() method with the input vector as the first parameter. if brk=True, the activation through every layer will be captured, not just the output. This is very useful if you want to record all the activations for backpropagation.

Want a single perceptron? Just intialize:

n = NeuralNetwork(3,1)

This will create a perceptron with three inputs.

Backpropagation accepts an array of input vectors and an array of target value vectors. So, if you want to train a perceptron on the XOR table, you set up the data up like this:

n.bp( [[0,1],[1,0],[1,1],[0,0]] , [[1],[1],[0],[0]] )

Up-and-coming additions:
 - A better, easier to use evolutionary algorithm. Better parameter control.
 - Finished tic-tac-toe demo. Users will be able to play TTT against a trained MLP.
 - A demo for backpropagation. Possibly something like a logic table.

Additions for the far future:
 - Support for neural network variants:
	- Recurrent
	- LSTM
	- Convolutional
	- Self-organizing map
	- Spiking
 - More learning algorithms:
	- Self-organized learning
	- Adaptive resonance theory
	- STDP, BCM and Hebbian learning
 - Modularity. Initialize multiple neural networks and connect them together.
 - I/O support. Get input from an Arduino microcontroller or camera. Send output to an Arduino microcontroller.
 - Deep learning support
