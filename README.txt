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

About the current version:

Initialize a neural network by calling:

n = NeuralNetwork([3,8,7,2,1])

The 0th value is the input, the 1st value is the first hidden layer, etc.

To feedforward a network, we call its feedForward() method with inputs as parameters. if brk=True, the activation through every layer will be captured, not just the output. This is very useful when you want all the activations for backpropagation.

Want a single perceptron? Just intialize:

n = NeuralNetwork(3,1)

This will create a perceptron with three inputs.

Up-and-coming additions for 3.0:
 - A better, easier to use evolutionary algorithm.
 - Finished tic-tac-toe demo. Users will be able to play TTT against a trained MLP.
 - A very simple GUI to demonstrate TTT for non-programmers and people who don't want to use the console.
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
 - Neural networks designed for robot control
 - Deep learning support
