My own neural network implementation in Python. Does backpropagation. GUI support.

About the current version:

Initialize a neural network by calling:

n = NN([3,8,7,2,1], act='sigmoid', c=0.03, bias=1)

The 0th value of the list is the number of inputs, the 1st value is the 1st hidden layer, etc.

act is the activation function. Choices include:
 - 'step' (stepwise function)
 - 'rect' (rectilinear function)
 - 'softplus' (smooth approx. to rectilinear)
 - 'tanh' (hyperbolic function)
 - 'sigmoidSimple' (piecewise approximation of sigmoid for small gains in efficiency)
 - 'sigmoid' (default sigmoidal function).
 
c is the learning constant. Set it higher to learn faster, lower to learn slower but converge more easily.
 
bias is the input clamp on the bias nodes. Set it to 0 and the bias nodes will not activate.

To feedforward inputs through a network, call:

n.feedForward(inputs, brk=False)

with the input vector as the first parameter. For the first example, NN([3,8,7,2,1]) will take a vector of length 3 and return a vector of length 1.

if brk=True, the activation through every layer will be captured, not just the output. This is very useful when you want to record all the activations for backpropagation.

Want a single perceptron? Just intialize:

n = NN([3,1])

This will create a perceptron with three inputs.

Backpropagation accepts an array of input vectors and an array of target value vectors. So, if you want to train a perceptron on the XOR table, you set up the data up like this:

n.bp( [[0,1],[1,0],[1,1],[0,0]] , [[1],[1],[0],[0]], batch=True)

Call bp every time you want a training pass through your data to occur.

batch=True sets the learning mode to batch. If batch=False, stochastic learning will happen.

The last feature is the GUI. In the console, just type GUI(). Most of it is self-explanatory. When you upload a CSV to do backpropagation, make sure you use one row for each input-output pair, and two columns, the first for inputs and second for outputs. There is sample training data (Iris data set) in the folder.

You can import and export weights. The CSV formatting for import and export is the same, so you can export your weights and import them again without any problem. If you are importing a set from hand, make sure you know what you are doing. The number of layers, number of nodes in each layer, and number of weights in each node must exactly match the current network topology.

I originally had an evolutionary algorithm set up to train the weights, but that is now legacy code.

Up-and-coming additions:
 - GUI will get a tic-tac-toe window. Play tic-tac-toe against a neural network, game tree or another player.
 - Bug fixes, error handling.
 - Code is going to get optimized, documented, functions, get and set, methods get a consistent naming convention (e.g. everything gets capitalized so myFunc becomes MyFunc)
 - An executable file on sourceforge so you don't have to compile everything yourself.
 - A YouTube video for a run-through of the features.
 - And that's it. No new features will be added. I will move onto writing a self-organizing map or a cognitive robotics project, but those will be different projects.
