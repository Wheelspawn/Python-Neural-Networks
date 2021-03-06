
from Perceptron import NN

import math
import random

from tkinter import *
from tkinter.ttk import *

from HDFFuncs import *

class MainWindow(Frame):

    def __init__(self, parent, geometry=None, n=NN([4,9,8,3], act='sigmoid'), menus=[]):
        Frame.__init__(self, parent)

        self.parent = parent
        self.geometry = geometry # geometry object hold information about the graph representatoin of the neural network, including the canvas projecting it.
        self.n = n # the neural network model the window is currently initialized to use.
        self.menus=menus # optionmenus that specify the number of layers and the neurons each one contains
        
        self.initUI()
        
    def initButtons(self):
        titleTop = self.parent.title("Marionette 1.0.0")
        
        inputEntry = Entry(self.parent) # input values
        inputEntry.grid(row=0,column=0,padx=5,pady=5)
        inputEntry.insert(0, "Inputs")
        
        leftButton = Button(self.parent, text="Feedforward", command=lambda: propagate(self, self.geometry.canvas, inputEntry.get())) # feedforward button
        leftButton.grid(row=1,column=0,pady=5)
        
        backPropagation = Button(self.parent, text="Backpropagation", command=lambda: BackpropWindow(Tk(), self.n)) # button that lets you set up backprop training
        backPropagation.grid(row=2,column=0,padx=5,pady=5)
        
        load = Button(self.parent, text="Load weights from HDF", command=lambda: self.loadWeights()) # button that loads weights from HDF .h5 file
        load.grid(row=3,column=0,padx=5,pady=5)
        
        export = Button(self.parent, text="Export weights to HDF", command=lambda: self.exportWeights()) # button that exports weights to HDF .h5 file
        export.grid(row=4,column=0,pady=5)
        
        process = Button(self.parent, text="Process data and export to CSV", command=lambda: ProcessWindow(Tk(), self.n)) # button that exports weights to HDF .h5 file
        process.grid(row=5,column=0,padx=10,pady=5)
        
        # recurrencies = Button(self.parent, text="Recurrencies")
        # recurrencies.grid(row=6,column=0,pady=5)
        
        # ticTacToe = Button(self.parent, text="Play tic-tac-toe!")
        # ticTacToe.grid(row=7,column=0,pady=5)
        
        layerOptions=["","0","1","2","3","4","5","6","7","8","9","10","11","12"] # max number of neurons in a layer is 12
        
        for i in range(9): # here we initalize eight optionmenus and give them a callback function for updating the layer information
            var = StringVar(self)
            var.set(self.n.l[i] if i<len(self.n.l) else layerOptions[1])
            OptionMenu(self.parent, var, *layerOptions).grid(row=9,column=2+i,pady=15)
            self.menus.append(var)
        
        var2 = StringVar(self)
        var2.set("sigmoid")
        
        actFunction = OptionMenu(self.parent, var2, "", "sigmoid", "tanh", "step", "rect", "softplus", "linear", command=lambda x: chooseActFunction(self.n, var2.get())) # optionmenu to set the activation function
        actFunction.grid(row=9,column=12)
        
        bottomButton = Button(self.parent, text="Reinitialize", command=self.reInit) # reinitializes network based on parameters given by user (or by default)
        bottomButton.grid(row=9,column=13,pady=15)

    def initUI(self): # create the user interface
    
        self.initButtons() # create buttons in a separate method
        
        self.geometry = NetworkGraphic() # this class will hold variables to represent the canvas
                                         # and graphical objects that make up the neural network such as circles, lines and labels
        self.geometry.canvas = Canvas(self.parent,width=self.geometry.width,height=self.geometry.height,relief="sunken",borderwidth=1) # initalize with parameters inherited from self
        self.geometry.canvas.grid(row=0,column=1,rowspan=8,columnspan=15)
        self.setGraphics(self.geometry.canvas,self.geometry.values,self.geometry.lines,self.geometry.labels,self.geometry.acts) # initialize graphics
        
        # print(self.geometry.values)

    def setGraphics(self, canvas, values, lines, labels, acts):
        
        x = canvas.winfo_width()+int(780/2) # (x,y) is the center of the canvas
        y = canvas.winfo_height()+int(480/2)
        
        margin = 40 # margin for neurons
        radius = 15 # radius for neurons
        
        values = [[] for x in range(len(self.n.l[:])) ] # initialize with empty arrays that match neural network dimensions, to be filled later
        labels = [[] for x in range(len(self.n.l[:])) ]
        lines = []
        
        layerDist = distances(self.n.l,(0 if len(self.n.l)%2==1 else 0.5)) # get the distances between layers
        
        for i in range(len(layerDist)):
            layerDist[i] = layerDist[i]*2 # multiply each constant by two-too many lines of code for this problem, to be fixed later
        
        for i in range(len(self.n.l)):
            networkDist=[]
            networkDist.extend(0 for k in range(0 ,self.n.l[i]))
            networkDist=distances(networkDist,(0 if self.n.l[i]%2==1 else 0.5)) # same thing as layerDist, but for the neurons in each layer
            for j in range(0, self.n.l[i]):
                values[i].append(canvas.create_circle(canvas, # add circle objects
                                                      x+margin*layerDist[i],
                                                      y+margin*networkDist[j],
                                                      radius,
                                                      outline="black",
                                                      fill=(("light gray") if (acts == [] or i==0) else hexConverter(acts[i][j], self.n.act)), # "light gray"
                                                      width=1))
                labels[i].append(canvas.create_text(x+margin*layerDist[i],y+margin*networkDist[j],text="" if acts == [] else round(acts[i][j],1))) # add labels for activations
        
        
        for i in range(len(values)-1):
            for j in range(len(values[i])):
                for k in range(len(values[i+1])):
                    # canvas.create_text(values[i][j].x,values[i][j].y,text="0.0")
                    lines.append(canvas.create_line((values[i][j].x)+values[i][j].r,values[i][j].y,(values[i+1][k].x)-values[i+1][k].r,values[i+1][k].y))
                    
        labels.append([])
        
        for i in range(len(values[-1])):
            lines.append(canvas.create_line((values[-1][i].x)+(values[-1][i].r*2)-10,values[-1][i].y+3,(values[-1][i].x)+(values[-1][i].r*2),values[-1][i].y)) # left arrowhead.
            lines.append(canvas.create_line((values[-1][i].x)+values[-1][i].r,values[-1][i].y,(values[-1][i].x)+(values[-1][i].r*2),values[-1][i].y))
            lines.append(canvas.create_line((values[-1][i].x)+(values[-1][i].r*2)-10,values[-1][i].y-3,(values[-1][i].x)+(values[-1][i].r*2),values[-1][i].y)) # right arrowhead

            labels[-1].append(canvas.create_text((values[-1][i].x)+(values[-1][i].r*2)+radius,values[-1][i].y,text="" if acts == [] else round(acts[-1][i],1)))
            
        self.geometry.values = values
        self.geometry.labels = labels
        self.geometry.lines = lines
        
        self.popup = Menu(self.parent, tearoff=0)
        canvas.bind("<Button-3>", self.selector)
        canvas.bind("<Button-1>", self.selector)
        
    def selector(self, e):
        
        for i in range(len(self.geometry.values)):
            for j in range(len(self.geometry.values[i])):
                dist = math.sqrt( ( e.x-self.geometry.values[i][j].x )**2 + ( e.y-self.geometry.values[i][j].y )**2 )
                if dist <= 15 and i != 0:
                    # self.popup.tk_popup(e.x_root,e.y_root,0)
                    # self.popup.add_command(label="Edit weights", command=self.initWeightWindow(Tk(), self.n.neuronWeights(i-1,j)))
                    self.initWeightWindow(Tk(), self.n.neuronWeights(i-1,j))
                    break
                    
    def initWeightWindow(self,parent,neuron):
        WeightWindow(parent,neuron)
    
    def updateGraphics(self):
        self.resetGraphics()
        self.geometry.canvas = Canvas(self.parent,width=self.geometry.width,height=self.geometry.height,relief="sunken",borderwidth=1)
        self.geometry.canvas.grid(row=0,column=1,rowspan=8,columnspan=15)
        self.setGraphics(self.geometry.canvas,self.geometry.values,self.geometry.lines,self.geometry.labels,self.geometry.acts)
            
    def resetGraphics(self):
        self.geometry.canvas.delete("all")
        
    def exportWeights(self):
        from tkinter import filedialog
        g = filedialog.asksaveasfilename()
        weightsToHDF(self.n.w, str(g))
        
    def loadWeights(self):
        from tkinter import filedialog
        g = filedialog.askopenfilename()
        self.n.setWeights(hdfToWeights(g))
        self.updateGraphics()
        
    def reInit(self):
        
        screengrab=[]
        
        for var in self.menus:
            screengrab.append(int(var.get()))
        
        screengrab = list(filter(lambda a: a != 0, screengrab))
        
        self.n.l = screengrab
        
        self.n.initWeights()
        self.geometry.acts=[]
        self.updateGraphics()
        
class ProcessWindow(Frame):
    def __init__(self, parent, network, data=None, processedData=None):
        Frame.__init__(self, parent)

        self.parent = parent
        self.network = network
        
        self.initUI()
        
    def initUI(self):
        title = self.parent.title("Process data")
        
        uploadInputs = Button(self.parent, text="Upload CSV input pairs", command=lambda: self.upload())
        uploadInputs.grid(row=0,column=0,padx=15,pady=15)
        
        processButton = Button(self.parent, text="Process data", command=lambda: self.process())
        processButton.grid(row=1,column=0,padx=15,pady=15)
        
        exportCSV = Button(self.parent, text="Export to CSV", command=lambda: self.export())
        exportCSV.grid(row=2,column=0,padx=15,pady=15)
        
    def upload(self):
        from tkinter import filedialog
        f = filedialog.askopenfilename()
        self.data = vectorsToArray(f)
        print(self.data)
        pass
    
    def process(self):
        self.processedData=[]
        for entry in self.data:
            self.processedData.append(self.network.feedForward(entry))
            for i in range(len(self.processedData[-1])):
                self.processedData[-1][i] = round(self.processedData[-1][i],2)
        pass
    
    def export(self):
        from tkinter import filedialog
        g = filedialog.asksaveasfilename()
        weightsToHDF(self.processedData, str(g))
        
class WeightWindow(Frame):
    def __init__(self, parent, neuron, entries=[], labels=[]):
        Frame.__init__(self, parent)

        self.parent = parent
        self.neuron = neuron
        self.entries = entries[:]
        self.labels = labels[:]
        
        self.initUI()
        
    def initUI(self):
        title = self.parent.title("Edit weights")        
        
        for i in range(0,len(self.neuron)):
            self.entries.append(Entry(self.parent))
            self.entries[i].insert(0, str(self.neuron[i]))
            self.entries[i].grid(row=i,column=0,padx=15,pady=5)
            
            self.labels.append(Label(self.parent, text=(("Weight " + str(i)) if i != len(self.neuron)-1 else ("Weight " + str(i) + " (bias)"))))
            self.labels[i].grid(row=i,column=1,padx=15,pady=1)
        
        leftButton = Button(self.parent, text="Update weights", command=self.updateWeights)
        leftButton.grid(row=len(self.neuron),column=0,padx=15,pady=15)
        
        
    def updateWeights(self):
        for i in range(len(self.entries)):
            self.neuron[i] = self.entries[i].get()
            
class BackpropWindow(Frame):
    def __init__(self, parent, network, inputs=None, targets=None, epochNum=1, noisy=False):
        Frame.__init__(self, parent)

        self.parent = parent
        self.network = network
        self.noisy = noisy
        
        self.initUI()
        
    def initUI(self):
        titleTop = self.parent.title("Backpropagation")
    
        uploadInputs = Button(self.parent, text="Upload HDF input pairs", command=lambda: self.loadInputs())
        uploadInputs.grid(row=0,column=0,padx=15,pady=15)
        
        uploadTargets = Button(self.parent, text="Upload HDF target pairs", command=lambda: self.loadTargets())
        uploadTargets.grid(row=1,column=0,padx=15,pady=15)
        
        epochs = Entry(self.parent) # input values
        epochs.grid(row=2,column=0,padx=5,pady=5)
        epochs.insert(0, "Number of epochs")
        
        self.learnMode = IntVar()
        self.learnMode.set(1)
        
        batch = Radiobutton(self.parent, text="Batch", variable=self.learnMode, value=1, command=lambda: self.learnMode.set(1))
        batch.grid(row=0,column=1,padx=5,pady=5)
        batch.invoke()

        stochastic = Radiobutton(self.parent, text="Stochastic", variable=self.learnMode, value=0, command=lambda: self.learnMode.set(0))
        stochastic.grid(row=1,column=1,padx=5,pady=5)
        
        noiseButton = Checkbutton(self.parent, text="Input noise", command=lambda:self.inputNoise())
        noiseButton.grid(row=2,column=1,padx=5,pady=5)
        
        backProp = Button(self.parent, text="Train", command=lambda:self.backProp(epochs.get()))
        backProp.grid(row=3,column=0,padx=15,pady=15)
        
    def inputNoise(self):
        self.noisy=not(self.noisy)
        
    def loadInputs(self):
        from tkinter import filedialog
        f = filedialog.askopenfilename()
        self.inputs = vectorsToArray(f)
        
    def loadTargets(self):
        from tkinter import filedialog
        g = filedialog.askopenfilename()
        self.targets = vectorsToArray(g)
        
    def backProp(self, epochNum):

        print(self.learnMode.get())
        
        if self.inputs != None and self.targets != None:
            if int(epochNum) > 0:
                inputCopy = self.inputs[:]
                
                for i in range(int(epochNum)):
                    
                    if self.noisy==True:
                        for j in range(len(inputCopy)):
                            for k in range(len(inputCopy[j])):
                                inputCopy[j][k] += round(random.gauss(0,0.01),4)
                                
                    print(i)
                                
                    self.network.bp(inputCopy, self.targets, bool(self.learnMode.get()))
                print("Done")
            
def propagate(frame, canvas, entry):
    from tkinter import messagebox
    try:
        entry = entry.split(",")
        entry = list(map(float, entry))
        if len(entry) == frame.n.l[0]:
            frame.geometry.acts = frame.n.feedForward(entry,brk=True)
            frame.updateGraphics()
        else:
            raise ValueError
    except ValueError:
        messagebox.showerror("", "Input vector has wrong size and/or non-numerical inputs.")
        
def chooseActFunction(n, v):
    if n.act == 'step' or 'sigmoid' or 'tanh':
        n.act=v
            
class NetworkGraphic(object):

    def __init__(self,canvas=None,values=[],lines=[],labels=[],acts=[],width=780,height=480,margin=40,radius=15):
        
        self.canvas = canvas
        self.values = values
        self.lines = lines
        self.labels = labels
        self.acts = acts
        self.width = width
        self.height = height
        self.margin = margin
        self.radius = radius
            
            
def sigmoidToHex(a): # maps numbers in range [0,1] to colors along blue-red spectrum
    if a >= 0.5:
        rgb=(255,int(255*-2*(a-1)),int(255*-2*(a-1)))
    else:
        rgb=(int(255*(a*2)),int(255*(a*2)),255)
    return '#%02x%02x%02x' % rgb
    
def stepToHex(a):
    if a >= 0.5:
        return '#ff0000'
    else:
        return '#0000ff'
    
def tanhToHex(a): # maps numbers in range [-1,1] to colors along blue-red spectrum
    if a >= 0.0:
        rgb=(255,int(255*(1-a)),int(255*(1-a)))
    else:
        rgb=(int(255*(1-abs(a))),int(255*(1-abs(a))),255)
    return '#%02x%02x%02x' % rgb
    
def hexConverter(a, func):
    if func == 'sigmoid':
        return sigmoidToHex(a)
    elif func == 'step':
        return stepToHex(a)
    elif func == 'tanh':
        return tanhToHex(a)
    elif (func == 'rect') or (func == 'softplus'):
        return sigmoidToHex(boundRect(a))
    elif func == 'linear':
        return sigmoidToHex(sigmoid(a))

# turns rectilinear and softmax values >= 0 into a nice number between 0 and 1
def boundRect(a):
    if a <= 0:
        return 0
    else:
        return -1/(np.sqrt(a)+1)+1

def distances(l,offset=0): # l = [ l_1, l_2, l_3, ... l_n ]
    if len(l) == 2:
        return [-1+offset,1-offset]
    elif len(l) == 1:
        return [0]
    else:
        return [-(len(l)//2)+offset] + distances(l[1:-1],offset) + [len(l)//2-offset] # return [ len(l), len(l)-1, len(l)-2, ..., len(l)-2, len(l)-1, len(l) ]
        
        
class create_circle(object):
    def __init__(self,canvas,x,y,r,outline,fill,width,**kwargs): # **kwargs
        self.canvas = canvas
        self.x = x
        self.y = y
        self.r = r
        self.outline = outline
        self.fill = fill
        self.width = width

        x1=self.x+self.r
        y1=self.y+self.r
        x2=self.x-self.r
        y2=self.y-self.r
    
        circle = Canvas.create_oval(self.canvas,x1,y1,x2,y2,outline=self.outline,fill=self.fill,width=self.width,**kwargs)
        
    def __repr__(self):
        return "circle (" + str(self.x) + "," + str(self.y) + ") with radius " + str(self.r)


Canvas.create_circle = create_circle

def main():

    root = Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
