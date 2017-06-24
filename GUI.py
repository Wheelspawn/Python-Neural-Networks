from Perceptron import NN

import random

from tkinter import *
from tkinter.ttk import *

class MainWindow(Frame):

    def __init__(self, parent, geometry=None, n=NN([4,9,8,1]), menus=[]):
        Frame.__init__(self, parent)

        self.parent = parent
        self.geometry = geometry # geometry object hold information about the graph representatoin of the neural network, including the canvas projecting it.
        self.n = n # the neural network model the window is currently initialized to use.
        self.menus=menus # optionmenus that specify the number of layers and the neurons each one contains
        
        self.initUI()
        
    def initButtons(self):
        titleTop = self.parent.title("Artificial neural network simulator")
        
        inputEntry = Entry(self.parent) # input values
        inputEntry.grid(row=0,column=0,padx=5,pady=5)
        
        leftButton = Button(self.parent, text="Feedforward", command=lambda: propagate(self, self.geometry.canvas, inputEntry.get())) # feedforward button
        leftButton.grid(row=1,column=0,pady=5)
        
        load = Button(self.parent, text="Load weights from CSV") # button that loads weights from (an undecided extension) file
        load.grid(row=2,column=0,padx=5,pady=5)
        
        export = Button(self.parent, text="Export weights to CSV") # button that exports weights to (an undecided extension) file
        export.grid(row=3,column=0,pady=5)
        
        layerOptions=["","0","1","2","3","4","5","6","7","8","9","10","11","12"] # max number of neurons in a layer is 12
        
        for i in range(9): # here we initalize eight optionmenus and give them a callback function for updating the layer information
            var = StringVar(self)
            var.set(self.n.l[i] if i<len(self.n.l) else layerOptions[1])
            OptionMenu(self.parent, var, *layerOptions).grid(row=5,column=2+i,pady=15)
            self.menus.append(var)
        
        var = StringVar(self)
        var.set("sigmoid")
        
        actFunction = OptionMenu(self.parent, var, "sigmoidal", "tanh", "rectilinear") # optionmenu to set the activation function
        actFunction['menu'].config(bg="white") # this does nothing at the moment
        actFunction.grid(row=5,column=12)
        
        bottomButton = Button(self.parent, text="Reinitialize", command=self.reInit) # reinitializes network based on parameters given by user (or by default)
        bottomButton.grid(row=5,column=13,pady=15)

    def initUI(self): # create the user interface
    
        self.initButtons() # create buttons in a separate method
        
        self.geometry = NetworkGraphic() # this class will hold variables to represent the canvas
                                         # and graphical objects that make up the neural network such as circles, lines and labels
        self.geometry.canvas = Canvas(self.parent,width=self.geometry.width,height=self.geometry.height,relief="sunken",borderwidth=1) # initalize with parameters inherited from self
        self.geometry.canvas.grid(row=0,column=1,rowspan=4,columnspan=15)
        self.setGraphics(self.geometry.canvas,self.geometry.values,self.geometry.lines,self.geometry.labels,self.geometry.acts) # initialize graphics

    def setGraphics(self, canvas, values, lines, labels, acts):
        
        x = canvas.winfo_width()+int(780/2) # (x,y) is the center of the canvas
        y = canvas.winfo_height()+int(480/2)
        
        margin = 40 # margin for neurons
        radius = 15 # radius for neurons
        
        values = [[] for x in range(len(self.n.l[:])) ] # initialize with empty arrays that match neural network dimensions, to be filled later
        labels = [[] for x in range(len(self.n.l[:])) ]
        
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
                                                      fill=(("light gray") if (acts == [] or i==0) else sigmoidToHex(acts[i][j])), # "light gray"
                                                      width=1))
                labels[i].append(canvas.create_text(x+margin*layerDist[i],y+margin*networkDist[j],text="0.0" if acts == [] else round(acts[i][j],1))) # add labels for activations
        
        
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

            labels[-1].append(canvas.create_text((values[-1][i].x)+(values[-1][i].r*2)+radius,values[-1][i].y,text="0.0" if acts == [] else round(acts[-1][i],1)))
            
    def updateGraphics(self):
        self.resetGraphics()
        self.geometry.canvas = Canvas(self.parent,width=self.geometry.width,height=self.geometry.height,relief="sunken",borderwidth=1)
        self.geometry.canvas.grid(row=0,column=1,rowspan=4,columnspan=15)
        self.setGraphics(self.geometry.canvas,self.geometry.values,self.geometry.lines,self.geometry.labels,self.geometry.acts)
            
    def resetGraphics(self):
        self.geometry.canvas.delete("all")
        
    def reInit(self):
        
        screengrab=[]
        
        for var in self.menus:
            screengrab.append(int(var.get()))
        
        screengrab = list(filter(lambda a: a != 0, screengrab))
        
        self.n.l = screengrab
        
        self.n.initWeights()
        self.geometry.acts=[]
        self.updateGraphics()
            
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
            
            
def sigmoidToHex(w): # maps numbers in range [0,1] to colors along blue-red spectrum
    if w > 0.5:
        rgb=(255,int(255*-2*(w-1)),int(255*-2*(w-1)))
    else:
        rgb=(int(255*(w*2)),int(255*(w*2)),255)
    return '#%02x%02x%02x' % rgb
    
    '''
    if w > 0.5:
        rgb=(int(255*w),int(255*(1-w)),int(255*(1-w)))
        return '#%02x%02x%02x' % rgb
    elif w <= 0.5:
        rgb=(int(255*(w)),int(255*w),int(255*(1-w)))
        return '#%02x%02x%02x' % rgb
    '''
        
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
