from Perceptron import NN

import random

from tkinter import *
from tkinter.ttk import *

class MainWindow(Frame):

    def __init__(self, parent, geometry, n=NN([2,5,8,7,4,3,3]),width=720,height=480):
        Frame.__init__(self, parent)

        self.parent = parent
        self.geometry=geometry
        self.n = n
        
        self.width=width
        self.height=height
        
        self.initUI()

    def initUI(self):

        print(X)

        titleTop = self.parent.title("Neural networks")
        
        '''
        leftFrame=Frame(self)
        leftFrame.pack(side="left", fill=Y, expand=True)
        canvasFrame=Frame(self)
        canvasFrame.pack(side="right",fill=X,expand=True)
        bottomFrame=Frame(self)
        bottomFrame.pack(side="bottom",fill=Y,expand=True)
        
        lbl1 = Label(leftFrame, text="")
        lbl1.pack(side="left", fill=Y, expand=True)
        
        lbl2 = Label(canvasFrame, text="")
        lbl2.pack(side="right",fill=X,expand=True)
        
        lbl3 = Label(bottomFrame, text="")
        lbl3.pack(side="bottom",fill=Y,expand=True)
        '''
        
        inputEntry = Entry(self.parent, text="Inputs")
        inputEntry.grid(row=0,column=0,padx=5,pady=5)
        
        leftButton = Button(self.parent, text="Feedforward")
        leftButton.grid(row=1,column=0,pady=5)
        
        load = Button(self.parent, text="Load weights")
        load.grid(row=2,column=0,pady=5)
        
        export = Button(self.parent, text="Export weights")
        export.grid(row=3,column=0,pady=5)
        
        canvas = Canvas(self.parent,width=720,height=480,relief="sunken",borderwidth=1)
        canvas.grid(row=0,column=1,rowspan=4,columnspan=15)
        self.setGraphics(canvas)
        
        bottomButton = Button(self.parent, text="Reset")
        bottomButton.grid(row=4,column=1,pady=15)
        
        
        layerOptions=["","0","1","2","3","4","5","6","7","8","9","10","11","12"]
        
        for i in range(10):
            var = StringVar(self)
            var.set(self.n.l[i] if i<len(self.n.l) else layerOptions[1])
            OptionMenu(self.parent, var, *layerOptions).grid(row=4,column=2+i)

    def setGraphics(self, canvas):
        x = canvas.winfo_width()+int(self.width/2)
        y = canvas.winfo_height()+int(self.height/2)
        
        margin = 40
        radius = 15
        
        values = []
        for i in range(len(self.n.l)):
            values.append([])
        
        layerDist = distances(self.n.l,(0 if len(self.n.l)%2==1 else 0.5))
        
        for i in range(len(layerDist)):
            layerDist[i] = layerDist[i]*2
            
        print(layerDist)
        
        for i in range(len(self.n.l)):
            networkDist=[]
            networkDist.extend(0 for k in range(0 ,self.n.l[i]))
            networkDist=distances(networkDist,(0 if self.n.l[i]%2==1 else 0.5))
            print(networkDist)
            for j in range(0, self.n.l[i]):
                values[i].append(canvas.create_circle(canvas,
                                                      x+margin*layerDist[i],
                                                      y+margin*(networkDist)[j],
                                                      radius,
                                                      outline="black",
                                                      fill=(sigmoidToColor(random.random()) if i != 0 else ""),
                                                      width=1))
        
        print(values)
        
        for i in range(len(values)-1):
            for j in range(len(values[i])):
                for k in range(len(values[i+1])):
                    # canvas.create_text(values[i][j].x,values[i][j].y,text="0.0")
                    canvas.create_line(values[i][j].x+values[i][j].r,values[i][j].y,values[i+1][k].x-values[i+1][k].r,values[i+1][k].y)
        
        for i in range(len(values[-1])):
            canvas.create_line(values[-1][i].x+values[-1][i].r,values[-1][i].y,values[-1][i].x+(values[-1][i].r*2),values[-1][i].y)
            
        def updateGraphics(self):
            print("Updates the activation labels and colors of the weights")
            
        def resetGraphics(self):
            print("Deletes everything on the canvas and reinitializes it")

def sigmoidToColor(w): # maps numbers in range [0,1] to colors along blue-red spectrum
    print(w)
    if w > 0.5:
        rgb=(255,int(255*(1-w)),int(255*(1-w)))
        return '#%02x%02x%02x' % rgb
    elif w <= 0.5:
        rgb=(int(255*w),int(255*w),255)
        return '#%02x%02x%02x' % rgb
        
def distances(l,offset=0): # l = [ l_1, l_2, l_3, ... l_n ]
    if len(l) == 2:
        return [-1+offset,1-offset]
    elif len(l) == 1:
        return [0]
    else:
        return [-(len(l)//2)+offset] + distances(l[1:-1],offset) + [len(l)//2-offset] # return [ len(l), len(l)-1, len(l)-2, ..., len(l)-2, len(l)-1, len(l) ]
        
        
class create_circle(object):
    def __init__(self,canvas,x,y,r,**kwargs):
        self.canvas = canvas
        self.x=x
        self.y=y
        self.r=r

        x1=x+r
        y1=y+r
        x2=x-r
        y2=y-r
    
        circle = Canvas.create_oval(self.canvas,x1,y1,x2,y2,**kwargs)
        
    def __repr__(self):
        return "circle (" + str(self.x) + "," + str(self.y) + ") with radius " + str(self.r)

'''
def create_circle(self,x,y,r,**kwargs):
    x1=x+r
    y1=y+r
    x2=x-r
    y2=y-r
    
    return Canvas.create_oval(self,x1,y1,x2,y2,**kwargs)
'''

Canvas.create_circle = create_circle

def main():

    root = Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
