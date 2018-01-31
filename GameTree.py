
#
# Author: Nathaniel Le Sage 6/4/2017
#

class Node(object):
    def __init__(self,   element=None, # the board layout
                         parent=None,
                         children=[],
                         hasChildren=False,
                         numChildren=0,
                         depth=0,
                         util=[0,0,0,0,0,0,0,0,0]): # utility of the next move
        self.element = element # generic value associated with the node
        self.parent = parent
        self.children = children
        self.hasChildren = hasChildren
        self.numChildren = numChildren
        self.depth = depth
        self.util=[0,0,0,0,0,0,0,0,0]

    def setChild(self, c): # adds a child
        if type(c) == Node:
            if self.children == []: # stops new object from initializing with child array from parent
                reset = []       # if you don't have this logic, an infinite loop occurs
                self.children = reset
            self.children.append(c)
            c.depth = self.depth + 1
            self.updateNumChildren(1)
            if c.parent == None:
                c.parent = self
            else:
                print("node already has parent!")
            self.hasChildren = True
        else:
            print("Not a node!")
            
    def Children(self):
        return self.children
    
    def Element(self):
        return self.element
    
    def HasChildren(self):
        return self.hasChildren

    def remove(self): # removes node and all children of node
        for c in self.children:
            if c.hasChildren == True:
                c.remove()
            else:
                c.updateNumChildren(-1)
                del c # does nothing
        self.children = []
        gc.collect()

    def updateNumChildren(self, p): # keeps number of children updated
        self.numChildren += p
        
    def DFS(self, board): # pre-order DFS
        t = []
        if self.hasChildren == False: # end of tree
            return [self]
        else:
            t.append(self) # add node before children have been visited
            for c in self.children:
                t.extend(c.PreOrderTraversal()) # recursively visit each child
            return t

    def BFS(self, d=1): # breadth-first traversal
        t = []
        if self.isRoot() == True: # root is first in list
            t = [self]
            
        for c in self.children: # add children
            t.append(c)
                
        for c in t: # for children of children
            if c.depth != 0: # end of recursive depth
                t.extend(c.BFS(d+1)) # add next nodes to list with +1 level of depth
        return t

    def isRoot(self): # sees if the node is the root node
        if self.depth==0:
            return True
        else:
            return False
        
    def __repr__(self):
        return str(self)

    def __str__(self):
        r = ""
        s = str("element: " + str(self.element))
        if self.parent == None:
            r = "   root"
        return s + r + "   children: " +str(self.numChildren) + "   depth: " +str(self.depth)

class BinaryNode(Node):
    def setChild(self, c):
        if self.children == []: # stops new object from initializing with child array from parent
            reset = []
            self.children = reset
        if len(self.children) < 2:
            self.children.append(c)
        else:
            print("Parent cannot have more than two children")
        c.parent=self
        c.depth = self.depth + 1
        self.updateNumChildren(1)
        if c.getParent() == None:
            c.parent = self
        else:
            print("node already has parent!")
        self.hasChildren = True

def printTrace(l):
    for node in l:
        print(str(node))

def Demo():
    print("Pre/in/postorder and BFS traversals.")
    
    a = Node(1)

    b = Node(2)
    c = Node(3)

    d = Node(4)
    e = Node(5)
    f = Node(6)
    g = Node(7)

    a.setChild(b)
    a.setChild(c)

    b.setChild(d)
    b.setChild(e)

    c.setChild(f)
    c.setChild(g)

    print("---")
    print("Pre-order traversal:")
    printTrace(a.PreOrderTraversal())
    print("---")

    print("---")
    print("In-order traversal:")
    printTrace(a.InOrderTraversal())
    print("---")

    print("---")
    print("Post-order traversal:")
    printTrace(a.PostOrderTraversal())
    print("---")

    print("---")
    print("Breadth-first traversal:")
    printTrace(a.BFS())
    print("---")
