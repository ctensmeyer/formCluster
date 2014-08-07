#!/usr/bin/python

import PIL
import tkFileDialog
import random
import doc
import utils
import sys
import mds
import driver
from hierarchy import Hierarchy
from Tkinter import *
from ttk import Style
from PIL import Image, ImageTk


class Point:
    def __init__(self, point, canvas=None):
        self.point = point

        if(canvas != None):
            self.drawfunction = canvas.create_oval

    def setCanvas(self, canvas):
        self.drawfunction = canvas.create_oval
    
    def setDrawFunction(self, func):
        self.drawfunction = func

    def draw(*params):
        self.drawfunction(params)

    def __getitem__(self, index):
        return self.point[index]

    def __setitem__(self, key, value):
        self.point[key] = value

    def __add__(self, other):
        return self.point + other

    def __sub__(self, other):
        return self.point - other

    def __mul__(self, other):
        return self.point * other

    def __div__(self,other):
        return self.point.__div__(other)

class GraphFrame(Frame):
    def __init__(self, parent, hierarchy):
        
        self.width = 800
        self.height = 800
        
        Frame.__init__(self,parent, width=self.width, height=self.height)

        #Radius of Ovals
        self.pointRadius=18

        #self.docs = docs.members
        self.hierarchy = hierarchy
        
        #precompute similarity matrix. Does not change.
        #self.similarities = utils.pairwise(self.docs, lambda x,y: x.similarity(y))

        self.classic = False

        self.canvas = Canvas(self,width=self.width, height=self.height, offset="5,5", bg="white", highlightthickness=0)

        #Draw axis lines
        self.canvas.create_line(0,self.height/2, self.width, self.height/2, fill="light grey")
        self.canvas.create_line(self.width/2,0,self.width/2, self.height, fill="light grey")

        #compute and display MDS points
        self.displayPoints()

        #Bind text and ovals to clickPoint
        self.canvas.tag_bind("point", "<Double-Button-1>",self.doubleClickPoint)
        self.canvas.tag_bind("point", "<ButtonPress-1>", self.clickPoint)
        self.canvas.tag_bind("point", "<B1-Motion>", self.dragPoint)

        self.canvas.pack(fill=BOTH, expand=1)


        #Press r to refresh the MDS points
        parent.bind("r",self.displayPoints)
        self.bind("<Configure>", self.on_resize)


    #resize canvas to fit into window on resize
    def on_resize(self, event):
        dim = min(event.width, event.height)
        scale = float(dim)/min(self.width,self.height)
        #wscale = float(event.width)/self.width
        #hscale = float(event.height)/self.height
        
        #self.width = event.width
        #self.height = event.height
        
        self.width = dim
        self.height = dim
        
        self.canvas.width = dim
        self.canvas.height = dim
        
        
        self.config(width=self.width, height=self.height)
        self.canvas.config(width=self.canvas.width, height=self.canvas.height)
        
        self.canvas.scale("all",0,0,scale,scale)        

    def clickPoint(self, event):
        #print "Click:", event.x, event.y

        self.lastTags = self.canvas.itemcget(event.widget.find_closest(event.x, event.y), "tags").split(" ")[1]
        self.lastPos = (event.x,event.y)

        self.canvas.tag_raise(self.lastTags)
        #print "Tags:", self.lastTags

    def dragPoint(self, event):
        
        vector = (event.x-self.lastPos[0], event.y-self.lastPos[1])

        self.lastPos= (event.x, event.y)

        self.canvas.move(self.lastTags,vector[0], vector[1])

    #Find point double clicked, Load Image in a new window.
    def doubleClickPoint(self,event):
        #print "Double Click:", event.x, event.y

        allTags = self.canvas.itemcget(event.widget.find_closest(event.x,event.y), "tags").split(" ")

        docTag = filter(lambda x: x[:4] == "doc_", allTags)
        idx = docTag[0][4:]
        
        _doc = self.docs[int(idx)]

        im = _doc.draw()

        im = ImageTk.PhotoImage(resizeImage(im, 800))

        popup = Toplevel(self)
        popup.title(docTag[0])
    

        lbl = Label(popup, image=im)
        lbl.image = im
        lbl.pack()


    #Recalculates and displays points using MDS
    def displayPoints(self, event=None):
        print "Displaying Points"

        #Calculate MDS
        hierarchy = self.getPoints()

        #normalize points to fit in view
        self.points = self.normalizeHierarchy(hierarchy)
        self.drawPoints(self.points)


    def normalizePoint(self, point, high, low, maximum=1, minimum=0):
        for i in range(len(point)):
            point[i] = ((point[i] - low)/(high-low))*(maximum-minimum) + minimum

    def normalizeHierarchy(self, hierarchy):
        xcoord = map(lambda x: x[0],hierarchy.mdsPos)
        ycoord = map(lambda y: y[1],hierarchy.mdsPos)

        wMax = max(xcoord)
        hMax = max(ycoord)
        wMin = min(xcoord)
        hMin = min(ycoord)

        low = max(wMax,hMax)
        high = min(wMin,hMin)

        dim = min(self.width, self.height)

        for point in hierarchy.mdsPos:
            self.normalizePoint(point, high,low, 0.4*dim, -0.4*dim)
        

        return hierarchy.mdsPos
        #newX = map(lambda v: (((v-wMin)/(wMax-wMin))-.5)*(.9*self.w), xcoord)
        #newY = map(lambda v: (((v-hMin)/(hMax-hMin))-.5)*(.9*self.h), ycoord)

        #for i in range(len(self.points)):
        #   self.points[i] = (((self.points[i]-minimum)/(maximum-minimum))-.5)*(.9*self.width)


    def drawPoints(self, points):
        
        self.canvas.delete("point")

        radius = self.pointRadius
        self.origin = (self.width/2,self.height/2)

        for i,p in enumerate(points):
            #Tag for each point
            t = ("point","doc_"+str(i))

            #Bounding box containing the Oval
            bbox = (self.origin[0] + p[0],self.origin[1] + p[1], self.origin[0] + p[0]+radius, self.origin[1] + p[1]+radius)

            #if(self.centerMask[i]):
            #    self.canvas.create_rectangle(bbox, tags=t)
            #else:
            self.canvas.create_oval(bbox, tags=t)
            #Place text in center of Oval
            center = ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
            self.canvas.create_text(center, tags=t, text=str(i))


    def getPoints(self,docs=None):
        
        self.hierarchy.reduce(classic=self.classic)        
        self.classic = not self.classic
        return self.hierarchy

def main(args):

    if(len(args) != 3):
        print "Usage: clusterFrame.py C clustering.pkl"
        print "     C is the cluster in clustering.pkl to display"
        sys.exit(0)

    C = int(args[1])
    path = args[2]

    print "Loading"
    clustering = utils.load_obj(path)
    #clustering  = doc.get_docs_nested(driver.get_data_dir("very_small"))
    
    hierarchy = Hierarchy.createHierarchy(clustering)

    print "Starting GUI"
    root = Tk()
    frame = GraphFrame(root, hierarchy)
    frame.pack(fill=BOTH,expand=1)
    root.mainloop()  

def resizeImage(img, scale):
    
    wsize = hsize = scale

    width =  img.size[0]
    height = img.size[1]

    if(width > height):
        wpercent = (scale/float(width))
        hsize = int((float(height)*float(wpercent)))
    else:
        hpercent = (scale/float(height))
        wsize = int((float(width)*float(hpercent)))
    
    
    i = img.resize((4*wsize,4*hsize), PIL.Image.NEAREST)
    i = i.resize((2*wsize,2*hsize), PIL.Image.BILINEAR)
    i = i.resize((wsize,hsize),PIL.Image.ANTIALIAS)
    return i

def insertTag(tag, *args):
        for widget in args:
            widget.bindtags((tag,) + widget.bindtags())
    

if __name__ == '__main__':
    main(sys.argv) 
