#!/usr/bin/python

import PIL
import tkFileDialog
import random
import doc
import utils
import sys
import mds
from Tkinter import *
from ttk import Style
from PIL import Image, ImageTk



class GraphFrame(Frame):
    def __init__(self, parent, docs):
        
        self.width = 800
        self.height = 800
        
        Frame.__init__(self,parent, width=self.width, height=self.height)

        #Radius of Ovals
        self.pointRadius=18

        self.docs = docs.members

        #precompute similarity matrix. Does not change.
        self.similarities = utils.pairwise(self.docs, lambda x,y: x.similarity(y))



        self.canvas = Canvas(self,width=self.width, height=self.height, offset="5,5", bg="white")

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


    def clickPoint(self, event):
        print "Click:", event.x, event.y

        self.lastTags = self.canvas.itemcget(event.widget.find_closest(event.x, event.y), "tags").split(" ")[1]
        self.lastPos = (event.x,event.y)

        self.canvas.tag_raise(self.lastTags)
        print "Tags:", self.lastTags

    def dragPoint(self, event):
        
        vector = (event.x-self.lastPos[0], event.y-self.lastPos[1])

        self.lastPos= (event.x, event.y)

        self.canvas.move(self.lastTags,vector[0], vector[1])

    #Find point double clicked, Load Image in a new window.
    def doubleClickPoint(self,event):
        print "Double Click:", event.x, event.y
        print self.canvas.itemcget(event.widget.find_closest(event.x,event.y),"tags")
        #TODO: Load image in new window.

    #Recalculates and displays points using MDS
    def displayPoints(self, event=None):
        print "Displaying Points"

        #Calculate MDS
        self.points = self.getPoints()

        #normalize points to fit in view
        self.normalizePoints(self.points)
        self.drawPoints(self.points)


    def normalizePoints(self, points):
        xcoord = map(lambda x: x[0],points)
        ycoord = map(lambda y: y[1],points)

        wMax = max(xcoord)
        hMax = max(ycoord)
        wMin = min(xcoord)
        hMin = min(ycoord)

        maximum = max(wMax,hMax)
        minimum = min(wMin,hMin)

        #newX = map(lambda v: (((v-wMin)/(wMax-wMin))-.5)*(.9*self.w), xcoord)
        #newY = map(lambda v: (((v-hMin)/(hMax-hMin))-.5)*(.9*self.h), ycoord)

        for i in range(len(self.points)):
            self.points[i] = (((self.points[i]-minimum)/(maximum-minimum))-.5)*(.9*self.width)


    def drawPoints(self, points):
        
        self.canvas.delete("point")

        radius = self.pointRadius
        self.origin = (self.width/2,self.height/2)

        for i,p in enumerate(points):
            #Tag for each point
            t = ("point","doc_"+str(i))
            #Bounding box containing the Oval
            bbox = (self.origin[0] + p[0],self.origin[1] + p[1], self.origin[0] + p[0]+radius, self.origin[1] + p[1]+radius)
            self.canvas.create_oval(bbox, tags=t)
            #Place text in center of Oval
            center = ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
            self.canvas.create_text(center, tags=t, text=str(i))


    def getPoints(self,docs=None):
        if(docs != None):
            return mds.docReduction(docs)
        else:
            return mds.reduction(self.similarities)

def main(args):

    if(len(args) != 3):
        print "Usage: clusterFrame.py C clustering.pkl"
        print "     C is the cluster in clustering.pkl to display"
        sys.exit(0)

    C = int(args[1])
    path = args[2]

    print "Loading"
    clustering = utils.load_obj(path)

    print "Starting GUI"
    root = Tk()
    frame = GraphFrame(root, clustering[C])
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
