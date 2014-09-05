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
    def __init__(self, point, hierarchy, center=False, color=None):
        self.point = point
        self.isCenter = center
        self.color = color
        self.hierarchy = hierarchy
        self.image = None
        

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
    def __init__(self, parent, hierarchy, showReps=False):
        
        self.width = 800
        self.height = 800
        
        Frame.__init__(self,parent, width=self.width, height=self.height)

        #Radius of Ovals
        self.pointRadius=18

        #self.docs = docs.members
        self.hierarchy = hierarchy
        self.displayRepresentatives = showReps
        
        #precompute similarity matrix. Does not change.
        #self.similarities = utils.pairwise(self.docs, lambda x,y: x.similarity(y))

        self.classic = False

        self.canvas = Canvas(self,width=self.width, height=self.height, offset="5,5", bg="white", highlightthickness=0)

        #Draw axis lines
        self.canvas.create_line(0,self.height/2, self.width, self.height/2, fill="light grey")
        self.canvas.create_line(self.width/2,0,self.width/2, self.height, fill="light grey")

        #compute and display MDS points
        self.displayPoints()
        
        
        self.selected = []

        self.thumbnail = None
        self.thumbView = None

        #Bind text and ovals to clickPoint
        self.canvas.tag_bind("point", "<Double-Button-1>",self.doubleClickPoint)
        self.canvas.tag_bind("point", "<ButtonPress-1>", self.clickPoint)
        self.canvas.tag_bind("point", "<B1-Motion>", self.dragPoint)
        
        self.canvas.tag_bind("point", "<Shift-ButtonPress-1>", self.shiftClickPoint)
        self.canvas.tag_bind("point", "<Shift-B1-Motion>", self.shiftDragPoint)
        self.canvas.tag_bind("point", "<Shift-Double-Button-1>",self.shiftDoubleClickPoint)
        
        
        self.canvas.bind("<ButtonPress-1>", self.mouseDown)
        
        self.canvas.tag_bind("point", "<Enter>", self.enteredPoint)
        self.canvas.tag_bind("point", "<Leave>", self.leftPoint)

        self.canvas.pack(fill=BOTH, expand=1)


        #Press r to refresh the MDS points
        parent.bind("t", self.toggleReps)
        parent.bind("r",self.displayPoints)
        parent.bind("c", self.clusterSelected)
        parent.bind("<Return>", self.newWindow)
        
        self.bind("<Configure>", self.on_resize)
       
       
    def clusterSelected(self, event=None):
        popup = Toplevel(self)
        
        hierarchy = None
        
        if (self.displayRepresentatives):
            hierarchy = Hierarchy(representatives=map(lambda s: self.findHierarchy(s).hierarchy, self.selected))
        else:
            reps = []
            
            for tag in self.selected:
                reps += self.findHierarchy(tag).hierarchy.representatives
       
            hierarchy = Hierarchy(representatives=reps)
       
        frame = GraphFrame(popup, hierarchy)
        frame.pack(fill=BOTH, expand=1)
        
    def mouseDown(self, event):
        #print "Mouse Down on Canvas", CURRENT
        pass


    def enteredPoint(self,event):
        tags = self.canvas.itemcget(event.widget.find_closest(event.x, event.y), "tags").split(" ")
        #print tags
        tag = tags[1]
            
        #print tag
        point = self.findHierarchy(tag)
        #print "Found point"
        thumbIm = self.getPhotoImage(point, 300)
        
        
        self.displayThumb(thumbIm, title=point.hierarchy.uId)
        
        
        #print "generated Thumb"
        #self.canvas.delete("thumbnail")
        #self.canvas.create_image(event.x, event.y, image=self.thumbIm, tags="thumbnail")
        #print "Thumb displayed"

    
    def leftPoint(self, event):
        pass

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


    def newWindow(self, event=None):
        #TODO: Pass off a subset of hierarchy to other graphFrame.
        popup = Toplevel(self)
        
        hierarchy = Hierarchy(representatives=map(lambda s: self.findHierarchy(s).hierarchy, self.selected))
        
        frame = GraphFrame(popup, hierarchy, showReps=True)
        frame.pack(fill=BOTH,expand=1)

    def shiftDoubleClickPoint(self,event):
        pass
    
    def shiftDragPoint(self, event):
        vector = (event.x-self.lastPos[0], event.y-self.lastPos[1])

        self.lastPos= (event.x, event.y)

        map(lambda tag: self.canvas.move(tag,vector[0], vector[1]), self.selected)

    
    def shiftClickPoint(self, event):
        self.lastTags = self.canvas.itemcget(event.widget.find_closest(event.x, event.y), "tags").split(" ")[1]
        self.lastPos = (event.x,event.y)
        
        self.togglePoint(self.lastTags)
        if(self.lastTags in self.selected):
            self.selected.remove(self.lastTags)
        else:
            self.selected.append(self.lastTags)
        self.canvas.tag_raise(self.lastTags)

    def clickPoint(self, event):
        #print "Click:", event.x, event.y

        self.lastTags = self.canvas.itemcget(event.widget.find_closest(event.x, event.y), "tags").split(" ")[1]
        self.lastPos = (event.x,event.y)


        self.togglePoints(self.selected)
        self.selected = []
        
        self.togglePoint(self.lastTags)
        self.selected.append(self.lastTags)
        self.canvas.tag_raise(self.lastTags)

    def dragPoint(self, event):
        
        vector = (event.x-self.lastPos[0], event.y-self.lastPos[1])

        self.lastPos= (event.x, event.y)

        self.canvas.move(self.lastTags,vector[0], vector[1])
        
        #print "Drag:", self.lastPos
        
        
    #Find point double clicked, Load Image in a new window.
    def doubleClickPoint(self,event):
        #print "Double Click:", event.x, event.y

        docTag = self.canvas.itemcget(event.widget.find_closest(event.x,event.y), "tags").split(" ")[1]

        point = self.findHierarchy(docTag)

        im = self.getPhotoImage(point,800)

        popup = Toplevel(self)
        popup.title("Doc " + str(point.hierarchy.uId))
    

        lbl = Label(popup, image=im)
        lbl.image = im
        lbl.pack()

    def getPhotoImage(self, point, size=800):
        
        if(point.image == None):
            _doc = point.hierarchy.center

            point.image = _doc.draw()
        
        im = ImageTk.PhotoImage(resizeImage(point.image, size))
        
        return im

    def findHierarchy(self, docTag):
        idx = docTag[4:]
        return self.points[int(idx)]
    
    def toggleReps(self,event):
        self.displayRepresentatives = not self.displayRepresentatives
        self.points = self.normalizeHierarchy(self.hierarchy)
        self.drawPoints(self.points)

    def togglePoints(self, tags):
        map(self.togglePoint, tags)
        
    def togglePoint(self, tag):
        color = self.canvas.itemcget(tag, "fill")
        outline = self.canvas.itemcget(tag, "outline")
        
        options = {}
        if(outline == "black"):
            options = {'outline': color}
        else:
            options = {'outline': "black"}
        
        try:
            self.canvas.itemconfigure(tag,options)
        except:
            pass


    #Recalculates and displays points using MDS
    def displayPoints(self, event=None):
        print "Displaying Points"

        #Calculate MDS
        hierarchy = self.getPoints()

        #normalize points to fit in view
        self.points = self.normalizeHierarchy(hierarchy)
        self.drawPoints(self.points)
        
    
    def displayThumb(self, thumb, title=""):
        title = "Thumbnail " + str(title)
        
        
        try:
            self.thumbView.lbl.configure(image=thumb)
            self.thumbView.title(title)
            self.thumbView.lbl.im = thumb
        except:
            self.thumbView = Toplevel(self)
            self.thumbView.title(title)
    
                
            self.thumbView.lbl = Label(self.thumbView, image=self.thumbnail)
            self.thumbView.lbl.im = thumb
            self.thumbView.lbl.pack()
        
            
        


    def normalizePoint(self, point, high, low, maximum=1, minimum=0):
        p = [0]*len(point)
        for i in range(len(point)):
            p[i] = ((point[i] - low)/(high-low))*(maximum-minimum) + minimum
            
        return p

    def getRandomColor(self,seed = None):
        random.seed(seed)
        
        #ensure color is sufficiently bright.
        colorSum = 0
        while (colorSum < 2**12):
            red = random.getrandbits(12)
            green = random.getrandbits(12)
            blue = random.getrandbits(12)
            
            colorSum = red + green + blue
        
        return "#%03x%03x%03x" % (red, green, blue)

    def normalizeHierarchy(self, hierarchy ):
        
        points=[]
        
        xcoord = map(lambda x: x[0],hierarchy.mdsPos)
        ycoord = map(lambda y: y[1],hierarchy.mdsPos)

        wMax = max(xcoord)
        hMax = max(ycoord)
        wMin = min(xcoord)
        hMin = min(ycoord)

        low = max(wMax,hMax)
        high = min(wMin,hMin)
        dim = min(self.width, self.height)
        
        lBound = -0.4*dim
        hBound = 0.4*dim

        Q = []   
        
        #Doc Centers
        for i,point in enumerate(hierarchy.mdsPos):
            color = self.getRandomColor(hierarchy.representatives[i].uId)
            p = Point(self.normalizePoint(point, high,low, hBound, lBound), hierarchy.representatives[i],center=True, color = color)
            points.append(p)
            if (self.displayRepresentatives):
                Q.append((p, hierarchy.representatives[i]))

        dim = 200
        #handle representatives
        while(len(Q) > 0):
            cPoint, rep = Q.pop(0)
            
            if(rep.representatives == None):
                continue
            
            xcoord = map(lambda x: x[0],rep.mdsPos)
            ycoord = map(lambda y: y[1],rep.mdsPos)

            wMax = max(xcoord)
            hMax = max(ycoord)
            wMin = min(xcoord)
            hMin = min(ycoord)
    
            low = max(wMax,hMax)
            high = min(wMin,hMin)
            
            
            lBound = -0.4*dim
            hBound = 0.4*dim
            
            for i,point in enumerate(rep.mdsPos):
                p = Point(self.normalizePoint(point, high,low, hBound, lBound),rep.representatives[i], color = cPoint.color)
                
                p[0] += cPoint[0]
                p[1] += cPoint[1]
                
                points.append(p)
                Q.append((p,rep.representatives[i]))

        return points
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

            if(p.isCenter):
                self.canvas.create_rectangle(bbox, tags=t, fill=p.color)
            else:
                self.canvas.create_oval(bbox, tags=t, fill=p.color)
                
            #Place text in center of Oval
            center = ((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)
            self.canvas.create_text(center, tags=t, text=str(p.hierarchy.uId))


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
    
    print clustering[0].members[0].label
    
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
