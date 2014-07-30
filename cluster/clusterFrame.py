#!/usr/bin/python

import PIL
import tkFileDialog
import random
import doc
import utils
import sys
from Tkinter import *
from ttk import Style
from PIL import Image, ImageTk



class ClusterFrame(Frame):
    def __init__(self, parent, cluster):
        Frame.__init__(self,parent)

        self.cluster = cluster
        self.prototype = self.cluster.center
        self.protoImg = self.cluster.center.draw()

        img = resizeImage(self.protoImg,800)
        i = ImageTk.PhotoImage(img)

        self.protolbl = Label(self,image=i, bg="grey", width=800, height=800, padx=5, pady=5)
        self.protolbl.image = i
        self.protolbl.grid()

        self.thumbCanvas = Canvas(self, width=400, offset="5,5")
        self.thumbCanvas.grid(row=0, column=1, sticky=N+E+S)

        self.fillThumbs()
        
        self.scroll = Scrollbar(self, orient=VERTICAL, command=self.thumbCanvas.yview)
        self.scroll.grid(row=0,column=3, sticky=N+S)

        self.thumbCanvas['yscrollcommand'] = self.scroll.set

        self.thumbCanvas.bind("<Button-4>", lambda event: self.thumbCanvas.yview("scroll", -1, 'units'))
        self.thumbCanvas.bind("<Button-5>", lambda event: self.thumbCanvas.yview("scroll", 1, 'units'))


    def fillThumbs(self):
        members =self.cluster.members
        
        self.thumbImgs = []

        x = 200
        y = 200

        total = len(members)

        for k,m in enumerate(members):
            print "adding Image to thumbnails (%d/%d)" % (k+1,total)
            img = resizeImage(m.draw(), 400)
            i = ImageTk.PhotoImage(img)
            self.thumbCanvas.create_image(x,y,image=i)
            self.thumbImgs.append(i)
            y += i.height()+10

        self.thumbCanvas.config(scrollregion=(0,0,x,y))


def main(args):

    if(len(args) != 3):
        print "Usage: clusterFrame.py C clustering.pkl"
        print "     C is the cluster in clustering.pkl to display"
        sys.exit(0)

    C = int(args[1])
    path = args[2]

    clustering = utils.load_obj(path)

    root = Tk()
    frame = ClusterFrame(root, clustering[C])
    frame.grid()
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
