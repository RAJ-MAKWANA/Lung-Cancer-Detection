
import cancer_ai_diagnoser

from tkinter import Frame, Tk, BOTH, Label, Menu, filedialog, messagebox, Text
from PIL import Image, ImageTk

import os
import codecs

screenWidth = "1560"
screenHeight = "840"
windowTitle = "Lung cancer from CT image using custom neural network"

import cv2

class Window(Frame):
    _PRIOR_IMAGE = None
   

    DIAGNOSIS_RESULT = ""
    DIAGNOSIS_RESULT_FIELD = None
    


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        
        load = Image.open("lung-cancer-1.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=(int(screenWidth)/2)-load.width/2, y=((int(screenHeight)/2))-load.height/2-80)
        self._PRIOR_IMAGE = img #setup prior image instance
        self.DIAGNOSIS_RESULT_FIELD = Text(self,  width=int(screenWidth), height=13)
        self.DIAGNOSIS_RESULT_FIELD.pack ( )
        self.DIAGNOSIS_RESULT_FIELD.place(x=0, y=int(screenHeight)-200)
        
    def addDiagnosisResult (self, value):
        self.DIAGNOSIS_RESULT_FIELD.delete("1.0","end") #clear diagnostic result text element
        self.DIAGNOSIS_RESULT = "" #clear diagnostic result string variable
        self.DIAGNOSIS_RESULT_FIELD.insert(1.0, value) #add new value
        
        
        
root = Tk()
app = Window(root)



############
#screen window size, window title
root.wm_title(windowTitle)
root.geometry(screenWidth + "x" + screenHeight)

############
#menu bar
menubar = Menu(root)

# Adding a cascade to the menu bar:
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Files", menu=filemenu)

CONSTANT_DIAGNOSIS_IMAGE_SPAN = 480

# Defining function to trigger file browser
    
def loadcancerImageFromDialog():
    currdir = os.getcwd()
    image_file = filedialog.askopenfile(mode ='r', parent=root, initialdir=currdir, title='Please select an Xray Image of suspected lung cancer case:')
    root.wm_title(windowTitle + " : " + image_file.name)
    loadcancerImageFromName(image_file.name)

def loadcancerImageFromName(filename):
    app._PRIOR_IMAGE.destroy() #destroy old image
    load = Image.open(filename)
    load = load.resize((CONSTANT_DIAGNOSIS_IMAGE_SPAN, CONSTANT_DIAGNOSIS_IMAGE_SPAN),Image.ANTIALIAS) #Resized "load" image to constant size on screen. However, neural network still runs on on original image scale from filename.
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth)/2)-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2, y=((int(screenHeight)/2))-CONSTANT_DIAGNOSIS_IMAGE_SPAN/2-80)
    app.DIAGNOSIS_RESULT +=  "**Cancer Mode Result**\n" + filename+"\n\n"
    app.DIAGNOSIS_RESULT += cancer_ai_diagnoser.doOnlineInference_cancer (filename)
    print(app.DIAGNOSIS_RESULT)
    app._PRIOR_IMAGE = img #set latest instance of old image
    app.addDiagnosisResult(app.DIAGNOSIS_RESULT)
    enableDiagnosisResultColouring ( )

# Adding a load image button to the cascade menu "File"
# filemenu.add_command(label="Load image to test for pneumonia", command=loadRegularPneumoniaImageFromDialog)
filemenu.add_command(label="Load image to test for cancer", command=loadcancerImageFromDialog)

def colourDiagnosisMessageText ( diagnosisContent, startIndexText, endIndexText ):
    #If pneumonia or cancer is detected
    if ( cancer_ai_diagnoser.DIAGNOSIS_MESSAGES[0] in diagnosisContent or cancer_ai_diagnoser.DIAGNOSIS_MESSAGES[1] in diagnosisContent ):
        app.DIAGNOSIS_RESULT_FIELD.tag_add("DIAGNOSIS_RESULT_MESSAGE", startIndexText, endIndexText)
        app.DIAGNOSIS_RESULT_FIELD.tag_configure("DIAGNOSIS_RESULT_MESSAGE", background="red", foreground ="white")

    #If normal lungs state is detected
    if ( cancer_ai_diagnoser.DIAGNOSIS_MESSAGES[2] in diagnosisContent ):
        app.DIAGNOSIS_RESULT_FIELD.tag_add("DIAGNOSIS_RESULT_MESSAGE", startIndexText, endIndexText)
        app.DIAGNOSIS_RESULT_FIELD.tag_configure("DIAGNOSIS_RESULT_MESSAGE", background="green", foreground ="white")
        

def enableDiagnosisResultColouring ( ):
    diagnosisResultFieldContent = app.DIAGNOSIS_RESULT_FIELD.get("1.0","end")
    colourDiagnosisMessageText ( diagnosisResultFieldContent, "4.0", "4.21" )

############
#root cycle
root.config(menu=menubar)
root.mainloop()
