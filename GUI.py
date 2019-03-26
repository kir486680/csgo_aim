from tkinter import *
import sys

class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.grid()
        self.createWidgets()
    def createWidgets(self):
        self.w = Scale( from_=-50, to=50)
        
        self.w.grid()

app = Application()
app.title("Sample application")
app.mainloop()