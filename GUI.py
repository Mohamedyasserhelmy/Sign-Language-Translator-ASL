from tkinter import *
import cv2
from PIL import ImageTk, Image

class GUI:
    
    def __init__(self, title, size):
        self.root = Tk()
        self.root.title(title)
        self.root.geometry(size)

    def create_frame(self, width, height, anchor, relx, rely, background='white'):
        frame = Frame(self.root, bg=background, width=width, height=height)
        frame.place(anchor=anchor, relx=relx, rely=rely)
        return frame
        
    def create_labels(self, label_num, labels, anchor, relx, rely, x_spacing=0, y_spacing=0, create_entrybox_per_label=False):
        entry_labels = {}
        entry_boxes = {}
        relx = relx
        rely = rely

        longest_label_spacing = len(max(labels, key=len))/100.0
        
        for i in range(label_num):
            label = Label(self.root, text = labels[i]+": ",
                           font = ("TimesNewRoman", 15))
            label.place(anchor=anchor, relx=relx, rely=rely)
            
            entry_labels[labels[i]] = label
            if create_entrybox_per_label:
                entry_box = Text(self.root, font=("TimesNewRoman", 20), height=1, width=10)
                entry_box.place(anchor=anchor, relx=relx+longest_label_spacing+0.02, rely=rely)
                
                entry_boxes[labels[i]+'_entrybox'] = entry_box
            rely += y_spacing
            relx += x_spacing
        return entry_labels, entry_boxes

    def create_buttons(self, button_num, text, anchor, relx, rely, command=None, x_spacing=0, y_spacing=0):
        buttons = {}
        relx = relx
        rely = rely
        
        for i in range(button_num):
            btn = Button(self.root, command=command, text=text[i])
            btn.place(anchor=anchor, relx=relx, rely=rely)

            buttons[text[i]+' button'] = btn
            
            rely += y_spacing
            relx += x_spacing

        return buttons
        
##def p(args,**kwargs):
##    #kwargs.update(dict(zip(args, kwargs)))
##    print(kwargs)
##    print(args)
##    p2(**kwargs)
##
##def p2(*args):
##    print(args)
##
##p('a',age='mahmoud')
##
##threshold = 0.95
##
##def p2():
##    print(threshold)
##
##title = "Sign Language Recognition GUI"
##size = "500x500"
##
##gui = GUI(title, size)
##
##buttons_num = 1
##text = ['first', 'second', 'third']
##
##Labels, labels_entryboxes = gui.create_labels(1, ['first'], 'nw', 0, 0, create_entrybox_per_label=1)
##
##buttons = gui.create_buttons(buttons_num, text, 'nw', 0.45, 0, command=lambda: p(labels_entryboxes['first_entrybox']))
##buttons2 = gui.create_buttons(buttons_num, ['print'], 'nw', 0, 0.1, command=p2, y_spacing=0.05)
##
##gui.root.mainloop()
