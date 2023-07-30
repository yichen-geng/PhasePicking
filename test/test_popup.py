from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from idlelib.tooltip import Hovertip

# Create the root window
# with specified size and title
root = Tk()
root.title("Root Window")
root.geometry("450x300")

# Create label for root window
label1 = Label(root, text="This is the root window")


# define a function for 2nd toplevel
# window which is not associated with
# any parent window
def open_Toplevel2():
    # Create widget
    top2 = Toplevel()

    # define title for window
    top2.title("Toplevel2")

    # specify size
    top2.geometry("200x100")

    # Create label
    label = Label(top2,
                  text="This is a Toplevel2 window")

    # Create exit button.
    button = Button(top2, text="Exit",
                    command=top2.destroy)

    label.pack()
    button.pack()

    # Display until closed manually.
    top2.mainloop()


# define a function for 1st toplevel
# which is associated with root window.
def open_Toplevel1():
    # Create widget
    top = Toplevel(root)

    # Define title for window
    top.title("Toplevel1")

    # specify size
    # top.aspect(500, 550, 500, 550)
    # top.aspect(10, 11, 10, 11)  # TODO
    top.geometry("+100+100")
    top.rowconfigure(0, minsize=500, weight=1)
    top.rowconfigure(1, minsize=50, weight=1)
    top.columnconfigure(0, minsize=500, weight=1)

    fig = plt.figure(constrained_layout=True, figsize=(5, 5), dpi=300)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    ax.plot(x, y, lw=2)
    ax.set_ylabel("y", labelpad=0)
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    # remove unneeded tool items
    NavigationToolbar2Tk.toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                                      t[0] not in (None, 'Pan', 'Subplots')]

    toolbar = NavigationToolbar2Tk(canvas, top, pack_toolbar=False)
    # change toolbar color
    toolbar.config(background='white')
    for item in toolbar.winfo_children():
        item.config(highlightbackground='white')  # for buttons
        item.config(background='white')  # for labels
    toolbar.grid(row=1, column=0, sticky='nsew')

    # # Create label
    # label = Label(top,
    #               text="This is a Toplevel1 window")
    #
    # # Create Exit button
    # button1 = Button(top, text="Exit",
    #                  command=top.destroy)
    #
    # # create button to open toplevel2
    # button2 = Button(top, text="open toplevel2",
    #                  command=open_Toplevel2)

    # label.pack()
    # button2.pack()
    # button1.pack()

    # Display until closed manually
    top.mainloop()


# Create button to open toplevel1
button = Button(root, text="open toplevel1",
                command=open_Toplevel1)
myTip = Hovertip(button, "This is a test.")
label1.pack()

# position the button
button.place(x=155, y=50)

# Display until closed manually
root.mainloop()

# from tkinter import *
# from tkinter import ttk
#
# root = Tk()
#
# content = ttk.Frame(root, padding=(3,3,12,12))
# frame = ttk.Frame(content, borderwidth=5, relief="ridge", width=200, height=100)
# namelbl = ttk.Label(content, text="Name")
# name = ttk.Entry(content)
#
# onevar = BooleanVar()
# twovar = BooleanVar()
# threevar = BooleanVar()
#
# onevar.set(True)
# twovar.set(False)
# threevar.set(True)
#
# one = ttk.Checkbutton(content, text="One", variable=onevar, onvalue=True)
# two = ttk.Checkbutton(content, text="Two", variable=twovar, onvalue=True)
# three = ttk.Checkbutton(content, text="Three", variable=threevar, onvalue=True)
# ok = ttk.Button(content, text="Okay")
# cancel = ttk.Button(content, text="Cancel")
#
# content.grid(column=0, row=0, sticky=(N, S, E, W))
# frame.grid(column=0, row=0, columnspan=3, rowspan=2, sticky=(N, S, E, W))
# namelbl.grid(column=3, row=0, columnspan=2, sticky=(N, W), padx=5)
# name.grid(column=3, row=1, columnspan=2, sticky=(N,E,W), pady=5, padx=5)
# one.grid(column=0, row=3)
# two.grid(column=1, row=3)
# three.grid(column=2, row=3)
# ok.grid(column=3, row=3)
# cancel.grid(column=4, row=3)
#
# root.columnconfigure(0, weight=1)
# root.rowconfigure(0, weight=1)
# content.columnconfigure(0, weight=3)
# content.columnconfigure(1, weight=3)
# content.columnconfigure(2, weight=3)
# content.columnconfigure(3, weight=1)
# content.columnconfigure(4, weight=1)
# content.rowconfigure(1, weight=1)
#
# root.mainloop()
