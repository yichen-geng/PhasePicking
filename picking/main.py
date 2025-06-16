import json
import tkinter as tk
from idlelib.tooltip import OnHoverTooltipBase
from tkinter import messagebox
from tkinter.filedialog import askdirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

from analysis import plot_earthquake_location, plot_station_times_on_map
from cursors import *
from traces import *

mpl.rcParams["savefig.directory"] = "../results/plots"
my_font = ('Helvetica', 24)
color_map = get_cmap('rainbow')
time_shift_ylim = 10
# remove unneeded tool items
NavigationToolbar2Tk.toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                                  t[0] not in (None, 'Pan', 'Subplots')]
hover_delay = 100


class Application(tk.Frame):
    def __init__(self, background, master):
        super().__init__(background=background, master=master)
        self.dir_path = None  # the directory that stores SAC files

        self.event_locs = np.empty(0)
        self.n_stations = None
        self.station_names = np.empty(0)
        self.station_locs = np.empty(0)
        self.distances = np.empty(0)
        self.dist2st = {}  # map distance to station number
        self.sort_indices = np.empty(0)
        self.distances_sorted = np.empty(0)
        self.distance_contours = None
        self.travel_times = np.empty(0)
        self.time_shifts = np.empty(0)
        self.time_shift_errs = np.empty(0)
        self.colors = np.empty(0)
        self.orders = np.empty(0)  # store the order to plot; highest quality on the top and lowest at the bottom
        self.sn_ratios = np.empty(0)
        self.stream_long = []
        self.stream_short = []

        self.curr_station = 0  # keep track of the current station

        # initialize frames
        self.fr_top = None
        self.fr_btm_left = None
        self.fr_btm_left_top = None
        self.fr_btm_left_btm = None
        self.fr_btm_right = None
        self.fr_btm_right_top = None
        self.fr_btm_right_ctr = None
        self.fr_btm_right_btm = None

        self.btn_load = None
        self.txt_input = None

        # for station time shift plot
        self.station_canvas = None
        self.btn_cursor1 = None
        self.connect1 = True
        self.station_cursor = None
        self.callback_id1 = None
        self.callback_id2 = None
        self.toolbar1 = None

        # for station information
        self.lbl_station = None
        self.option_menu = None
        self.quality_score = None

        # for waveform plots
        self.waveform_ax = None
        self.waveform_line = None
        self.waveform_canvas = None
        self.btn_cursor2 = None
        self.connect2 = True
        self.waveform_cursor = None
        self.callback_id3 = None
        self.callback_id4 = None
        self.toolbar2 = None

        # initialize the dictionary for storing metadata
        self.station_picks = {}  # dictionary to store picks and quality

        self.create_frames()

    def create_frames(self):
        # top frame
        self.fr_top = tk.Frame(background='white', master=self)
        self.fr_top.grid(row=0, column=0, columnspan=2, sticky='nsew')
        # create button
        self.btn_load = tk.Button(highlightbackground='white', master=self.fr_top, text="Load Data",
                                  command=self.load_event_data)

        # add tool tip
        MyTooltip(self.btn_load, "Load waveform data from an existing directory", hover_delay=hover_delay)

        # change font
        self.btn_load.config(font=my_font)

        # add button
        self.btn_load.grid(row=0, column=0)

        # bottom left frame
        self.fr_btm_left = tk.Frame(background='white', master=self)
        self.fr_btm_left.rowconfigure(0, minsize=frame_height / 20 * 19, weight=1)  # scatter plot
        self.fr_btm_left.rowconfigure(1, minsize=frame_height / 20, weight=1)  # zoom in/out buttons
        self.fr_btm_left.columnconfigure(0, weight=1)
        self.fr_btm_left.grid(row=1, column=0, sticky='nsew')

        # bottom left top frame
        self.fr_btm_left_top = tk.Frame(background='white', master=self.fr_btm_left)
        self.fr_btm_left_top.rowconfigure(0, weight=1)
        self.fr_btm_left_top.columnconfigure(0, weight=1)
        self.fr_btm_left_top.grid(row=0, column=0, sticky='nsew')

        # bottom left bottom frame
        self.fr_btm_left_btm = tk.Frame(background='white', master=self.fr_btm_left)
        self.fr_btm_left_btm.rowconfigure(0, weight=1)
        self.fr_btm_left_btm.columnconfigure(0, weight=1)
        # set to a minimum number of pixels to allow text to appear without expanding the column
        self.fr_btm_left_btm.columnconfigure(1, minsize=app_width / 5, weight=1)
        self.fr_btm_left_btm.grid(row=1, column=0, sticky='nsew')

        # bottom right frame
        self.fr_btm_right = tk.Frame(background='white', master=self)
        self.fr_btm_right.rowconfigure(0, minsize=frame_height / 20, weight=1)  # station info + quality menu
        self.fr_btm_right.rowconfigure(1, minsize=frame_height / 20 * 18, weight=1)  # waveform plots
        self.fr_btm_right.rowconfigure(2, minsize=frame_height / 20, weight=1)  # zoom in/out buttons
        self.fr_btm_right.columnconfigure(0, weight=1)
        self.fr_btm_right.grid(row=1, column=1, sticky='nsew')

        # bottom right top frame
        self.fr_btm_right_top = tk.Frame(background='white', master=self.fr_btm_right)
        self.fr_btm_right_top.rowconfigure(0, weight=1)
        self.fr_btm_right_top.columnconfigure(0, weight=1)
        self.fr_btm_right_top.columnconfigure(1, weight=1)
        self.fr_btm_right_top.grid(row=0, column=0, sticky='nsew')

        # bottom right middle frame
        self.fr_btm_right_ctr = tk.Frame(background='white', master=self.fr_btm_right)
        self.fr_btm_right_ctr.rowconfigure(0, weight=1)
        self.fr_btm_right_ctr.columnconfigure(0, weight=1)
        self.fr_btm_right_ctr.grid(row=1, column=0, sticky='nsew')

        # bottom right bottom frame
        self.fr_btm_right_btm = tk.Frame(background='white', master=self.fr_btm_right)
        self.fr_btm_right_btm.rowconfigure(0, weight=1)
        self.fr_btm_right_btm.columnconfigure(0, weight=1)
        # set to a minimum number of pixels to allow text to appear without expanding the column
        self.fr_btm_right_btm.columnconfigure(1, minsize=app_width / 5, weight=1)
        self.fr_btm_right_btm.grid(row=2, column=0, sticky='nsew')

    def load_event_data(self):
        if self.dir_path is not None:  # existing event in process
            # save the current station picks
            load = self.save_pick()
            if load:
                # save the picks at all stations to json
                self.save_station_picks()

                # reset the current station number
                self.curr_station = 0

                # reset the dictionary for storing metadata
                self.station_picks = {}  # dictionary to store picks and quality
            else:
                return

        self.dir_path = askdirectory(initialdir="../data/hinet")
        if not self.dir_path:
            return

        # open a popup window to show the progress bar
        top = tk.Toplevel(background='white')
        x = root.winfo_x()
        y = root.winfo_y()
        top.geometry("%dx%d+%d+%d" % (app_width, app_height, x, y))
        text = "Loading Data... Please Wait"
        load_text = tk.Label(text=text, background='white', foreground='black', master=top)
        load_text.configure(font=('Helvetica', 48))
        load_text.place(relx=0.5, rely=0.5, anchor='center')
        # make the load button invisible
        self.btn_load.grid_remove()
        root.wait_visibility(top)
        # preprocess event data
        self.event_locs, self.distance_contours, \
        self.station_names, self.station_locs, self.distances, self.travel_times, self.sn_ratios, \
        self.stream_long, self.stream_short = preprocess_traces(self.dir_path)
        top.destroy()
        self.n_stations = len(self.station_names)

        # build distance to station number dictionary
        for station, distance in enumerate(self.distances):
            self.dist2st[distance] = station

        # sort by distance for plotting only (snapping cursor assumes that x values are sorted)
        self.sort_indices = np.argsort(self.distances)
        self.distances_sorted = self.distances[self.sort_indices]
        self.time_shifts = np.zeros(self.n_stations)
        # self.PcP_time_shift_errs = np.zeros((2, self.n_stations))
        self.time_shift_errs = np.ones((2, self.n_stations)) * np.inf
        self.colors = np.array([(0.5, 0.5, 0.5, 1)] * self.n_stations)
        self.orders = np.zeros(self.n_stations)

        # load existing station picks and quality
        pick_path = '../results/picks/' + os.path.basename(self.dir_path) + '.json'
        if os.path.isfile(pick_path):
            with open(pick_path, 'r') as fp:
                self.station_picks = json.load(fp)
                number = 0
                for name, metadata in self.station_picks.items():
                    number = metadata['number']
                    picks = metadata['picks']
                    quality = metadata['quality']
                    if len(picks) == 3:
                        # update time shift
                        self.time_shifts[number] = picks[1]
                        # update error bar
                        # lower error
                        self.time_shift_errs[0][number] = picks[1] - picks[0]
                        # upper error
                        self.time_shift_errs[1][number] = picks[2] - picks[1]
                    # else:  # if no pick set error bar to be 0
                    #     # update error bar
                    #     # lower error
                    #     self.time_shift_errs[0][number] = 0
                    #     # upper error
                    #     self.time_shift_errs[1][number] = 0
                    # update color
                    self.colors[number] = color_map(int(quality) / 5)
                    # update order
                    self.orders[number] = int(quality)
                # update the current station number to the last station number
                self.curr_station = number

        # load station data
        # if there was no event loaded previously, no need to save the picks, either because there are no picks yet,
        # or the picks are loaded from json, so they are already saved
        # if there was event loaded previously and a new event is about to be loaded, no need to save the picks
        # because otherwise the code will save the picks of the last processed station of the old event
        # as the picks of the current station of the new event
        self.load_station_data(num=self.curr_station, save=False)

        self.create_tool_buttons()

    def create_tool_buttons(self):
        btn_prev = tk.Button(highlightbackground='white', master=self.fr_top, text="\N{LEFTWARDS BLACK ARROW}",
                             command=self.load_prev_station_data)
        btn_next = tk.Button(highlightbackground='white', master=self.fr_top, text="\N{RIGHTWARDS BLACK ARROW}",
                             command=self.load_next_station_data)
        btn_last = tk.Button(highlightbackground='white', master=self.fr_top, text="\N{UPWARDS BLACK ARROW}",
                             command=self.load_last_station_data)
        btn_delete = tk.Button(highlightbackground='white', master=self.fr_top, text="\N{CROSS MARK}",
                               command=self.delete_pick)
        btn_save = tk.Button(highlightbackground='white', master=self.fr_top, text="\N{FLOPPY DISK}",
                             command=self.save_station_picks)
        btn_earthquake = tk.Button(highlightbackground='white', master=self.fr_top, text="Plot Earthquake",
                                   command=self.plot_earthquake)
        btn_arrival = tk.Button(highlightbackground='white', master=self.fr_top, text="Plot Arrival",
                                command=self.plot_arrival)
        btn_residual = tk.Button(highlightbackground='white', master=self.fr_top, text="Plot Residual",
                                 command=self.plot_residual)
        self.txt_input = tk.Text(highlightbackground='black', master=self.fr_top,
                                 height=1, width=10)
        btn_go = tk.Button(highlightbackground='white', master=self.fr_top, text="Go",
                           command=self.load_input_station_data)

        # add tool tips
        MyTooltip(btn_prev, "Go to the previous station", hover_delay=hover_delay)
        MyTooltip(btn_next, "Go to the next station", hover_delay=hover_delay)
        MyTooltip(btn_last, "Go to the last processed station", hover_delay=hover_delay)
        MyTooltip(btn_delete, "Delete the picks", hover_delay=hover_delay)
        MyTooltip(btn_save, "Save the picks", hover_delay=hover_delay)
        MyTooltip(btn_earthquake, "Plot the earthquake location on a map", hover_delay=hover_delay)
        MyTooltip(btn_arrival, "Plot the arrival times on a map", hover_delay=hover_delay)
        MyTooltip(btn_residual, "Plot the residual times on a map", hover_delay=hover_delay)
        MyTooltip(self.txt_input, "Enter the station name", hover_delay=hover_delay)
        MyTooltip(btn_go, "Go to the station entered", hover_delay=hover_delay)

        # change font
        btn_prev.config(font=my_font)
        btn_next.config(font=my_font)
        btn_last.config(font=my_font)
        btn_delete.config(font=my_font)
        btn_save.config(font=my_font)
        btn_earthquake.config(font=my_font)
        btn_arrival.config(font=my_font)
        btn_residual.config(font=my_font)
        self.txt_input.config(font=my_font)
        btn_go.config(font=my_font)

        # add buttons
        self.btn_load.grid(row=0, column=0)
        btn_prev.grid(row=0, column=1)
        btn_next.grid(row=0, column=2)
        btn_last.grid(row=0, column=3)
        btn_delete.grid(row=0, column=4)
        btn_save.grid(row=0, column=5)
        btn_earthquake.grid(row=0, column=6)
        btn_arrival.grid(row=0, column=7)
        btn_residual.grid(row=0, column=8)
        self.txt_input.grid(row=0, column=9)
        btn_go.grid(row=0, column=10)

    def show_station_cursor(self):
        if self.connect1:  # click to connect
            self.btn_cursor1.config(foreground='black')
            self.station_cursor.set_cross_hair_visible(True)
            self.callback_id1 = self.station_canvas.mpl_connect('motion_notify_event',
                                                                self.station_cursor.on_mouse_move)
            self.callback_id2 = self.station_canvas.mpl_connect('button_press_event',
                                                                self.station_cursor.on_mouse_press)
            self.connect1 = False
            MyTooltip(self.btn_cursor1, "Deactivate cross cursor", hover_delay=hover_delay)
        else:  # click to disconnect
            self.btn_cursor1.config(foreground='grey')
            self.station_canvas.mpl_disconnect(self.callback_id1)
            self.station_canvas.mpl_disconnect(self.callback_id2)
            self.connect1 = True
            MyTooltip(self.btn_cursor1, "Activate cross cursor", hover_delay=hover_delay)

    def show_waveform_cursor(self):
        if self.connect2:  # click to connect
            self.btn_cursor2.config(foreground='black')
            self.waveform_cursor.set_cross_hair_visible(True)
            self.callback_id3 = self.waveform_canvas.mpl_connect('motion_notify_event',
                                                                 self.waveform_cursor.on_mouse_move)
            self.callback_id4 = self.waveform_canvas.mpl_connect('button_press_event',
                                                                 self.waveform_cursor.on_mouse_press)
            self.connect2 = False
            MyTooltip(self.btn_cursor2, "Deactivate cross cursor", hover_delay=hover_delay)
        else:  # click to disconnect
            self.btn_cursor2.config(foreground='grey')
            self.waveform_canvas.mpl_disconnect(self.callback_id3)
            self.waveform_canvas.mpl_disconnect(self.callback_id4)
            self.connect2 = True
            MyTooltip(self.btn_cursor2, "Activate cross cursor", hover_delay=hover_delay)

    def load_station_data(self, num, save=True):
        if save:
            load = self.save_pick()
        else:
            load = True
        if load:
            # update the current station
            self.curr_station = num

            # allow mpl connection
            self.connect1 = True
            self.connect2 = True

            # clear the previous plot if necessary
            if self.station_canvas:
                self.station_canvas.get_tk_widget().destroy()
            if self.lbl_station:
                self.lbl_station.destroy()
            if self.option_menu:
                self.option_menu.destroy()
            if self.btn_cursor1:
                self.btn_cursor1.destroy()
            if self.toolbar1:
                for item in self.toolbar1.winfo_children():
                    item.destroy()
            if self.waveform_canvas:
                self.waveform_canvas.get_tk_widget().destroy()
            if self.btn_cursor2:
                self.btn_cursor2.destroy()
            if self.toolbar2:
                for item in self.toolbar2.winfo_children():
                    item.destroy()
                # pass
            plt.close('all')

            # update time shift plot
            self.update_time_shift_plot()

            # show station information
            # station_info = "station name: " + self.station_names[num] + "   distance: " + \
            #                str(np.around(self.distances[num], 2)) + "\N{DEGREE SIGN}"
            station_info = "station name: " + self.station_names[num] + "   " + str(num + 1) + "/" + str(
                self.n_stations)
            self.lbl_station = tk.Label(background='white', foreground='blue', text=station_info,
                                        master=self.fr_btm_right_top)
            self.lbl_station.configure(font=my_font)
            self.lbl_station.grid(row=0, column=0)

            # show quality menu
            quality_scores = ["Select Quality", "5 (highest)", "4 (high)", "3 (medium)", "2 (low)", "1 (lowest)"]
            # quality_scores = ["Select Quality", "5", "4", "3", "2", "1"]
            self.quality_score = tk.StringVar(master=self.fr_btm_right_top)
            station_name = self.station_names[self.curr_station]
            # show quality if exists
            if station_name in self.station_picks:
                qs = self.station_picks[station_name]['quality']
                if qs == "5":
                    postfix = " (highest)"
                elif qs == "4":
                    postfix = " (high)"
                elif qs == "3":
                    postfix = " (medium)"
                elif qs == "2":
                    postfix = " (low)"
                elif qs == "1":
                    postfix = " (lowest)"
                else:
                    raise Exception("Invalid quality score")
                self.quality_score.set(qs + postfix)
            else:
                self.quality_score.set("Select Quality")
            self.option_menu = tk.OptionMenu(self.fr_btm_right_top, self.quality_score, *quality_scores)
            self.option_menu.config(font=my_font, width=10, bg='white', fg='black')
            self.option_menu.grid(row=0, column=1)

            # plot waveforms
            fig = plt.figure(constrained_layout=True, figsize=(6, 4), dpi=150)
            gs = fig.add_gridspec(3, 2, width_ratios=[2, 3], height_ratios=[2, 0.5, 1.5])
            self.waveform_ax = fig.add_subplot(gs[0, :])
            # ax01 = fig.add_subplot(gs[0, 1])
            ax1 = fig.add_subplot(gs[1, :])
            ax1.axis('off')
            ax2 = fig.add_subplot(gs[2, :])
            self.waveform_line = plot_traces(self.stream_long[num], self.waveform_ax)
            # plot_traces(self.stream_long[num], ax01)
            plot_record_section(self.stream_short, self.distances, num, self.sn_ratios, ax1, ax2)
            self.waveform_canvas = FigureCanvasTkAgg(fig, master=self.fr_btm_right_ctr)  # A tk.DrawingArea.
            self.waveform_canvas.get_default_filename = lambda: os.path.basename(
                self.dir_path) + "_waveform_" + station_name[-4:] + ".png"
            self.waveform_canvas.draw()
            self.waveform_canvas.get_tk_widget().grid(row=0, sticky='nsew')

            # create cursor
            self.waveform_cursor = WaveformCursor(self.waveform_ax, self.waveform_line)
            # assign and plot picks if exists
            if station_name in self.station_picks:
                picks = self.station_picks[station_name]['picks']
                self.waveform_cursor.picks = picks
                for pick in picks:
                    vline = self.waveform_ax.axvline(x=pick, color='m', lw=0.8, ls='-')
                    self.waveform_cursor.vlines.append(vline)

            # add tool bar
            self.btn_cursor2 = tk.Button(highlightbackground='white', foreground='grey', width=10,
                                         master=self.fr_btm_right_btm,
                                         text="\N{OPEN CENTRE CROSS}", command=self.show_waveform_cursor)
            MyTooltip(self.btn_cursor2, "Activate cross cursor", hover_delay=hover_delay)
            self.btn_cursor2.config(font=my_font)
            self.btn_cursor2.grid(row=0, column=0, sticky='w')
            self.toolbar2 = NavigationToolbar2Tk(self.waveform_canvas, self.fr_btm_right_btm, pack_toolbar=False)
            # change toolbar color
            self.toolbar2.config(background='white')
            for item in self.toolbar2.winfo_children():
                item.config(highlightbackground='white')  # for buttons
                item.config(background='white')  # for labels
            self.toolbar2.grid(row=0, column=1, sticky='nsew')

    def load_prev_station_data(self):
        curr_station = self.curr_station
        curr_station -= 1
        if curr_station == -1:
            curr_station = self.n_stations - 1
        self.load_station_data(num=curr_station)

    def load_next_station_data(self):
        curr_station = self.curr_station
        curr_station += 1
        if curr_station == self.n_stations:
            curr_station = 0
        self.load_station_data(num=curr_station)

    def load_last_station_data(self):
        curr_station = len(self.station_picks) - 1  # assume that picking is done in order
        self.load_station_data(num=curr_station)

    def load_input_station_data(self):
        curr_station_name = self.txt_input.get("1.0", "end-1c")
        # check if the name is valid
        if len(curr_station_name) != 4 and len(curr_station_name) != 6:
            msg = "Invalid station name. Should be either four letters or six letters that start with 'N.'"
            messagebox.showwarning(message=msg)
            return
        if len(curr_station_name) == 4:
            curr_station_name = 'N.' + curr_station_name
        if curr_station_name in self.station_names:
            curr_station = int(np.where(self.station_names == curr_station_name)[0][0])
            self.load_station_data(num=curr_station)
        else:
            msg = "Station name does not exist. Double check the spelling."
            messagebox.showwarning(message=msg)
            return

    def update_time_shift_plot(self):
        fig = plt.figure(constrained_layout=True, figsize=(3, 3), dpi=150)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        # plot with x values sorted
        plot_line, _, bar_lines = ax.errorbar(x=self.distances_sorted, y=self.time_shifts[self.sort_indices],
                                              yerr=self.time_shift_errs[:, self.sort_indices],
                                              ls='none', elinewidth=0.8, alpha=0.8, zorder=10)
        bar_lines[0].set_color(self.colors[self.sort_indices])
        # plot with highest quality on the top and lowest at the bottom
        for i in range(6):
            distances_selected = self.distances[self.orders == i]
            time_shifts_selected = self.time_shifts[self.orders == i]
            colors_selected = self.colors[self.orders == i]
            ax.scatter(distances_selected, time_shifts_selected, s=20, c=colors_selected, edgecolors='none', alpha=0.8,
                       zorder=11)
        ax.set_xlim(np.floor(self.distances_sorted[0]), np.ceil(self.distances_sorted[-1]))
        ax.set_ylim([-time_shift_ylim, time_shift_ylim])
        ax.set_xlabel(r"Distance ($^\circ$)")
        ax.set_ylabel("Residual time (s)")
        # add legend
        legend_elements = [Line2D([0], [0], ls='none', marker='o', markersize=5,
                                  markerfacecolor=(0.5, 0.5, 0.5, 1), markeredgecolor=(0.5, 0.5, 0.5, 1),
                                  label="N/A")]
        for i in range(1, 6):
            legend_elements.append(Line2D([0], [0], ls='none', marker='o', markersize=5,
                                          markerfacecolor=color_map(i / 5), markeredgecolor=color_map(i / 5),
                                          label=str(i)))
        # ax.legend(handles=legend_elements[::-1], fontsize=10, bbox_to_anchor=(0, 1., 1., 0.15), loc='upper left', ncol=3,
        #           columnspacing=0., handletextpad=0.)
        ax.legend(handles=legend_elements[::-1], fontsize=10, loc='upper left', ncol=3, columnspacing=0., handletextpad=0.)

        self.station_canvas = FigureCanvasTkAgg(fig, master=self.fr_btm_left_top)  # A tk.DrawingArea.
        self.station_canvas.get_default_filename = lambda: os.path.basename(self.dir_path) + "_residual_vs_distance.png"
        self.station_canvas.draw()
        self.station_canvas.get_tk_widget().grid(sticky='nsew')

        # create cursor
        self.station_cursor = StationCursor(ax, plot_line, dist2st=self.dist2st, app=self)

        # add tool bar
        self.btn_cursor1 = tk.Button(highlightbackground='white', foreground='grey', width=10,
                                     master=self.fr_btm_left_btm,
                                     text="\N{OPEN CENTRE CROSS}", command=self.show_station_cursor)
        MyTooltip(self.btn_cursor1, "Activate cross cursor", hover_delay=hover_delay)
        self.btn_cursor1.config(font=my_font)
        self.btn_cursor1.grid(row=0, column=0, sticky='w')
        self.toolbar1 = NavigationToolbar2Tk(self.station_canvas, self.fr_btm_left_btm, pack_toolbar=False)
        # change toolbar color
        self.toolbar1.config(background='white')
        for item in self.toolbar1.winfo_children():
            item.config(highlightbackground='white')  # for buttons
            item.config(background='white')  # for labels
        self.toolbar1.grid(row=0, column=1, sticky='nsew')

    def delete_pick(self):
        # remove picks associated with the cursor
        self.waveform_cursor.delete_pick()
        # remove the station from the dictionary
        station_name = self.station_names[self.curr_station]
        if station_name in self.station_picks:
            del self.station_picks[station_name]
        # update time shifts, errors and colors
        self.time_shifts[self.curr_station] = 0
        self.time_shift_errs[:, self.curr_station] = np.inf
        self.colors[self.curr_station] = (0.5, 0.5, 0.5, 1)
        self.orders[self.curr_station] = 0

    def save_pick(self):
        if self.waveform_cursor:
            picks = self.waveform_cursor.get_pick()
            n_picks = len(picks)
            station_name = self.station_names[self.curr_station]
            station_latitude = self.station_locs[self.curr_station][0]
            station_longitude = self.station_locs[self.curr_station][1]
            distance = self.distances[self.curr_station]
            travel_time = self.travel_times[self.curr_station]
            quality_score = self.quality_score.get()[0]
            load = False
            if (n_picks == 3 and quality_score in ['2', '3', '4', '5']) or (n_picks == 0 and quality_score == '1'):
                picks_sorted = np.sort(picks)  # sort
                # save to dictionary
                self.station_picks[station_name] = {}
                self.station_picks[station_name]['number'] = self.curr_station
                self.station_picks[station_name]['latitude'] = station_latitude
                self.station_picks[station_name]['longitude'] = station_longitude
                self.station_picks[station_name]['distance'] = distance
                self.station_picks[station_name]['travel_time'] = travel_time
                self.station_picks[station_name]['picks'] = picks_sorted.tolist()
                self.station_picks[station_name]['quality'] = quality_score
                if n_picks == 3:
                    # update time shift
                    self.time_shifts[self.curr_station] = picks_sorted[1]
                    # update error bar
                    # lower error
                    self.time_shift_errs[0][self.curr_station] = picks_sorted[1] - picks_sorted[0]
                    # upper error
                    self.time_shift_errs[1][self.curr_station] = picks_sorted[2] - picks_sorted[1]
                # else:  # if no pick set error bar to be 0
                #     # update error bar
                #     # lower error
                #     self.time_shift_errs[0][self.curr_station] = 0
                #     # upper error
                #     self.time_shift_errs[1][self.curr_station] = 0
                # update color
                self.colors[self.curr_station] = color_map(int(quality_score) / 5)
                # update order
                self.orders[self.curr_station] = int(quality_score)
                load = True
            elif n_picks == 3 and quality_score in ['S', '1']:  # select picks but no quality
                msg = "A valid quality was not selected."
                messagebox.showwarning(message=msg)
            elif n_picks == 0 and quality_score in ['2', '3', '4', '5']:  # select quality but no picks
                msg = "No picks were not selected."
                messagebox.showwarning(message=msg)
            elif 0 < n_picks < 3:
                msg = str(len(picks)) + " picks (< 3) were selected at " + station_name + "."
                messagebox.showwarning(message=msg)
            else:  # if no pick and no quality selected, popup a message box
                msg = "No picks and quality were selected. Do you wish to continue?"
                msg_box = tk.messagebox.askquestion(message=msg)
                if msg_box == "yes":
                    load = True
                else:
                    load = False
                # load = True  # TODO: change back
            return load
        else:
            return True

    def save_station_picks(self):
        with open('../results/picks/' + os.path.basename(self.dir_path) + '.json', 'w') as fp:
            json.dump(self.station_picks, fp, indent=4)

    def get_dir_path(self):
        return self.dir_path

    def get_station_picks(self):
        return self.station_picks

    def plot_earthquake(self):
        top = tk.Toplevel()

        # set title
        top.title("Earthquake")
        # # set position
        # top.geometry("+100+100")
        # set size
        top.rowconfigure(0, minsize=500, weight=1)
        top.rowconfigure(1, minsize=50, weight=1)
        top.columnconfigure(0, minsize=500, weight=1)

        # plot map
        fig = plt.figure(figsize=(5, 4), dpi=300)
        event_lat = self.event_locs[0]
        event_lon = self.event_locs[1]
        station_lat_avg = np.mean([station_loc[0] for station_loc in self.station_locs])
        station_lon_avg = np.mean([station_loc[1] for station_loc in self.station_locs])
        plot_earthquake_location(station_lat_avg, station_lon_avg, event_lat, event_lon, fig)
        map_canvas = FigureCanvasTkAgg(fig, master=top)
        map_canvas.get_default_filename = lambda: os.path.basename(self.dir_path) + "_earthquake_location.png"
        map_canvas.draw()
        map_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # add toolbar
        toolbar = NavigationToolbar2Tk(map_canvas, top, pack_toolbar=False)
        # change toolbar color
        toolbar.config(background='white')
        for item in toolbar.winfo_children():
            item.config(highlightbackground='white')  # for buttons
            item.config(background='white')  # for labels
        toolbar.grid(row=1, column=0, sticky='nsew')

    def plot_arrival(self):
        top = tk.Toplevel()

        # set title
        top.title("Arrival")
        # # set position
        # top.geometry("+100+100")
        # set size
        top.rowconfigure(0, minsize=500, weight=1)
        top.rowconfigure(1, minsize=50, weight=1)
        top.columnconfigure(0, minsize=500, weight=1)

        # plot map
        fig = plt.figure(figsize=(5, 4), dpi=300)
        plot_station_times_on_map(self.station_picks, fig, plot_type='arrival')
        map_canvas = FigureCanvasTkAgg(fig, master=top)
        map_canvas.get_default_filename = lambda: os.path.basename(self.dir_path) + "_arrival_map.png"
        map_canvas.draw()
        map_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # add toolbar
        toolbar = NavigationToolbar2Tk(map_canvas, top, pack_toolbar=False)
        # change toolbar color
        toolbar.config(background='white')
        for item in toolbar.winfo_children():
            item.config(highlightbackground='white')  # for buttons
            item.config(background='white')  # for labels
        toolbar.grid(row=1, column=0, sticky='nsew')

    def plot_residual(self):
        top = tk.Toplevel()

        # set title
        top.title("Residual")
        # # set position
        # top.geometry("+100+100")
        # set size
        top.rowconfigure(0, minsize=500, weight=1)
        top.rowconfigure(1, minsize=50, weight=1)
        top.columnconfigure(0, minsize=500, weight=1)

        # plot map
        fig = plt.figure(figsize=(5, 4), dpi=300)
        plot_station_times_on_map(self.station_picks, fig, plot_type='residual')
        map_canvas = FigureCanvasTkAgg(fig, master=top)
        map_canvas.get_default_filename = lambda: os.path.basename(self.dir_path) + "_residual_map.png"
        map_canvas.draw()
        map_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # add toolbar
        toolbar = NavigationToolbar2Tk(map_canvas, top, pack_toolbar=False)
        # change toolbar color
        toolbar.config(background='white')
        for item in toolbar.winfo_children():
            item.config(highlightbackground='white')  # for buttons
            item.config(background='white')  # for labels
        toolbar.grid(row=1, column=0, sticky='nsew')


class MyTooltip(OnHoverTooltipBase):
    "A tooltip that pops up when a mouse hovers over an anchor widget."

    def __init__(self, anchor_widget, text, hover_delay=1000):
        """Create a text tooltip with a mouse hover delay.

        anchor_widget: the widget next to which the tooltip will be shown
        hover_delay: time to delay before showing the tooltip, in milliseconds

        Note that a widget will only be shown when showtip() is called,
        e.g. after hovering over the anchor widget with the mouse for enough
        time.
        """
        super(MyTooltip, self).__init__(anchor_widget, hover_delay=hover_delay)
        self.text = text

    def showcontents(self):
        label = tk.Label(self.tipwindow, text=self.text, justify=tk.LEFT, relief=tk.SOLID, borderwidth=1)
        label.grid()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Phase Picking")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    # get screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # remove some pixels
    cut = 20
    screen_height -= cut
    screen_width -= cut
    # set app dimension
    # app_height = screen_height / 5 * 4
    app_width = screen_width
    app_height = app_width * 0.42
    frame_height = app_height / 20 * 19
    app = Application(background='white', master=root)
    # set minimum size for rows and columns
    app.rowconfigure(0, minsize=app_height / 20, weight=1)  # tool buttons
    app.rowconfigure(1, minsize=app_height / 20 * 19, weight=1)  # everything else
    app.columnconfigure(0, minsize=app_width / 3, weight=1)
    app.columnconfigure(1, minsize=app_width / 3 * 2, weight=1)
    app.grid(sticky='nsew')
    app.mainloop()
    # save station_picks to json if it is not empty
    station_picks = app.get_station_picks()
    if station_picks:
        with open('../results/picks/' + os.path.basename(app.get_dir_path()) + '.json', 'w') as fp:
            json.dump(app.get_station_picks(), fp, indent=4)

# TODO: consider use a pop-up window to explain the quality scale
# TODO: make option menu box larger
# TODO: go to the last station how to implement
