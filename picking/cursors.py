import numpy as np


class WaveformCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    def __init__(self, ax, line, color='c'):
        self.ax = ax
        self.horizontal_line = ax.axhline(color=color, lw=0.8, ls='-')
        self.vertical_line = ax.axvline(color=color, lw=0.8, ls='-')  # default x=0
        self.horizontal_line.set_visible(False)
        self.vertical_line.set_visible(False)
        self.x, self.y = line.get_data()
        self._last_index = None
        # # text location in axes coords
        # self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
        self.vlines = []
        self.picks = []

    def set_cross_hair_visible(self, visible):
        need_redraw = self.vertical_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        # self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            # self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()
        else:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax and len(self.picks) < 3:
            vline = self.ax.axvline(x=self.x[self._last_index], color='m', lw=0.8, ls='-')
            # store the vertical line and the x location of the pick
            self.vlines.append(vline)
            self.picks.append(self.x[self._last_index])

    def delete_pick(self):
        for vline in self.vlines:
            vline.remove()
        self.ax.figure.canvas.draw()
        self.vlines = []
        self.picks = []

    def get_pick(self):
        return self.picks


class StationCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    def __init__(self, ax, line, dist2st, app, color='k'):
        self.ax = ax
        self.horizontal_line = ax.axhline(color=color, lw=0.8, ls='-', zorder=12)
        self.vertical_line = ax.axvline(color=color, lw=0.8, ls='-', zorder=12)
        self.horizontal_line.set_visible(False)
        self.vertical_line.set_visible(False)
        self.x, self.y = line.get_data()
        self.dist2st = dist2st
        self.app = app
        self._last_index = None
        # # text location in axes coords
        # self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.vertical_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        # self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            # self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()

    def on_mouse_press(self, event):
        if event.inaxes:
            dist = self.x[self._last_index]
            curr_station = self.dist2st[dist]
            self.app.load_station_data(num=curr_station)
