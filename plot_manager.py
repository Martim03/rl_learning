import matplotlib.pyplot as plt


class PlotManager():
    def __init__(self):
        plt.ion()
        self.plots = {}

    def kill(self):
        plt.ioff()

    def create_plot(self, title: str, xlabel: str, ylabel: str, format: str = 'r-') -> str:
        fig, ax = plt.subplots()
        x_data, y_data = [], []
        line, = ax.plot([], [], format)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        self.plots[title] = [fig, ax, x_data, y_data, line]

        return title

    def update_plot(self, title: str, x: int, y: int):
        fig, ax, x_data, y_data, line = self._get_plot(key=title)

        # Update Plot
        x_data.append(x)
        y_data.append(y)
        line.set_xdata(x_data)
        line.set_ydata(y_data)

        # Rescale
        ax.relim()
        ax.autoscale_view()

        # Draw
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        

    def _get_plot(self, key: str) -> list:
        return self.plots[key]
