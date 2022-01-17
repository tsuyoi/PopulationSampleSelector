import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .select_from_collection import SelectFromCollection


def plot_samples(x_emb, le, labels, lookup_table, vcf_file, algorithm, n_aims, x_component=None, y_component=None):
    plt.clf()
    unique = np.unique(labels)
    colors = [plt.cm.tab10_r(i/float(len(unique)-1)) for i in range(len(unique))]
    assignments = [colors[i] for i in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    xc = []
    yc = []
    cols = []
    label_names = []
    patches = []
    for (i, cla) in enumerate(set(labels)):
        s = None
        if le.inverse_transform([cla])[0] == 'me':
            s = 500
        xc = xc + [p for (j, p) in enumerate(x_emb[:, x_component-1]) if labels[j] == cla]
        yc = yc + [p for (j, p) in enumerate(x_emb[:, y_component-1]) if labels[j] == cla]
        cols = cols + [c for (j, c) in enumerate(assignments) if labels[j] == cla]
        label_names = label_names + [le.inverse_transform([cla])[0]]
        patches = patches + [mpatches.Patch(color=cols[-1:][0], label=[le.inverse_transform([cla])[0]])]
    pts = ax.scatter(xc, yc, s=None, c=cols)
    plt.legend(handles=patches)
    plt.xlabel('Component {}'.format(x_component))
    plt.ylabel('Component {}'.format(y_component))
    selector = SelectFromCollection(ax, pts, lookup_table)
    return fig, selector


def draw_figure(canvas, fig):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)
