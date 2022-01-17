import pandas as pd
import numpy as np
import traceback

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from cyvcf2 import VCF, Writer

import warnings
warnings.filterwarnings('ignore')

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# import matplotlib
# matplotlib.use('TkAgg')

import PySimpleGUI as sg

from graphing import plot_samples, draw_figure


def vcf2df(vcf_fname, samples_file):
    """Convert a subsetted vcf file to pandas DataFrame
    and return sample-level population data"""
    # samples = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel'
    # samples = 'data/integrated_call_samples_v3.20130502.ALL.panel'
    dfsamples = pd.read_csv(samples_file, sep='\t')
    dfsamples.set_index('sample', inplace=True)
    dfsamples.drop(columns=['Unnamed: 4', 'Unnamed: 5'], inplace=True)

    vcf_file = VCF(vcf_fname)
    df = pd.DataFrame(index=vcf_file.samples)
    for variant in vcf_file():
        df[variant.ID] = variant.gt_types
    df = df.join(dfsamples, how='outer')
    df = df.drop(columns=['pop', 'super_pop', 'gender'])

    return df, dfsamples


def reduce_dim(x, algorithm='PCA', n_components=2):
    """Reduce the dimensionality of the 55 AISNPs
    :param x: One-hot encoded 1kG 55 AISNPs.
    :type x: array
    :param algorithm: The type of dimensionality reduction to perform.
        One of {PCA, UMAP, TSNE}
    :type algorithm: str
    :param n_components: The number of components to return in X_red
    :type n_components: int

    :returns: The transformed X[m, n] array, reduced to X[m, n_components] by algorithm.
    """
    if algorithm == 'PCA':
        x_red = PCA(n_components=n_components).fit_transform(x)
    elif algorithm == 'TSNE':
        # TSNE, Barnes-Hut have dim <= 3
        if n_components > 3:
            print('The Barnes-Hut method requires the dimensionaility to be <= 3')
            return None
        else:
            x_red = TSNE(n_components=n_components, n_jobs=4).fit_transform(x)
            # x_red = tsne(n_components=n_components, n_jobs=4).fit_transform(x)
    elif algorithm == 'UMAP':
        x_red = umap.UMAP(n_components=n_components).fit_transform(x)
    else:
        return None
    return x_red


def encode_class(dfsamples, pop_level: str = 'pop'):
    """Encode the population lables for plotting.
    :param dfsamples: VCF derived samples in a Pandas dataframe
    :param pop_level: Population level {pop, super_pop}
    """
    _le = LabelEncoder()
    if pop_level == 'pop':
        _labels = _le.fit_transform(dfsamples['pop'].values)
    elif pop_level == 'super_pop':
        _labels = _le.fit_transform(dfsamples['super_pop'].values)
    else:
        return None
    return _le, _labels


def gui_main():
    sg.ChangeLookAndFeel('Default 1')
    _menu = [
        [
            "&File",
            [
                "&Configuration",
                "---",
                "E&xit",
            ],
        ],
        [
            "&Help",
            [
                "&About",
            ],
        ],
    ]
    _graph = [[sg.Canvas(key='fig_cv')]]
    _samples_file_selector = [
        sg.Text('Samples File:', size=(10, 1)),
        sg.Input(key="SAMPLESFILEPATH", size=(30, 1)),
        sg.FileBrowse(
            file_types=(
                ("Panel", "*.panel"),
            )
        ),
    ]
    _vcf_file_selector = [
        sg.Text('VCF File:', size=(10, 1)),
        sg.Input(key="VCFFILEPATH", size=(30, 1)),
        sg.FileBrowse(
            file_types=(
                ("VCF", "*.vcf"),
            )
        ),
    ]
    _algorithm_selector = [
        sg.Text('Algorithm:', size=(17, 1)),
        sg.Radio('PCA', 1, True, key='-ALGO_PCA-', enable_events=True),
        sg.Radio('T-SNE', 1, key='-ALGO_TSNE-', enable_events=True),
        sg.Radio('UMAP', 1, key='-ALGO_UMAP-', enable_events=True),
    ]
    _population_resolution_selector = [
        sg.Text('Population Resolution:', size=(17, 1)),
        sg.Radio('Super Population', 2, True, key='-POP_SUPER-', enable_events=True),
        sg.Radio('Population', 2, key='-POP_REGULAR-', enable_events=True),
    ]
    _update_button = [
        sg.Text(' ', size=(37, 1)),
        sg.Button('Update Graph', key="-UPDATE_GRAPH-", size=(11, 1))
    ]
    _layout = [
        [
            sg.Menu(_menu),
        ],
        [
            sg.Column(_graph),
            sg.Column([
                _samples_file_selector,
                _vcf_file_selector,
                _algorithm_selector,
                _population_resolution_selector,
                _update_button
            ])
        ],
    ]
    _window = sg.Window('Population Sample Selector', _layout)
    _fig = None
    while True:
        _event, _values = _window.read()
        if _event is None or _event == 'Exit' or _event == sg.WINDOW_CLOSED:
            break
        elif _event == "-UPDATE_GRAPH-":
            vcf_path = _values['VCFFILEPATH']
            samples_path = _values['SAMPLESFILEPATH']
            algorithm = 'PCA'
            if _values['-ALGO_PCA-']:
                pass
            elif _values['-ALGO_TSNE-']:
                algorithm = 'TSNE'
            elif _values['-ALGO_UMAP-']:
                algorithm = 'UMAP'
            pop_level = 'super_pop'
            if _values['-POP_SUPER-']:
                pop_level = 'super_pop'
            elif _values['-POP_REGULAR-']:
                pop_level = 'pop'
            df, dfsamples = vcf2df(vcf_path, samples_path)
            ncols = len(df.columns)
            ohe = OneHotEncoder(categories=[range(4)] * ncols, sparse=False)

            x = ohe.fit_transform(df.values)

            x_emb = reduce_dim(x, algorithm=algorithm, n_components=2)

            lookup_table = {}
            for i in range(len(x_emb)):
                key = '{:.8f}'.format(x_emb[i][0]) + "x" + '{:.8f}'.format(x_emb[i][1])
                lookup_table[key] = dfsamples.index[i]

            # pop_level can be either 'pop' or 'super_pop'
            le, labels = encode_class(dfsamples, pop_level=pop_level)
            fig, selector = plot_samples(x_emb, le, labels, lookup_table, vcf_path, algorithm, df.shape[1],
                                         x_component=1,
                                         y_component=2)
            draw_figure(_window['fig_cv'].TKCanvas, fig)
