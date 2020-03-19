import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib as mpl
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging import version
__matplotlib_version__ = mpl.__version__
__below__ = version.parse(__matplotlib_version__) < version.parse("3.1.0")

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class MathTextSciFormatter(mticker.Formatter):
    '''
    This formatter can be fed to set ticklabels in scientific notation without
    the annoying "1e" notation (why would anyone do that?).
    Instead, it formats ticks in proper notation with "10^x".

    fmt: the decimal point to keep
    Usage = ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    '''

    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

class FormatScalarFormatter(mticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        mticker.ScalarFormatter.__init__(self, useOffset=offset,
                                            useMathText=mathText)

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mticker._mathdefault(self.format)

class nheatmap():
    def __init__(self, data:pd.DataFrame, lrows=None, rrows=None, tcolumns=None,
            figsize=(4, 6), bcolumns=None, sub_title_font_size=10, widths=None,
            heights=None, dfr=None, dfc=None, edgecolors='k', border=True,
            linewidths=1, wspace=0.1, hspace=0.05, xrot=45, yrot=0,
            cmapCenter='viridis', cmapDiscrete='tab20b',
            rdendrogram_size=1, cdendrogram_size=1, srot=0, cmaps={},
            showxticks=None, showyticks=None, show_cbar=True):
        """
        ## Inspired by pheatmap in R, this plotting tool aims to enable multi-level heatmap with the option to perform hierarchical clustering. The goal is to develop a python plotting package that is both intuitive in usage and extensive in plotting configuration.

        Parameters
        ----------
        data : pandas DataFrame
            The frame that stores values to plot for the center heatmap.
            Hierarchical clustering is also performed on this dataframe.
            In the future, will allow AnnData object.
        dfr : pandas DataFrame, optional
            The frame that stores values for the left side-plots. Column values
            must be unique and match the key provided in 'cmap', if provided.
        dfc : pandas DataFrame, optional
            The frame that stores values for the top side-plots. Column values
            must be unique and match the key provided in 'cmap', if provided.
        lrows : list, numpy array, optional
            The the columns keys to plot for the left side-plots. By
            default, use all columns of 'dfr'
        tcolumns : list, numpy array, optional
            The the columns keys to plot for the top side-plots. By
            default, use all columns of 'dfc'
        figsize : set, optional
            The figure size. Defaults to '(4, 6)'
        xrot : float, optional
            The degree to which the xaxis ticklabels of the center heatmap
            are rotated. Defaults to '45' degree.
        yrot : float, optional
            The degree to which the yaxis ticklabels of the center heatmap
            are rotated. Defaults to '0' degree.
        srot : float, optional
            The degree to which the titles of the side plots are rotated.
            Defaults to '0' degree.
        cmapCenter : str, optional
            The name of colormap to use for the center heatmap. Defaults to
            'viridis'.
        cmapDiscrete : str, optional
            The name of colormap to use for discrete values in side plots.
            Defaults to 'tab20b'
        cmaps : dict, optional
            A dictionary with keys that correspond to the column values of side
            plots frames. The value of the dictionary dictates the name of the
            colormap to use for the corresponding column values.
        showxticks : bool, optional
            Boolean to set whether the xtick labels of the center plot should
            be shown.
        showyticks : bool, optional
            Boolean to set whether the ytick labels of the center plot should
            be shown.
        show_cbar : bool, optional
            Boolean to set whether legends and colorbars should be shown on the
            right handside of the figure. Defaults to 'True'.

        Other parameters
        ----------------
        """
        self.data = data
        self.size = np.shape(data)
        self.figsize = figsize
        self.sub_title_font_size = sub_title_font_size
        self.widths = widths
        self.heights = heights
        self.dfr = dfr
        self.dfc = dfc
        self.min_side_width = 0.5
        self.min_side_height = 0.4
        self.border = border
        self.edgecolors = edgecolors
        self.linewidths = linewidths
        self.hspace = hspace
        self.wspace = wspace
        self.rorder = np.arange(self.size[0])
        self.corder = np.arange(self.size[1])
        self.xrot = xrot
        self.yrot = yrot
        self.showRdendrogram = False
        self.showCdendrogram = False
        self.cdendrogram_size = rdendrogram_size
        self.rdendrogram_size = cdendrogram_size
        self.side_plot_label_rot = srot
        self.cmaps = cmaps
        self.show_cbar = show_cbar
        self.default_cmaps = {'center':cmapCenter, 'discrete':cmapDiscrete}
        ## I set up some common sense strategy to hide xticks or yticks labels
        ## when the number of labels to show is above 100. Of course this can be
        ## override by setting 'showxticks' or 'showyticks'.
        if showxticks is None:
            self.showxticks = self.size[1] <= 100
        else:
            self.showxticks = showxticks
        if showyticks is None:
            self.showyticks = self.size[0] <= 100
        else:
            self.showyticks = showyticks
        if self.dfr is not None:
            assert np.shape(self.dfr)[0] == self.size[0], 'The rows size of metadata and dataframe must match'
        if self.dfc is not None:
            assert np.shape(self.dfc)[0] == self.size[1], 'The columns size of metadata and dataframe must match'
        self.cbar_data = {}
        self.legend_data = {}
        if lrows is None and self.dfr is not None:
            self.lrows = self.dfr.columns.values[::-1]
        elif lrows is None:
            self.lrows = []
        else:
            self.lrows = lrows
        self.rrows = rrows
        if tcolumns is None and self.dfc is not None:
            self.tcolumns = self.dfc.columns.values
        elif tcolumns is None:
            self.tcolumns = []
        else:
            self.tcolumns = tcolumns
        self.bcolumns = bcolumns

    def set_up_cmap(self):
        if '__discrete__' not in self.default_cmaps:
            self.default_cmaps['discrete'] = 'tab20b'
        if '__center__' not in self.default_cmaps:
            self.default_cmaps['center'] = 'vlag'

    def make_dendrogram(self, df, method='average', metric='euclidean'):
        """
        function to only generate the linkage tree and dendrogram.
        I call 'hierarchy.dendrogram' so that I know how to reorder the data.
        A separate function called 'plot_dendrogram' is the function responsible
        for actually plotting the dendrogram.

        Parameters
        ----------
        df : pandas DataFrame
            The dataframe to make the dendrogram out of.
        method : str, optional
            The clustering method. See 'hierarchy.linkage' documentation for
            available options. Defaults to 'average'.
        metric : str, optional
            The distance metric to cluster the values. See 'hierarchy.linkage'
            documentation for available options. Defaults to 'euclidean'.

        """
        linkage = hierarchy.linkage(df, method=method, metric=metric)
        dendrogram = hierarchy.dendrogram(linkage, no_plot=True)
        reorder = dendrogram['leaves']
        return linkage, reorder

    def remove_border(self, ax):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    def remove_ticks(self, ax):
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelleft=False,
            labelbottom=False) # labels along the bottom edge are off

    def plot_dendrogram(self, linkage, ax, which_side, **args):
        if 'color_threshold' not in args:
            link_color_func = lambda k: 'k'
            dendrogram = hierarchy.dendrogram(linkage, ax=ax,
                    orientation=which_side, no_labels=True,
                    link_color_func=link_color_func, **args)
        else:
            dendrogram = hierarchy.dendrogram(linkage, ax=ax,
                    orientation=which_side, no_labels=True, **args)
        number_of_leaves = len(dendrogram['leaves'])
        max_dependent_coord = max(map(max, dendrogram['dcoord']))
        min_dependent_coord = min(map(max, dendrogram['dcoord']))
        ## Constants 10 and 1.05 come from
        ## `scipy.cluster.hierarchy._plot_dendrogram`
        if which_side == 'top':
            ax.set_xlim(0, number_of_leaves * 10)
            ax.set_ylim(min_dependent_coord * 0.95, max_dependent_coord * 1.05)
        if which_side == 'left':
            ax.set_ylim(0, number_of_leaves * 10)
            ax.set_xlim(min_dependent_coord * 0.95, max_dependent_coord * 1.05)
            ax.invert_xaxis()
            ax.invert_yaxis()
        ax.grid(False)
        self.remove_border(ax)
        ax.set_yticks([])
        ax.patch.set_alpha(0)
        self.remove_ticks(ax)
        return dendrogram

    def imshow(self, df, ax, cmap, *args):
        ax.pcolormesh(df, cmap=cmap, edgecolors=self.edgecolors,
                linewidths=self.linewidths, *args)

    def dtype_numerical(self, df):
        return df.dtypes in ['float64', 'float32', 'int64', 'int32', 'int',
                'float']

    def map_discrete_to_numeric(self, df, cmap='tab20b', vmin=None, vmax=None):
        """
        function to transform each unique categorical value to a scalar value.
        This allows us to make a legend which can map each unique category to a
        color based on the 'ScalarMappable' function.
        """
        mapped_df = pd.DataFrame(0, index=df.index, columns=df.columns)
        vals = df.values.flatten()
        unique_labels = sorted(set(vals))
        if type(cmap) is not dict:
            tick_dictionary = {y:x for x, y in enumerate(unique_labels)}
            c = np.array([tick_dictionary[x] for x in unique_labels])
            if vmin is not None:
                minima = vmin
            else:
                minima = min(c)
            if vmax is not None:
                maxima = vmax
            else:
                maxima = max(c)
            norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mapped_df = df.replace(tick_dictionary)
        return mapped_df, mapper, unique_labels, tick_dictionary

    def make_center_plot(self, df, ax, cmap=None, *args):
        artist = self.imshow(df, ax, cmap=cmap, *args)
        ax.grid(False)
        if self.border is False:
            self.remove_border(ax)
        if self.showxticks:
            ax.set_xticks(np.arange(0.5, np.shape(df)[1]+0.5, 1))
            ax.set_xticklabels(df.columns.values, rotation=self.xrot)
        else:
            ax.set_xticks([])
        if self.showyticks:
            ax.set_yticks(np.arange(0.5, np.shape(df)[0]+0.5, 1))
            ax.set_yticklabels(df.index.values, rotation=self.yrot)
            ax.yaxis.tick_right()
        else:
            ax.set_yticks([])
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            length=0)          # labels along the bottom edge are off

    def hcluster(self, row_cluster=True, col_cluster=True,
            showRdendrogram=None, showCdendrogram=None, **args):
        """
        function to perform hierarchical clustering on the center data.

        Parameters
        ----------
        row_cluster : bool, optional
            Boolean to set whether to run 'hcluster' on the rows of the center
            data. Defaults to 'True'
        col_cluster : bool, optional
            Boolean to set whether to run 'hcluster' on the columns of the
            center data. Defaults to 'True'
        showRdendrogram : bool, optional
            Boolean to set whether to run 'hcluster' on the rows of the
            center data. Defaults to 'None', which will set 'showRdendrogram' to
            take the same value as 'row_cluster'
        showCdendrogram : bool, optional
            Boolean to set whether to run 'hcluster' on the columns of the
            center data. Defaults to 'None', which will set 'showCdendrogram' to
            take the same value as 'col_cluster'
        args : optional
            Additional parameters passed directly to 'make_dendrogram'.
        """

        if showRdendrogram is not None:
            self.showRdendrogram = showRdendrogram
        else:
            self.showRdendrogram = row_cluster
        if showCdendrogram is not None:
            self.showCdendrogram = showCdendrogram
        else:
            self.showCdendrogram = col_cluster
        if row_cluster:
            self.rlinkage, self.rorder = self.make_dendrogram(self.data, **args)
        else:
            self.rorder = np.arange(self.size[0])
        if col_cluster:
            self.clinkage, self.corder = self.make_dendrogram(self.data.T, **args)
        else:
            self.corder = np.arange(self.size[1])

    def make_side_plots(self, D, ax, which_side, title=None, label=None,
            key=None, *args):
        if key in self.cmaps:
            cmap = self.cmaps[key]
        else:
            cmap = None
        if isinstance(D, pd.DataFrame) is False:
            df = pd.DataFrame(D)
        else:
            df = D
        numerical_dtype = self.dtype_numerical(D)
        if numerical_dtype is False:
            if cmap is None:
                cmap = self.default_cmaps['discrete']
            df, mapper, ulab, tdict = self.map_discrete_to_numeric(df, cmap)
            self.legend_data[key] = {}
            self.legend_data[key]['transformed'] = df
            self.legend_data[key]['ulab'] = ulab
            self.legend_data[key]['mapper'] = mapper
            self.legend_data[key]['cmap'] = cmap
            self.legend_data[key]['tdict'] = tdict
        else:
            self.cbar_data[key] = {}
            self.cbar_data[key]['df'] = df
            self.cbar_data[key]['cmap'] = cmap
            minima = np.min(df.values)
            maxima = np.max(df.values)
            mid_val = (maxima - minima)/2 + minima
            norm = MidpointNormalize(vcenter=mid_val, vmin=minima, vmax=maxima)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            self.cbar_data[key]['mapper'] = mapper
            self.cbar_data[key]['norm'] = norm
        if which_side in ['left', 'right']:
            self.imshow(df, ax, cmap=cmap, *args)
        else:
            self.imshow(df.T, ax, cmap=cmap, *args)
        if self.border is False:
            self.remove_border(ax)
        ax.grid(False)
        self.remove_ticks(ax)
        if self.wspace <= 0.1 and self.side_plot_label_rot == 0:
            rotation = 45
        else:
            rotation = self.side_plot_label_rot
        if title:
            if which_side in ['left', 'right']:
                ax.set_title(title, fontsize=self.sub_title_font_size,
                        rotation=rotation, va='bottom')
            else:
                ax.yaxis.set_label_position('right')
                if which_side == 'top':
                    ax.set_ylabel(title, fontsize=self.sub_title_font_size,
                            rotation=0, ha='left', va='center')
                if which_side == 'bottom':
                    ax.set_ylabel(title, fontsize=self.sub_title_font_size,
                            rotation=0, ha='left', va='center')
        if label:
            if which_side in ['left', 'right']:
                ax.yaxis.set_label_position(which_side)
                ax.set_ylabel(label, fontsize=self.sub_title_font_size)
            else:
                ax.set_title(label, fontsize=self.sub_title_font_size)

    def run(self, rdendrogram_args={}, cdendrogram_args={}):
        """
        function to generate the figure.

        Parameters
        ----------
        rdendrogram_args : dict, optional
            A dictionary with additional parameters for when plotting the row
            dendrogram. Available parameters of 'scipy.hierarchy.dendrogram'.
        cdendrogram_args : dict, optional
            A dictionary with additional parameters for when plotting the column
            dendrogram. Available parameters of 'scipy.hierarchy.dendrogram'.
        """

        self.determine_grid_sizes()
        self.fig = plt.figure(constrained_layout=False, figsize=self.figsize)
        ncols, nrows = len(self.widths), len(self.heights)
        gspec = self.fig.add_gridspec(ncols=ncols, nrows=nrows, width_ratios=self.widths,
                                  height_ratios=self.heights)
        center_plot = [len(self.heights)-1, len(self.widths)-1-self.show_cbar*2]
        self.center_plot = center_plot
        self.plots = []
        rowHide = (len(self.tcolumns) + self.showCdendrogram)
        colHide = (len(self.lrows) + self.showRdendrogram)
        for row in range(nrows):
            _tmp = []
            for col in range(ncols):
                ax = self.fig.add_subplot(gspec[row, col])
                if row < rowHide and col < colHide:
                    self.hide_extra_grid(ax)
                elif self.show_cbar:
                    if (row != (nrows-1) and col in [ncols - 1, ncols - 2]):
                        self.hide_extra_grid(ax)
                _tmp.append(ax)
            self.plots.append(_tmp)
        self.fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        for i, r in enumerate(self.lrows):
            row, col = center_plot[0], i + self.showRdendrogram
            self.make_side_plots(self.dfr[r].iloc[self.rorder], ax=self.plots[row][col],
                    which_side='left', title=r, key=r)
        for i, c in enumerate(self.tcolumns):
            row, col = i + self.showCdendrogram, center_plot[1]
            self.make_side_plots(self.dfc[c].iloc[self.corder], ax=self.plots[row][col],
                    which_side='top', title=c, key=c)
        if self.showRdendrogram:
            self.rdendrogram = self.plot_dendrogram(self.rlinkage,
                    ax=self.plots[-1][0], which_side='left', **rdendrogram_args)
        if self.showCdendrogram:
            self.cdendrogram = self.plot_dendrogram(self.clinkage,
                    ax=self.plots[0][-1-self.show_cbar*2], which_side='top',
                    **cdendrogram_args)
        self.make_center_plot(df=self.data.iloc[self.rorder, self.corder],
                ax=self.plots[center_plot[0]][center_plot[1]],
                cmap=self.default_cmaps['center'])
        if self.show_cbar:
            self.set_up_cbar()
        return self.fig, self.plots

    def determine_grid_sizes(self):
        """
        function to determine number of subplots and individual subplot sizes.
        """
        ratio = self.figsize[0] / self.figsize[1]
        min_width = 3/self.figsize[0]*self.min_side_width
        min_height = 3/self.figsize[1]*self.min_side_height
        self.widths = ([min_width]*len(self.lrows))
        self.heights = ([min_height]*len(self.tcolumns))
        if self.showRdendrogram:
            self.widths.insert(0, self.rdendrogram_size)
        if self.showCdendrogram:
            self.heights.insert(0, self.cdendrogram_size*(ratio/1))
        self.widths.append(3+self.showRdendrogram*self.rdendrogram_size)
        self.heights.append(3+self.showCdendrogram*self.cdendrogram_size)
        if self.show_cbar:
            self.widths.append(0.25)
            self.widths.append(1)

    def hide_extra_grid(self, ax):
        ax.set_visible(False)

    def plot_cbar(self, stored:dict, ax, discrete, key, s=100, fmt=None, *args):
        """
        function to generate the colorbar and legends.

        Parameters
        ----------
        stored : dict
            The backend data on the entry values, colormap, normalized scale to
            use when generating the colorbar / legends.
        ax : matplotlib axes
            The subplot axis to to plot the colorbar / legends on.
        discrete : bool
            Whether the entry values are categorical. If they are categorical,
            then 'discrete' should be 'True', otherwise, they are assumed to be
            continuous and thus 'discrete' should be False.
        key : str
            The title of the current colorbar / legend. Should correspond to the
            column value of the current side-plot data.
        fmt : mtick.Formatter
            For formatting the tick values of the continuous colorbar.
        """

        if discrete:
            cbar_height = self.figsize[0] * self.heights[-1] / sum(self.heights)
            adjusted_s = s * (cbar_height/10)**2
            length = len(stored['ulab'])
            ncol = int(length / 10)
            if ncol < 1: ncol = 1
            for i, u in enumerate(stored['ulab']):
                val = stored['tdict'][u]
                ax.scatter([], [], color=stored['mapper'].to_rgba(val),
                        s=adjusted_s, label=u)
                ax.patch.set_alpha(0)
            ax.legend(frameon=False, title=key, ncol=ncol, loc='center',
                    fontsize=self.sub_title_font_size,
                    bbox_to_anchor=(1, 0.5))
        else:
            if __below__:
                stored['mapper'].set_array([])
                cb = mpl.colorbar.ColorbarBase(ax=ax, cmap=stored['cmap'], norm=stored['norm'])
            else:
                cb = self.fig.colorbar(stored['mapper'], cax=ax, format=fmt)
            cb.ax.text(0, 1.1, key, ha='left', va='center',
                    fontsize=self.sub_title_font_size)
            ax.patch.set_alpha(0)
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1.5)
        ax.grid(False)
        self.remove_border(ax)
        self.remove_ticks(ax)

    def set_up_cbar(self):
        """
        function to set up the colobar subplots. We iterate through each of the
        side-plots that are shown, and determine where to place them.
        The general configuration that I have settled on is to carve the
        right side of the figure into two vertical subplots. Left half of the
        subplot is reserved for continuous colorbar, right half of the subplot
        is reserved for discrete colorbar.
        """
        fmt = FormatScalarFormatter("%.2g")
        self.caxs = {}

        ## colorbar plots
        if len(self.cbar_data.keys()) > 0:
            for i, k in enumerate(self.cbar_data.keys()):
                if i == 0:
                    self.caxs[k]= self.plots[-1][-2]
                    divider_cbar = make_axes_locatable(self.caxs[k])
                else:
                    self.caxs[k] = divider_cbar.new_vertical(size='100%',
                            pad=self.figsize[0]/20 * 1)
                    self.fig.add_axes(self.caxs[k], label=k)
                self.plot_cbar(self.cbar_data[k], self.caxs[k], discrete=False,
                        key=k, fmt=fmt)
        else:
            self.plots[-1][-2].set_visible(False)

        ## legend plots
        if len(self.legend_data.keys()) > 0:
            for i, k in enumerate(self.legend_data.keys()):
                if i == 0:
                    self.caxs[k]= self.plots[-1][-1]
                    divider_legend = make_axes_locatable(self.caxs[k])
                else:
                    self.caxs[k] = divider_legend.new_vertical(size='100%',
                            pad=self.figsize[0]/20 * 1)
                    self.fig.add_axes(self.caxs[k], label=k)
                self.plot_cbar(self.legend_data[k], self.caxs[k], discrete=True,
                        key=k)
        else:
            self.plots[-1][-1].set_visible(False)

