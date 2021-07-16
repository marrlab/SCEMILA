from bokeh.plotting import figure, show, save, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ColorBar, LinearColorMapper
from bokeh.models import FuncTickFormatter, FixedTicker, Legend, BasicTickFormatter, Panel, Tabs
from bokeh.palettes import Turbo256 as palette_umap
from bokeh.transform import linear_cmap
import matplotlib.colors as mpt_colors
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pkl
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.optimize import fmin
from scipy.spatial import distance
import os

FONTSIZE=12
output_notebook()

col_scheme = pkl.load(open('suppl_data/color_scheme.pkl', 'rb'))
col_map = col_scheme[0]
col_get = lambda x: mpt_colors.rgb2hex(col_map[x][0])
col_edge_get = lambda x: mpt_colors.rgb2hex(col_map[x][1])
shape_get_bokeh = lambda x: shape_bokeh_dict[col_map[x][2]]
shape_get_matplotlib = lambda x: col_map[x][2]
legend_order = col_scheme[1]

capitalize = lambda y: y[0].upper() + y[1:]

shape_bokeh_dict = {
    'o':lambda fig,**kwargs: fig.circle(**kwargs),
    '^':lambda fig,**kwargs: fig.triangle(**kwargs),
    's':lambda fig,**kwargs: fig.square(**kwargs),
    'P':lambda fig,**kwargs: fig.plus(**kwargs),
}

pool_dict = {
    'ambiguous':'No clinical assessment',
    'other':'No clinical assessment',
    'mononucleosis':'Healthy / AML unrelated',
    'monocyte':'Can indicate AML',
    'normo':'Can indicate AML',
    'erythroblast':'Can indicate AML',
    'proerythroblast':'Can indicate AML',
    'neoplastic lymphocyte':'Healthy / AML unrelated',
    'reactive lymphocyte':'Healthy / AML unrelated',
    'plasma cell':'Healthy / AML unrelated',
    'large granulated lymphocyte':'Healthy / AML unrelated',
    'typical lymphocyte':'Healthy / AML unrelated',
    'hair cell':'Healthy / AML unrelated',
    'basophil granulocyte':'Healthy / AML unrelated',
    'eosinophil granulocyte':'Healthy / AML unrelated',
    'neutrophil granulocyte (segmented)':'Healthy / AML unrelated',
    'neutrophil granulocyte (band)':'Healthy / AML unrelated',
    'metamyelocyte':'Healthy / AML unrelated',
    'myelocyte':'Can indicate AML',
    'promyelocyte':'Can indicate AML',
    'atypical promyelocyte':'AML-PML-RARA specific',
    'faggot cell':'AML-PML-RARA specific',
    'atypical promyelocyte with auer rod':'AML-PML-RARA specific',
    'atypical promyelocyte, bilobed':'AML-PML-RARA specific',
    'myeloblast':'Indicates AML',
    'cup-like blast':'AML-NPM1 specific',
    'myeloblast with auer rod':'Indicates AML',
    'myeloblast with long auer rod':'AML-RUNX1-RUNX1T1 specific',
    'pathological eosinophil':'AML-CBFB-MYH11 specific',
    'monoblast':'Indicates AML',
    'promonocyte':'Indicates AML',
    'smudge':'No clinical assessment',
    'cell':'cell'
}

pool_labels = lambda x: pool_dict[x]

def swarmplot(df, xlim, ylim, title="Swarmplot", legend_header="", **kwargs):
    df = df.drop(columns=[str(x) for x in range(12800)])

    # if annotation exists, drop 'cell' datapoints
    if len(df['color_values'].unique()) > 1:
        df = df.loc[~(df['color_values']=='cell')]

    df['color'] = df['color_values'].apply(col_get)
    df['edgecolor'] = df['color_values'].apply(col_edge_get)
    size=6
    
    plot_figure=figure(title=title, plot_width=900, 
                       plot_height=500, tools=(''),
                    x_axis_type="log", x_range=xlim, y_range=ylim,
                    x_axis_label = 'Single cell attention')
    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Annotation:</span>
            <span style='font-size: 12px'>@mll_annotation</span>
            <span style='font-size: 12px'>@im_tiffname</span>
        </div>
    </div>
    """))

    legend = Legend()
    legend.title = legend_header
    legend.click_policy="hide"
    plot_figure.add_layout(legend, 'right')
    
    plot_figure.yaxis.visible = False
    plot_figure.xgrid.grid_line_color = None
    plot_figure.ygrid.grid_line_color = None
    plot_figure.xaxis.formatter = BasicTickFormatter(use_scientific=False)
    plot_figure.outline_line_color = None
    plot_figure.title.align = 'center'
    
    for ctype in legend_order:
        
        ctype_df = df.loc[df['color_values'] == ctype]   
        if len(ctype_df) > 0:
            datasource = ColumnDataSource(ctype_df)
            marker_function = shape_get_bokeh(ctype)
            marker_function(fig=plot_figure, x='x', y='y', fill_color='color', line_color="edgecolor", 
                               source=datasource, legend_label=capitalize(ctype), size=size, line_width=0.5, **kwargs)
    

    return plot_figure
    
def multi_swarmplot(df, xlim, ylim, title, path_save=None, **kwargs):
    swarm_regular = swarmplot(df, xlim, ylim, title, legend_header="Annotated cell type", **kwargs)
    tab1 = Panel(child=swarm_regular, title="Full annotation")
    
    df_simplified = df.copy()
    df_simplified['color_values'] = df_simplified['color_values'].apply(pool_labels)
    swarm_simplified = swarmplot(df_simplified, xlim, ylim, title, legend_header="Annotated cell group", **kwargs)
    tab2 = Panel(child=swarm_simplified, title="Reduced annotation")
    
    if path_save is None:
        # if no path_save is given, show
        show(Tabs(tabs=[tab1, tab2]))
    else:
        # otherwise, save
        output_file(path_save)
        save(Tabs(tabs=[tab1, tab2]))

def export_swarmplot(df, xlim, ylim, title, highlight_idx=None, path_save=None, **kwargs):
    
    dotsize=35
    custom_zoom=0.7
    ylim=(ylim[0]*custom_zoom, ylim[1]*custom_zoom)
    
    df = df.copy()
    # if annotation exists, drop 'cell' datapoints
    if len(df['color_values'].unique()) > 1:
        df = df.loc[~(df['color_values']=='cell')]

    fig, ax = plt.subplots(figsize=(10,11))
    ax.set_xscale('log')
    yrange = ylim[0]-ylim[1]
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[1], yrange - ylim[1])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('Single cell attention', fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE, ha='center')
    
    df['color'] = df['color_values'].apply(col_get)
    df['edgecolor'] = df['color_values'].apply(col_edge_get)
    
    # plot regular swarmplot
    for ctype in legend_order:
        ctype_df = df.loc[df['color_values'] == ctype]   
        if len(ctype_df) > 0:
            ax.scatter(x=ctype_df.x, y=ctype_df.y, color=ctype_df.color, edgecolor=ctype_df.edgecolor, 
                               label=capitalize(ctype), marker=shape_get_matplotlib(ctype), s=dotsize, linewidth=0.5, **kwargs)
            
    leg = ax.legend(loc=6, bbox_to_anchor=(1.1, 0.0, 0.5, 0.5), title="Annotated cell type", 
              title_fontsize=FONTSIZE, edgecolor='w', fontsize=FONTSIZE)
    leg._legend_box.align = "left"

    # plot simplified swarmplot
    ax2 = ax.twinx()
    ax2.set_xscale('log')
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.set_ylim(ylim[1], yrange - ylim[1])
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks([])
    
    df['color_values_pooled'] = df['color_values'].apply(pool_labels)
    df['color'] = df['color_values_pooled'].apply(col_get)
    df['edgecolor'] = df['color_values_pooled'].apply(col_edge_get)
    # plot regular swarmplot
    for ctype in legend_order:
        ctype_df = df.loc[df['color_values_pooled'] == ctype]   
        if len(ctype_df) > 0:
            ax2.scatter(x=ctype_df.x, y=ctype_df.y+yrange, color=ctype_df.color, edgecolor=ctype_df.edgecolor, 
                               label=capitalize(ctype), marker=shape_get_matplotlib(ctype), s=dotsize, linewidth=0.5, **kwargs)
            
    leg = ax2.legend(loc=6, bbox_to_anchor=(1.1, 0.5, 0.5, 0.5), title="Grouped cell type", 
              title_fontsize=FONTSIZE, edgecolor='w', fontsize=FONTSIZE)
    leg._legend_box.align = "left"
    
    # plot in highlighted images
    # draw out lines and plot images
    if not highlight_idx is None:
        im_buffer = {}
        for identifier in highlight_idx:
            cell = df.loc[df['im_id']==identifier].iloc[0]
            x, y = cell.x, cell.y
            class_lbl = cell.color_values
            ax2.plot([x, x], [y, y+yrange], c='k', zorder=5)
        
            # load and display image
            im = Image.open(cell.im_path)        
            im_buffer[x] = im
        
            ax2.scatter(x, y, color = col_get(class_lbl), linewidth=0.5,
                    s=dotsize, zorder=10, marker=shape_get_matplotlib(class_lbl), edgecolors=col_edge_get(class_lbl))
        
            class_lbl = cell.color_values_pooled
            ax2.scatter(x, y+yrange, color = col_get(class_lbl), linewidth=0.5,
                    s=dotsize, zorder=10, marker=shape_get_matplotlib(class_lbl), edgecolors=col_edge_get(class_lbl))


        # shift images a little bit to improve optics
        global xpoints
        xpoints = sorted(im_buffer.keys())

        def log_x_dist(x1, x2):
            if min(x1, x2) <= 0:
                return 10000
            return math.log10(max(x1, x2)/min(x1, x2))

        def f_positions(shifts):
            global xpoints

            # calculate distances to close points
            xpoints_shifted = [xpoints[x]*shifts[x] for x in range(len(xpoints))]
            el_dists = np.array([log_x_dist(xpoints_shifted[x], xpoints_shifted[x+1]) for x in range(len(xpoints)-1)])
            mean_dist = np.mean(el_dists)
            dist_loss = np.sum(np.square(el_dists-mean_dist))    

            return dist_loss

        # calculate coordinates
        shift_images = fmin(f_positions, np.array([1]*len(xpoints)))

        # add images
        for x in xpoints:
            im = im_buffer[x]
            ab = AnnotationBbox(OffsetImage(im, zoom=0.5), (x*shift_images[xpoints.index(x)], yrange+ylim[1]), frameon=True, pad=0.0)
            ab.set_zorder(10)
            ax2.add_artist(ab)

    
    ax.text(x=0.01, y=0.01, s="Low attention", transform=ax.transAxes, ha='left', fontsize=FONTSIZE)
    ax.text(x=0.99, y=0.01, s="High attention", transform=ax.transAxes, ha='right', fontsize=FONTSIZE)
    
    if not path_save is None:
        fig.savefig(path_save, bbox_inches='tight')
    plt.close('all')


def calculate_cells_in_quantiles(df, target_column, quantiles=[0.25, 0.5, 0.75], percent_columns=True, sort_by_percentage=True, group_index=True):
    
    global borders 
    borders = df[target_column].quantile([0.25,0.5,0.75])
    
    def calculate_single_quantile(value):
        global borders
        
        min_quant = 0.0
        min_val = 0
        for quantile, value_quantile in borders.iteritems():
            
            if min_val <= value < value_quantile:
                return str(min_quant) + ' - ' + str(quantile)

            min_quant = quantile
            min_val = value_quantile
    
        return str(min_quant) + ' - ' + str(1.0)

    df_quant = df.copy()
    df_quant = df_quant.loc[~df_quant['mll_annotation'].isna()]

    if(group_index):
        df_quant['mll_annotation'] = df_quant['mll_annotation'].apply(pool_labels)
    df_quant['quantiles'] = df[target_column].apply(calculate_single_quantile)

    quants_available = sorted(list(df_quant['quantiles'].unique()))

    df_buffer = []
    for ctype in legend_order:
        df_filtered = df_quant.loc[df_quant['mll_annotation']==ctype]
        
        if len(df_filtered) == 0:
            continue
        
        quantiles_ctype = df_filtered['quantiles'].value_counts()

        buffer_entry = [ctype]
        for q in quants_available:
            try:
                buffer_entry.append(quantiles_ctype[q])
            except KeyError:
                buffer_entry.append(0)
        df_buffer.append(buffer_entry)

    columns_out = ['Cell type']
    columns_out.extend(quants_available)

    df_out = pd.DataFrame(df_buffer, columns=columns_out).set_index('Cell type')
    if(percent_columns):
        for q in quants_available:
            df_out[q] = (df_out[q]/sum(df_out[q]))
            if(sort_by_percentage):
                df_out = df_out.sort_values(by=q, ascending=False)

    df_out = df_out.round(2)
    return df_out
        


            


    

    








# below: UMAP
class MidpointNormalize(mpt_colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpt_colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Note that I'm ignoring clipping and other edge cases here.
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)


def umap(df, title="UMAP", legend_header="Annotated cell type", data_column='mll_annotation', grayscatter=True, **kwargs):
    df = df.copy()

    if('148' in list(df.columns)):
        df = df.drop(columns=[str(x) for x in range(12800)])

    df['info'] = df[data_column]
    size=8
    
    plot_figure=figure(title=title, plot_width=900, 
                        plot_height=700, tools=('pan, wheel_zoom, reset'),
                        aspect_scale=2)
    
    #     plot_figure.yaxis.visible = False
    plot_figure.xgrid.grid_line_color = None
    plot_figure.ygrid.grid_line_color = None
    plot_figure.outline_line_color = None
    plot_figure.title.align = 'center'
    
    if grayscatter:
        df['outline'] = ['black']*len(df)
        df['fill'] = ['white']*len(df)
        background_dsource = ColumnDataSource(df)
        plot_figure.circle(source=background_dsource, x='x', y='y', fill_color='outline', line_color='outline',radius=0.15)
        plot_figure.circle(source=background_dsource, x ='x', y='y', fill_color='fill', line_color='fill', radius=0.14)
    
    if(data_column=='mll_annotation'):
        df['color_values'] = df[data_column].fillna('cell')


        df['color'] = df['color_values'].apply(col_get)
        df['edgecolor'] = df['color_values'].apply(col_edge_get)
        df['pat_id'] = df.index
        
    
    


        legend = Legend()
        legend.title = legend_header
        legend.click_policy="hide"
        plot_figure.add_layout(legend, 'right')
        
        for ctype in legend_order:
            if ctype == 'cell':
                continue
        
            ctype_df = df.loc[df['color_values'] == ctype]   
            if len(ctype_df) > 0:
                datasource = ColumnDataSource(ctype_df)
                marker_function = shape_get_bokeh(ctype)
                marker_function(fig=plot_figure, x='x', y='y', fill_color='color', line_color="edgecolor", 
                                   source=datasource, legend_label=capitalize(ctype), size=size, line_width=0.5, name='needshover', **kwargs)
                
    if('occl' in data_column):
        norm = MidpointNormalize(vmin=-0.15, vmax=0.15, midpoint=0)
        cmap = cm.bwr.reversed()
        
        # order scatterplot by absolute values
        df['zorder'] = df[data_column].apply(abs)
        df = df.sort_values(by='zorder', ascending=True)
        
        colors = [
            "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*cmap(norm(df[data_column]))
        ]
        df['colors']=colors
        
        datasource = ColumnDataSource(df)
        plot_figure.circle(source=datasource, x ='x', y='y', fill_color='colors', line_color='colors', size=size, name='needshover')
        
    if('att' in data_column):
        norm = mpt_colors.Normalize(vmin=df[data_column].min()*1.2, vmax=df[data_column].max())
        cmap = cm.jet
    
        # order scatterplot by absolute value        
        df = df.sort_values(by=data_column, ascending=True)
        
        colors = [
            "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*cmap(norm(df[data_column]))
        ]
        df['colors']=colors
        
        datasource = ColumnDataSource(df)
        plot_figure.circle(source=datasource, x ='x', y='y', fill_color='colors', line_color='colors', size=size, name='needshover')
    
    
    plot_figure.add_tools(HoverTool(names=['needshover'], tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 18px; color: #224499'>Info:</span>
            <span style='font-size: 18px'>@info</span>
            <span style='font-size: 18px'>@index</span>
        </div>
    </div>
    """))

    show(plot_figure)



    
def export_umap(df_in, minimalize=True, title='UMAP embedding: Predicted single cell class', data_column='mll_annotation', 
                legend_capt='Predicted class', highlight=False, custom_label_order=None,  zorder_adapt_by_color=True, 
               grayscatter=True, dotsize=35, path_save=None):
    
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)

    x_min, x_max = min(df_in.x)-1, max(df_in.x)+1
    y_min, y_max = min(df_in.y)-1, max(df_in.y)+1
        
    ax.set_xlabel('UMAP_1', fontsize=FONTSIZE)
    ax.set_ylabel('UMAP_2', fontsize=FONTSIZE)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.axis('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axisbelow(True)
    
    scatter_highlight_buffer = []
    
    if(grayscatter):
#         ax.scatter(df_grayscatter.x, df_grayscatter.y, color='whitesmoke', edgecolor='whitesmoke', s=df_grayscatter.dotsize)
        ax.scatter(df_in.x, df_in.y, color='k', edgecolor='k', s=56)
    
        ax.scatter(df_in.x, df_in.y, color='white', edgecolor='white', s=50)
    
    if(data_column == 'mll_annotation'):
        
        # for the legend
        classes_order = legend_order
        if not (custom_label_order is None):
            classes_order = custom_label_order
            
        # drop non-annotated cells
        df_categorical = df_in.loc[~df_in[data_column].isna()].copy()
            
        if zorder_adapt_by_color:
            val_count = df_categorical[data_column].value_counts()
            zorder_transform = lambda x: val_count[x]
            df_categorical['order_z'] = df_categorical[data_column].apply(zorder_transform)
            df_categorical = df_categorical.sort_values(by=['order_z'], ascending=False)
            
        for label in classes_order:
            
            if not label in df_categorical[data_column].unique():
                continue
            
            # first plot a single point for the legend
            
            df_plot_tmp = df_categorical.loc[df_categorical[data_column] == label].iloc[0]
            
            label_cap = label[0].upper() + label[1:]
            sc = ax.scatter(df_plot_tmp.x, df_plot_tmp.y, color=col_get(label), edgecolor=col_edge_get(label),
                            s=dotsize, label=label_cap, linewidth=0.5,
                            marker = shape_get_matplotlib(label))
        
        # then plot all the points in the correct order
        for label in df_categorical[data_column].unique():
            df_plot_tmp = df_categorical.loc[df_categorical[data_column] == label]
            sc = ax.scatter(df_plot_tmp.x, df_plot_tmp.y, color=col_get(label), edgecolor=col_edge_get(label),
                                s=dotsize, marker=shape_get_matplotlib(label), linewidth=0.5,)
            
        if highlight:
            scatter_highlight_df = df_categorical.loc[df_categorical.highlight]
            for label in scatter_highlight_df[data_column].unique():
                df_plot_tmp = scatter_highlight_df.loc[scatter_highlight_df[data_column] == label]
                sc = ax.scatter(df_plot_tmp.x, df_plot_tmp.y, color=col_get(label), edgecolor='k',
                                s=dotsize+50, marker=shape_get_matplotlib(label), linewidth=1,zorder=100000)
            scatter_highlight_buffer = []
            idx_counter = 0
            if 'im_path' in scatter_highlight_df.columns:
                for el in list(scatter_highlight_df.im_path):
                    im = Image.open(el)
                    save_dirname = os.path.join(os.path.dirname(path_save), str(idx_counter)+'.TIF')
                    im.save(save_dirname)
                    idx_counter += 1
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=legend_capt, fontsize=FONTSIZE, title_fontsize=FONTSIZE,
                 edgecolor='w')
    if('att' in data_column):
        norm = mpt_colors.Normalize(vmin=df_in[data_column].min()*1.2, vmax=df_in[data_column].max())
        cmap = cm.jet
        
        # sort dataframe
        df_in = df_in.sort_values(by=data_column, ascending=True)
    
        sc = ax.scatter(df_in.x, df_in.y, c=df_in[data_column], s=dotsize, norm=norm, cmap=cmap)
        
        cbar = plt.colorbar(sc)
        cbar.set_label('Single cell attention for ' + data_column, rotation=90)
        
        if highlight:
            scatter_highlight_df = df_in.loc[df_in.highlight]
            sc = ax.scatter(scatter_highlight_df.x.values, scatter_highlight_df.y.values, c=scatter_highlight_df[data_column],
                            s=75, edgecolors='k')
            # scatter_highlight_buffer = []
            # if 'name' in scatter_highlight_df.columns:
            #     for el in list(scatter_highlight_df.name):
            #         im = Image.open(el)
            #         scatter_highlight_buffer.append(im)
    
    if('occl' in data_column):
        
        norm = MidpointNormalize(vmin=-0.15, vmax=0.15, midpoint=0)
        cmap = cm.bwr.reversed()
        
        # order scatterplot by absolute values
        df_in['zorder'] = df_in[data_column].apply(abs)
        df_in = df_in.sort_values(by='zorder', ascending=True)
    
        sc = ax.scatter(df_in.x, df_in.y, c=df_in[data_column], s=dotsize, norm=norm, cmap=cmap)
        
        cbar = plt.colorbar(sc)
        cbar.set_label('Change in attention through occlusion for ' + data_column, rotation=90)
        
        if highlight:

            # if highlighting is active, look up X cells
            tmp = df_in.sort_values(by=data_column, ascending=True)
            
            highest_occl = tmp.iloc[-3:]
            lowest_occl = tmp.iloc[:3]
            

            ''' From here on out, it gets very confusing. 
            The next lines deal with highlighting the right cells in the umap and drawing the actual cell image into the figure. 
            For this, the image function is approximated with f_min from scipy, where multiple factors flow into a distance
            metric. Manual work would probably have been more efficient, but after many changes in parameters this now works quite well
            (as long as not too many cells are highlighted!)
            '''
            global f_pos_distance_sample, f_pos_target_sample, highlight_cells, plotted_images, c_counter
            highlight_cells = pd.concat([highest_occl, lowest_occl])
            f_pos_distance_sample = np.array(df_in.sample(frac=0.1)[['x', 'y']])
            f_pos_target_sample = np.array(highlight_cells[['x', 'y']])
            plotted_images = []
            
            def f_positions(locs):
                global f_pos_distance_sample, f_pos_target_sample, highlight_cells, plotted_images, c_counter
                min_dist = 2

                locs = np.reshape(locs, (int(locs.shape[0]/2), 2))
                # calculate distances to rough outlines. Maximize distance along first axis
                dist_outline = distance.cdist(locs, f_pos_distance_sample, 'euclidean')
                
                distance_loss = np.sum(np.where(dist_outline < min_dist))
#                 dist_min_outline = np.amin(dist_outline, axis=1)
#                 distance_loss = np.sum(np.exp(-(dist_min_outline-3)))
                
                # calculate distances to plotted images. Maximize!
                if len(plotted_images)>0:
                    plotted = np.array(plotted_images)
                    dist_outline = distance.cdist(locs, plotted, 'euclidean')
                    dist_min_outline = np.amin(dist_outline, axis=1)
                    distance_loss_plotted = np.sum(np.exp(-(dist_min_outline-4.2)))
                else:
                    distance_loss_plotted = 0
                
                # calculate distances to points the images belong to. Minimize
                dist_target = distance.cdist(locs, f_pos_target_sample[c_counter, np.newaxis], 'euclidean')
                target_dist = np.square(dist_target[0, 0])
                
                return distance_loss + target_dist + distance_loss_plotted
            
            c_counter = 0
            for idx, row in highlight_cells.iterrows():
                # load and display image
                im = Image.open(row.im_path)
                
                # find good position for image
                coord_start = np.array([row.x, row.y])
                coord, fopt, _,_,_ = fmin(f_positions, coord_start, maxiter=10000, maxfun=10000, disp=False, full_output=1)

                thresh=15
                
                while distance.cdist(np.reshape(coord_start, (1,2)), np.reshape(coord, (1,2)), 'euclidean') < 2 or fopt > thresh or (not (x_max > coord[0] > x_min) or (not(y_max > coord[1] > y_min))):
                    coord, fopt, _,_,_ = fmin(f_positions, coord_start+np.random.normal(loc=0.0, scale=1.0, size=(2)), maxiter=10000, maxfun=10000, disp=False, full_output=1)
                    thresh += 0.5
                    
                x = coord[0]
                y = coord[1]
                plotted_images.append([x, y])
                
                ax.plot([row.x, x], [row.y, y], color='k', zorder=9, linewidth=0.5)
                ab = AnnotationBbox(OffsetImage(im, zoom=0.3), (x, y), frameon=True, pad=0.0)
                ab.set_zorder(10)
                ax.add_artist(ab)  
                c_counter += 1

            scatter_highlight_df = highlight_cells
            sc = ax.scatter(scatter_highlight_df.x.values, scatter_highlight_df.y.values, c=scatter_highlight_df[data_column], s=dotsize, norm=norm, cmap=cmap,
                            edgecolors='k', zorder=11, linewidth=0.5)
            # scatter_highlight_buffer = []
            # if 'name' in scatter_highlight_df.columns:
            #     for el in list(scatter_highlight_df.name):
            #         im = Image.open(el)
            #         scatter_highlight_buffer.append(im)

    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if not path_save is None:
        fig.savefig(path_save, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()
    
    return scatter_highlight_buffer, fig, ax