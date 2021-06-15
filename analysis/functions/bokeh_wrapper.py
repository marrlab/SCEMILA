from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ColorBar, LinearColorMapper
from bokeh.models import FuncTickFormatter, FixedTicker, Legend, BasicTickFormatter, Panel, Tabs
from bokeh.palettes import Turbo256 as palette_umap
from bokeh.transform import linear_cmap
import matplotlib.colors as mpt_colors
import matplotlib.pyplot as plt
import pickle as pkl
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fontsize=12
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
    df = df.drop(columns=range(12800))
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
    
def multi_swarmplot(df, xlim, ylim, title, **kwargs):
    swarm_regular = swarmplot(df, xlim, ylim, title, legend_header="Annotated cell type", **kwargs)
    tab1 = Panel(child=swarm_regular, title="Full annotation")
    
    df_simplified = df.copy()
    df_simplified['color_values'] = df_simplified['color_values'].apply(pool_labels)
    swarm_simplified = swarmplot(df_simplified, xlim, ylim, title, legend_header="Annotated cell group", **kwargs)
    tab2 = Panel(child=swarm_simplified, title="Reduced annotation")
    
    show(Tabs(tabs=[tab1, tab2]))

def export_swarmplot(df, xlim, ylim, title, highlight_idx=None, **kwargs):
    
    dotsize=35
    custom_zoom=0.7
    ylim=(ylim[0]*custom_zoom, ylim[1]*custom_zoom)
    
    df = df.copy()
    fig, ax = plt.subplots(figsize=(10,11))
    ax.set_xscale('log')
    yrange = ylim[0]-ylim[1]
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[1], yrange - ylim[1])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('Single cell attention', fontsize=fontsize)
    
    df['color'] = df['color_values'].apply(col_get)
    df['edgecolor'] = df['color_values'].apply(col_edge_get)
    
    # plot regular swarmplot
    for ctype in legend_order:
        ctype_df = df.loc[df['color_values'] == ctype]   
        if len(ctype_df) > 0:
            marker_function = shape_get_matplotlib(ctype)
            ax.scatter(x=ctype_df.x, y=ctype_df.y, color=ctype_df.color, edgecolor=ctype_df.edgecolor, 
                               label=capitalize(ctype), s=dotsize, linewidth=0.5, **kwargs)
            
    ax.legend(loc=6, bbox_to_anchor=(1.1, 0.0, 0.5, 0.5), title="Annotated cell type", 
              title_fontsize=fontsize, edgecolor='w', fontsize=fontsize)

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
            marker_function = shape_get_matplotlib(ctype)
            ax2.scatter(x=ctype_df.x, y=ctype_df.y+yrange, color=ctype_df.color, edgecolor=ctype_df.edgecolor, 
                               label=capitalize(ctype), s=dotsize, linewidth=0.5, **kwargs)
            
    ax2.legend(loc=6, bbox_to_anchor=(1.1, 0.5, 0.5, 0.5), title="Grouped cell type", 
              title_fontsize=fontsize, edgecolor='w', fontsize=fontsize)
    
    # plot in highlighted images
    # draw out lines and plot images
    for identifier in highlight_idx:
        cell = df.loc[df['im_id']==identifier].iloc[0]
        x, y = cell.x, cell.y
        class_lbl = cell.color_values
        ax2.plot([x, x], [y, y+yrange], c='k', zorder=5)
        
        # load and display image
        im = Image.open(cell.im_path)
        ab = AnnotationBbox(OffsetImage(im, zoom=0.5), (x, yrange+ylim[1]), frameon=True)
        ab.set_zorder(10)
        ax2.add_artist(ab)
        
        ax2.scatter(x, y, color = col_get(class_lbl), linewidth=0.5,
                   s=dotsize, zorder=10, marker=shape_get_matplotlib(class_lbl), edgecolors=col_edge_get(class_lbl))
        
        class_lbl = cell.color_values_pooled
        ax2.scatter(x, y+yrange, color = col_get(class_lbl), linewidth=0.5,
                   s=dotsize, zorder=10, marker=shape_get_matplotlib(class_lbl), edgecolors=col_edge_get(class_lbl))







# below: UMAP

def umap(df, title="UMAP", legend_header="Annotated cell type", **kwargs):
    df = df.copy()
    df['color_values'] = df['mll_annotation'].fillna('cell')

    df = df.drop(columns=range(12800))
    df['color'] = df['color_values'].apply(col_get)
    df['edgecolor'] = df['color_values'].apply(col_edge_get)
    size=6
    
    plot_figure=figure(title=title, plot_width=900, 
                       plot_height=500, tools=('wheel_zoom'),
                        aspect_scale=2)

    legend = Legend()
    legend.title = legend_header
    legend.click_policy="hide"
    plot_figure.add_layout(legend, 'right')
    
    plot_figure.yaxis.visible = False
    plot_figure.xgrid.grid_line_color = None
    plot_figure.ygrid.grid_line_color = None
    plot_figure.outline_line_color = None
    plot_figure.title.align = 'center'

    hover_labels = []

    for ctype in legend_order:
        if ctype == 'cell':
            continue
        
        ctype_df = df.loc[df['color_values'] == ctype]   
        if len(ctype_df) > 0:
            datasource = ColumnDataSource(ctype_df)
            marker_function = shape_get_bokeh(ctype)
            marker_function(fig=plot_figure, x='x', y='y', fill_color='color', line_color="edgecolor", 
                               source=datasource, legend_label=capitalize(ctype), size=size, line_width=0.5, **kwargs)
            hover_labels.append(capitalize(ctype))
    
    plot_figure.add_tools(HoverTool(names=hover_labels, tooltips="""
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

    show(plot_figure)