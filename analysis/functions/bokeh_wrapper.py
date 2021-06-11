from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ColorBar, LinearColorMapper
from bokeh.models import FuncTickFormatter, FixedTicker, Legend, BasicTickFormatter, Panel, Tabs
from bokeh.palettes import Turbo256 as palette_umap
from bokeh.transform import linear_cmap
import matplotlib.colors as mpt_colors
import pickle as pkl

output_notebook()

col_scheme = pkl.load(open('suppl_data/color_scheme.pkl', 'rb'))
col_map = col_scheme[0]
col_get = lambda x: mpt_colors.rgb2hex(col_map[x][0])
col_edge_get = lambda x: mpt_colors.rgb2hex(col_map[x][1])
shape_get = lambda x: shape_bokeh_dict[col_map[x][2]]
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
            <span style='font-size: 18px; color: #224499'>Annotation:</span>
            <span style='font-size: 18px'>@mll_annotation</span>
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
            marker_function = shape_get(ctype)
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