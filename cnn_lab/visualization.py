"""Readable local Plotly schematics derived from actual CNN modules and shape rows."""
import html
import numpy as np
import plotly.graph_objects as go
from torch import nn
from .data import ANIMAL_NAMES

COLORS = {"entrada":"#647875", "convolución":"#087E83", "normalización":"#8D6A48", "activación":"#A75D42", "pooling":"#618256", "vector":"#66839B", "densa":"#506CB2", "salida":"#9E536B"}


def _kind(name):
    if '_conv' in name: return 'convolución'
    if '_bn' in name: return 'normalización'
    if '_act' in name or ' MLP ' in name: return 'activación'
    if '_pool' in name or name=='promedio global': return 'pooling'
    if name=='flatten': return 'vector'
    if name.startswith('densa'): return 'densa'
    if name.startswith('dropout'): return 'normalización'
    return 'salida'


def _node(label,shape,params,kind,details='',rf=None):
    hover=f'<b>{html.escape(label)}</b><br>Salida: {html.escape(str(shape))}<br>Parámetros: {params:,}'
    if rf is not None and isinstance(rf,(int,float)): hover+=f'<br>RF teórico: {rf} píxeles'
    if details: hover+='<br>'+details
    return {'label':label,'shape':str(shape),'parameters':params,'kind':kind,'hover':hover,'rf':rf if isinstance(rf,(int,float)) else None}


def architecture_nodes(model,detailed=False):
    nodes=[_node('Imagen de entrada',f'{model.input_channels} × {model.input_size} × {model.input_size}',0,'entrada')]
    modules=dict(model.features.named_children())
    if detailed:
        for row in model.rows:
            name=row['capa'];detail=''
            module=modules.get(name)
            if isinstance(module,nn.Conv2d):
                detail=f'Kernel {module.kernel_size} · stride {module.stride}<br>Padding {module.padding} · dilatación {module.dilation}'
            elif isinstance(module,(nn.MaxPool2d,nn.AvgPool2d)):
                detail=f'Kernel {module.kernel_size} · stride {module.stride}'
            elif name.startswith('densa oculta'):
                layer=model.config.dense[int(name.split()[-1])-1]
                detail=f'Activación {layer.activation} · dropout {layer.dropout} · bias {layer.bias}'
            elif 'entrada' in row: detail=f"Entrada: {row['entrada']} unidades"
            nodes.append(_node(name,row['salida'],row['parámetros'],_kind(name),detail,row['RF teórico']))
    else:
        for index,block in enumerate(model.config.blocks,1):
            rows=[row for row in model.rows if row['capa'].startswith(f'b{index}_')]
            final=rows[-1]
            detail=f'{len(block.layers)} conv · activación {block.activation}<br>BatchNorm {"sí" if block.batchnorm else "no"} · pooling {block.pooling}'
            detail+='<br>'+'<br>'.join(f'Conv {i}: {layer.channels} canales, K{layer.kernel}/S{layer.stride}/P{layer.padding}/D{layer.dilation}' for i,layer in enumerate(block.layers,1))
            nodes.append(_node(f'Bloque convolucional {index}',final['salida'],sum(row['parámetros'] for row in rows),'convolución',detail,final['RF teórico']))
        vector=next(row for row in model.rows if row['capa']=='flatten')
        nodes.append(_node('GAP + Flatten' if model.config.head=='gap' else 'Flatten',f"Vector de {vector['salida']} características",0,'vector'))
        for index,layer in enumerate(model.config.dense,1):
            row=next(row for row in model.rows if row['capa']==f'densa oculta {index}')
            nodes.append(_node(f'Densa oculta {index}',f"{row['entrada']} → {row['salida']} unidades",row['parámetros'],'densa',f'Activación {layer.activation} · dropout {layer.dropout} · bias {layer.bias}'))
        output=model.rows[-1]
        nodes.append(_node('Densa de salida · logits',f"{output['entrada']} → {output['salida']} clases",output['parámetros'],'salida','Sin softmax dentro del modelo'))
    return nodes


def _layout(fig,height,y_range,x_range):
    fig.update_layout(height=height,margin=dict(l=25,r=25,t=35,b=25),paper_bgcolor='#F7F5EF',plot_bgcolor='#F7F5EF',font=dict(family='sans-serif',size=15,color='#253836'),showlegend=False,dragmode='pan',hoverlabel=dict(font_size=14),xaxis=dict(visible=False,range=x_range,fixedrange=False),yaxis=dict(visible=False,range=y_range,fixedrange=False))


def architecture_figure(model,detailed=False):
    nodes=architecture_nodes(model,detailed)
    count=len(nodes);ys=[-i for i in range(count)]
    fig=go.Figure()
    # The only connections here are forward operation order, not neuron connectivity.
    fig.add_trace(go.Scatter(x=[0]*count,y=ys,mode='lines',line=dict(color='#9AAEAA',width=2),hoverinfo='skip',name='conexiones'))
    fig.add_trace(go.Scatter(x=[0]*count,y=ys,mode='markers',marker=dict(size=30,color=[COLORS[n['kind']] for n in nodes],line=dict(color='#F7F5EF',width=2)),text=[n['hover'] for n in nodes],hovertemplate='%{text}<extra></extra>',name='nodos'))
    for node,y in zip(nodes,ys):
        text=f"<b>{html.escape(node['label'])}</b><br>{html.escape(node['shape'])} · {node['parameters']:,} parámetros"
        fig.add_annotation(x=.14,y=y,text=text,showarrow=False,xanchor='left',align='left',font=dict(size=15,color='#253836'))
    _layout(fig,max(480,count*90+50),[-count+.35,.65],[-.1,1.15])
    fig.update_layout(meta={'view':'operations' if detailed else 'blocks','nodes':nodes,'edge_count':count-1})
    return fig


def _class_names(model):
    if model.config.dataset=='CIFAR10Animals': return ANIMAL_NAMES
    if model.config.dataset=='FashionMNIST':
        from torchvision.datasets import FashionMNIST
        return FashionMNIST.classes
    return [str(i) for i in range(model.num_classes)]


def neuron_figure(model,cap=12):
    if not 4<=cap<=12: raise ValueError('Usá entre 4 y 12 neuronas representativas por capa.')
    linears=[module for module in model.classifier if isinstance(module,nn.Linear)]
    sizes=[linears[0].in_features]+[module.out_features for module in linears]
    shown=[np.unique(np.linspace(0,size-1,min(cap,size),dtype=int)).tolist() for size in sizes[:-1]]+[list(range(sizes[-1]))]
    points=[[(float((j-(len(indices)-1)/2)*.7),float(-level*2.4)) for j in range(len(indices))] for level,indices in enumerate(shown)]
    fig=go.Figure();edge_x=[];edge_y=[]
    for left,right in zip(points,points[1:]):
        for x1,y1 in left:
            for x2,y2 in right:
                edge_x.extend([x1,x2,None]);edge_y.extend([y1,y2,None])
    fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(color='rgba(99,123,137,0.24)',width=.8),hoverinfo='skip',name='conexiones densas'))
    names=_class_names(model)
    for level,(indices,coordinates,size) in enumerate(zip(shown,points,sizes)):
        output=level==len(sizes)-1
        kind='salida' if output else 'vector' if level==0 else 'densa'
        label='Salida · logits' if output else 'Características de Flatten/GAP' if level==0 else f'Densa oculta {level}'
        hover=[f'<b>{label}</b><br>{"Clase: "+html.escape(names[index]) if output else "Índice de unidad: "+str(index)}<br>{len(indices)} mostradas de {size}' for index in indices]
        fig.add_trace(go.Scatter(x=[p[0] for p in coordinates],y=[p[1] for p in coordinates],mode='markers+text' if output else 'markers',marker=dict(size=25 if output else 21,color=COLORS[kind],line=dict(color='#F7F5EF',width=1.5)),text=[html.escape(names[i]) for i in indices] if output else None,textposition='bottom center',textfont=dict(size=14),hovertext=hover,hovertemplate='%{hovertext}<extra></extra>',name=label))
        description=f'<b>{label}</b> · {len(indices)} mostradas de {size}'
        if 0<level<len(sizes)-1:
            layer=model.config.dense[level-1]
            description+=f'<br>{layer.activation} · dropout {layer.dropout} · bias {layer.bias}'
        if len(indices)<size: description+=' · …'
        fig.add_annotation(x=0,y=coordinates[0][1]+.75,text=description,showarrow=False,font=dict(size=15),align='center')
    height=max(520,len(sizes)*185)
    _layout(fig,height,[-(len(sizes)-1)*2.4-1,1.4],[-5.1,5.1])
    fig.update_layout(meta={'view':'neurons','actual_sizes':sizes,'shown_indices':shown,'edge_count':sum(len(a)*len(b) for a,b in zip(shown,shown[1:])),'output_classes':names})
    return fig
