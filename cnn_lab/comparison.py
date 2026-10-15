"""Compare immutable completed snapshots without predicting or training again."""
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .metrics import evaluate

COLORS=['#087e83','#a75d42','#506cb2','#8d6a48','#618256','#9e536b']


def run_identity(experiment):
    return experiment.get('_experiment_id') or experiment.get('_interpret_id') or f'memory-{id(experiment)}'


def run_label(experiment):
    config=experiment['config'];number=experiment.get('experiment_sequence','—')
    date=experiment.get('completed_at','sesión actual')[:19].replace('T',' ')
    return f"#{number} · {config['dataset']} · {config['optimizer']} · {date} · {run_identity(experiment)[:8]}"


def test_fingerprint(experiment):
    """Order-independent multiset identity: source IDs plus exact image/label bytes."""
    data=experiment['data'];rows=[]
    for index in data.test:
        item=hashlib.sha256()
        source=data.source_indices[int(index)] if data.source_indices is not None else None
        item.update(json.dumps(source,ensure_ascii=False).encode())
        item.update(data.images[int(index)].detach().cpu().contiguous().numpy().tobytes())
        item.update(int(data.labels[int(index)]).to_bytes(8,'little',signed=True))
        rows.append(item.digest())
    digest=hashlib.sha256()
    digest.update(json.dumps([experiment['config']['dataset'],data.names],ensure_ascii=False).encode())
    for row in sorted(rows):digest.update(row)
    return digest.hexdigest()


# Not a pytest function even if imported into a test module.
test_fingerprint.__test__=False


def comparison_tables(experiments):
    metrics=[];settings=[]
    for experiment in experiments:
        config=experiment['config'];data=experiment['data'];history=experiment['history']
        report=evaluate(data.labels[data.test],experiment['probabilities'])
        label=run_label(experiment)
        metrics.append({'Modelo':label,'Accuracy · TEST':report['accuracy'],'F1 macro · TEST':report['f1'],
                        'Precision macro · TEST':report['precision'],'Recall macro · TEST':report['recall'],
                        'Mín. loss · VALIDACIÓN':min(row['val_loss'] for row in history),'Mejor época':experiment['best_epoch'],
                        'Épocas completadas':experiment.get('epochs_completed',len(history)),
                        'Entrenamiento (s)':experiment['elapsed'],'Parámetros':sum(parameter.numel() for parameter in experiment['model'].parameters())})
        settings.append({'Modelo':label,'Dataset':config['dataset'],'Clases':', '.join(data.names),
                         'Train / val / test':f'{len(data.train)} / {len(data.val)} / {len(data.test)}',
                         'Optimizador':config['optimizer'],'Learning rate':config['lr'],'Batch':config['batch_size'],
                         'Bloques':len(config['blocks']),'Convoluciones':sum(len(block['layers']) for block in config['blocks']),
                         'MLP oculta':' → '.join(f"{layer['units']}/{layer['activation']}/dropout{layer['dropout']}" for layer in config['dense']) or 'Sin ocultas',
                         'Augmentation':config['augment'],'Normalización':config['normalize'],'Semilla':config['seed'],
                         'Máximo de épocas':config['epochs'],'Parada temprana':config['early_stopping'],
                         'Paciencia':config['patience'],'Min delta':config['min_delta']})
    return pd.DataFrame(metrics),pd.DataFrame(settings)


def comparison_figure(experiments,metric='Loss',series=('Train','Validación')):
    figure=go.Figure();suffix='loss' if metric=='Loss' else 'accuracy'
    for index,experiment in enumerate(experiments):
        history=experiment['history'];short=f"#{experiment.get('experiment_sequence','—')} · {run_identity(experiment)[:8]}"
        for name,prefix,dash in [('Train','train','solid'),('Validación','val','dash')]:
            if name not in series:continue
            figure.add_trace(go.Scatter(x=[row['epoch'] for row in history],y=[row[f'{prefix}_{suffix}'] for row in history],
                             mode='lines+markers',name=f'{short} · {name}',legendgroup=run_identity(experiment),
                             line=dict(color=COLORS[index%len(COLORS)],dash=dash),hovertemplate='Época %{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>'))
    figure.update_layout(height=430,xaxis_title='Época realmente completada',yaxis_title='Cross-entropy media' if metric=='Loss' else 'Accuracy',
                          margin=dict(l=35,r=20,t=20,b=30),paper_bgcolor='#f7f5ef',plot_bgcolor='#f7f5ef',legend=dict(orientation='h',y=-.2))
    if metric!='Loss':figure.update_yaxes(range=[0,1])
    return figure


def render_comparison(experiments):
    st.header('Comparar modelos entrenados')
    st.caption('Sólo resultados ya completados: no entrena, no descarga y no vuelve a predecir. Esta selección no cambia la red de Interpretabilidad ni la última CNN usada para subir imágenes.')
    if len(experiments)<2:
        st.info('Necesitás al menos2 entrenamientos completados para compararlos. Se conservan los últimos6 localmente.');return
    lookup={run_identity(experiment):experiment for experiment in experiments}
    selected=st.multiselect('Experimentos para comparar',list(lookup),default=list(lookup)[-2:],
                            format_func=lambda identity:run_label(lookup[identity]),max_selections=6,key='comparison_runs_'+':'.join(lookup))
    if len(selected)<2:
        st.info('Seleccioná al menos2 modelos.');return
    chosen=[lookup[identity] for identity in selected]
    fingerprints=st.session_state.get('comparison_test_fingerprints',{})
    fingerprints={identity:value for identity,value in fingerprints.items() if identity in lookup}
    for identity in selected:
        if identity not in fingerprints:fingerprints[identity]=test_fingerprint(lookup[identity])
    st.session_state.comparison_test_fingerprints=fingerprints
    if len({fingerprints[identity] for identity in selected})>1:
        st.warning('Datasets, clases o conjuntos de test distintos: las diferencias de métricas no aíslan el efecto de la arquitectura. No es una comparación directa ni se declara un ganador.')
    else:
        st.caption('Mismo dataset, clases y muestras reales de test verificados por identidad/contenido. Aun así, pueden variar entrenamiento, semilla y preprocesado.')
    metric=st.radio('Curva para comparar',['Loss','Accuracy'],horizontal=True,key='comparison_metric')
    series=st.multiselect('Series por modelo',['Train','Validación'],default=['Train','Validación'],key='comparison_series')
    if series:st.plotly_chart(comparison_figure(chosen,metric,series),width='stretch')
    else:st.caption('Elegí al menos una serie para ver la curva.')
    st.caption('Un color por modelo; línea continua = train, discontinua = validación. Las curvas terminan en la última época real: no se rellenan ni prolongan. Train incluye actualizaciones/augmentation/dropout; validación se mide en eval.')
    metrics,settings=comparison_tables(chosen)
    st.subheader('Test · pesos del mejor checkpoint de validación')
    st.dataframe(metrics,hide_index=True,width='stretch')
    st.caption('Accuracy, precision, recall y F1 macro son de TEST; loss mínima y mejor época son de VALIDACIÓN. Mirar test repetidamente para elegir hiperparámetros sesga su evaluación independiente.')
    st.subheader('Configuraciones congeladas')
    st.dataframe(settings,hide_index=True,width='stretch')
    st.download_button('Descargar comparación CSV',metrics.to_csv(index=False),'comparacion-modelos.csv','text/csv',key='comparison_csv')
