"""Bounded test gallery and attribution workspace; no dataset loading or training."""
import math
import uuid
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from .data import display_image
from .explain import gradcam_all, explain, scale_map

PAGE_SIZE = 10


def page_bounds(total, page):
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(0, min(int(page), pages - 1))
    return page, pages, page * PAGE_SIZE, min(total, (page + 1) * PAGE_SIZE)


def selection_state(previous, run_id, total):
    if previous is None or previous['run_id'] != run_id or previous['total'] != total:
        return dict(run_id=run_id, total=total, page=0, sample=0)
    page, _, start, end = page_bounds(total, previous['page'])
    return dict(previous, page=page, sample=max(start, min(previous['sample'], max(start, end - 1))))


def resize_pixels(image, side=256):
    array = np.rint(np.clip(image, 0, 1) * 255).astype('uint8')
    height, width = array.shape[:2]
    return np.asarray(Image.fromarray(array).resize((max(1, round(width * side / max(height, width))),
                                                    max(1, round(height * side / max(height, width)))), Image.Resampling.NEAREST))


def overlay_image(original, heat):
    """Warm heat, opacity proportional to magnitude: zero CAM leaves image intact."""
    base = resize_pixels(original).astype('float32') / 255
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    heat = np.asarray(Image.fromarray(np.asarray(heat, dtype='float32')).resize(
        (base.shape[1], base.shape[0]), Image.Resampling.BILINEAR))
    color = np.stack([np.ones_like(heat), .8 * (1 - heat), .12 * (1 - heat)], axis=-1)
    opacity = .65 * heat[..., None]
    return np.clip(base * (1 - opacity) + color * opacity, 0, 1)


def layer_title(name):
    block, conv = name.removeprefix('b').split('_conv')
    return f"Bloque {block} · convolución {conv}"


def render_interpretability(run):
    model, data = run['model'], run['data']
    # Identity belongs to this run, never its mutable position in the six-run history.
    identity = run.setdefault('_interpret_id', uuid.uuid4().hex)
    state = selection_state(st.session_state.get('interpret_selection'), identity, len(data.test))
    st.session_state.interpret_selection = state
    if not len(data.test):
        st.info('Este experimento no contiene muestras de test.'); return
    prefix = f'interpret_{identity}'
    st.subheader('1 · Elegí una imagen de test')
    st.caption(f"Experimento seleccionado: {run['config']['dataset']} · checkpoint época {run['best_epoch']}. Navegar no entrena ni descarga datos.")
    page, pages, start, end = page_bounds(len(data.test), state['page'])
    navigation = st.columns([1, 2, 1])
    backward = navigation[0].button('Anterior', disabled=page == 0, key=prefix+'_previous', width='stretch')
    forward = navigation[2].button('Siguiente', disabled=page == pages - 1, key=prefix+'_next', width='stretch')
    if backward or forward:
        page += 1 if forward else -1
        page, pages, start, end = page_bounds(len(data.test), page)
        state.update(page=page, sample=start)
        st.rerun()
    navigation[1].caption(f'Página {page + 1}/{pages} · {start + 1}–{end} de {len(data.test)}')
    st.caption('10 imágenes por página. Al cambiar de página se selecciona su primera imagen. El borde del botón indica la seleccionada.')
    probabilities = run['probabilities']
    with st.container(key='interpret_gallery'):
        for row_start in range(start, end, 5):
            columns = st.columns(5)
            for column, sample in zip(columns, range(row_start, min(row_start + 5, end))):
                index = data.test[sample]
                predicted = int(probabilities[sample].argmax())
                with column:
                    st.image(resize_pixels(display_image(data.images[index]), 160), width='stretch')
                    st.caption(f"Real: {data.names[int(data.labels[index])]}\n\nPred.: {data.names[predicted]} · {probabilities[sample, predicted]:.1%}")
                    if st.button(f"{'✓ ' if state['sample'] == sample else ''}Imagen {sample + 1}", key=f'{prefix}_tile_{sample}',
                                 type='primary' if state['sample'] == sample else 'secondary', width='stretch'):
                        state['sample'] = sample
                        st.rerun()
    sample = state['sample']; index = data.test[sample]
    original = display_image(data.images[index]); actual = int(data.labels[index])
    predicted = int(probabilities[sample].argmax())
    sample_key = f'{prefix}_sample_{sample}'
    st.subheader(f'2 · Imagen {sample + 1} seleccionada')
    left, right = st.columns([1, 2])
    left.image(resize_pixels(original), caption=f'Entrada original · índice test {sample} · fuente {index}', width=256)
    right.markdown(f"**Real: {data.names[actual]} · predicción: {data.names[predicted]} ({probabilities[sample,predicted]:.1%})**")
    right.caption('Coincide con la etiqueta real.' if predicted == actual else 'No coincide con la etiqueta real.')
    right.dataframe(pd.DataFrame({'Clase': data.names, 'Probabilidad softmax': probabilities[sample]}).sort_values('Probabilidad softmax', ascending=False),
                    hide_index=True, width='stretch', height=245, column_config={'Probabilidad softmax': st.column_config.NumberColumn(format='%.4f')})
    mode = st.radio('Clase para explicar', ['Clase predicha', 'Clase real', 'Elegir clase'], horizontal=True, key=sample_key+'_target_mode')
    target = predicted if mode == 'Clase predicha' else actual
    if mode == 'Elegir clase':
        target = st.selectbox('Clase objetivo', list(range(len(data.names))), index=predicted,
                              format_func=lambda item: data.names[item], key=sample_key+'_target')
    st.caption(f'Objetivo actual: logit de {data.names[target]}. Softmax no es confianza calibrada ni prueba de corrección.')
    cache_key = (identity, sample, target)
    cache = st.session_state.get('interpret_maps')
    if cache is None or cache['key'] != cache_key:
        try:
            cache = dict(key=cache_key, result=gradcam_all(model, data.normalized([index]), target))
            st.session_state.interpret_maps = cache
        except (ValueError, RuntimeError, IndexError) as error:
            st.error(f'No se pudo calcular la atribución: {error}'); return
    results = cache['result']['layers']
    st.subheader(f'3 · Grad-CAM · todas las convoluciones ({len(results)})')
    st.caption('Una superposición por convolución para la misma imagen y clase. Cada mapa se normaliza a [0,1] por separado: la intensidad NO compara magnitudes entre capas. Rojo = atribución positiva mayor; imagen base sin suavizar, CAM interpolado bilinealmente.')
    items = list(results.items())
    for start_layer in range(0, len(items), 3):
        columns = st.columns(3)
        for column, (name, result) in zip(columns, items[start_layer:start_layer + 3]):
            with column:
                st.markdown(f'**{layer_title(name)}**')
                if 'error' in result:
                    st.warning(result['error']); continue
                st.image(overlay_image(original, result['heat']), width='stretch', clamp=True)
                st.caption(f"{name} · mapas {' × '.join(map(str, result['features'].shape))}")
                if result['constant']:
                    st.caption('CAM constante/cero: no hay contraste positivo; no indica igual importancia de todos los píxeles.')
    st.caption('Grad-CAM explica sensibilidad del logit, no causalidad. Resoluciones pequeñas producen atribuciones gruesas; no demuestran que la clasificación sea correcta.')
    st.subheader('4 · Mapas de activación por canal')
    st.caption('Salidas crudas de cada convolución, antes de BatchNorm/activación: NO son Grad-CAM. Cada canal se escala por separado. Se muestran 8 canales por página.')
    for name, result in items:
        if 'features' not in result: continue
        maps = result['features']
        with st.expander(f"{layer_title(name)} · {maps.shape[0]} canales"):
            channel_page = st.selectbox('Grupo de canales', list(range(math.ceil(len(maps) / 8))),
                                        format_func=lambda value: f'{value * 8}–{min(len(maps), (value + 1) * 8) - 1}', key=sample_key+name+'_channels')
            for row_start in range(channel_page * 8, min(len(maps), (channel_page + 1) * 8), 4):
                columns = st.columns(4)
                for column, channel in zip(columns, range(row_start, min(len(maps), row_start + 4, (channel_page + 1) * 8))):
                    normalized, constant = scale_map(torch.as_tensor(maps[channel]))
                    column.image(resize_pixels(normalized, 128), caption=f'Canal {channel}'+(' · constante' if constant else ''), width='stretch')
    with st.expander('Gradiente de una neurona y filtros aprendidos'):
        layer = st.selectbox('Capa para inspección detallada', list(results), format_func=layer_title, key=prefix+'_detail_layer')
        if 'features' not in results[layer]: return
        maps = results[layer]['features']; detail_key = sample_key+layer
        columns = st.columns(3)
        channel = columns[0].number_input('Canal de la neurona', 0, maps.shape[0]-1, 0, key=detail_key+'_channel')
        row = columns[1].number_input('Fila de la neurona', 0, maps.shape[1]-1, 0, key=detail_key+'_row')
        col = columns[2].number_input('Columna de la neurona', 0, maps.shape[2]-1, 0, key=detail_key+'_col')
        if st.checkbox('Calcular sensibilidad de esta neurona', key=prefix+'_neuron_enabled'):
            result = explain(model, data.normalized([index]), layer, channel=channel, row=row, col=col, method='neuron')
            st.image(overlay_image(original, result['heat']), caption='Gradiente absoluto respecto de la entrada · no es Grad-CAM', width=256)
            if result['constant']: st.caption('Gradiente constante/cero para esta unidad.')
        weights = model.features.get_submodule(layer).weight.detach().cpu().numpy()
        input_channel = st.number_input('Canal de entrada del filtro', 0, weights.shape[1]-1, 0, key=prefix+layer+'_filter_input')
        columns = st.columns(4)
        for index_filter, weight in enumerate(weights[:8, input_channel]):
            span = max(float(abs(weight).max()), 1e-9)
            columns[index_filter % 4].image(resize_pixels((weight/span+1)/2, 128), caption=f'Filtro {index_filter}', width='stretch')
        st.caption('Hasta 8 filtros; gris medio = 0, claro = positivo, oscuro = negativo. Cada filtro abarca todos sus canales de entrada.')
