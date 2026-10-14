"""Local Streamlit v2 thumbnail buttons; payloads are bounded to one page."""
import base64
import io
from PIL import Image
import streamlit as st

CSS = """
.gallery {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:16px;font-family:inherit;color:#253836;}
figure {margin:0;min-width:0;}
button {display:block;position:relative;width:100%;padding:0;border:3px solid transparent;border-radius:7px;background:transparent;cursor:pointer;overflow:hidden;min-height:44px;}
button[aria-pressed="true"] {border-color:#087e83;}
button:focus-visible {outline:3px solid #087e83;outline-offset:3px;}
img {display:block;width:100%;aspect-ratio:1;object-fit:contain;image-rendering:pixelated;background:#ecece4;}
.badge {position:absolute;right:5px;top:5px;background:#087e83;color:#fff;border-radius:50%;width:26px;height:26px;display:grid;place-items:center;font-weight:700;}
figcaption {font-size:14px;line-height:1.5;margin-top:5px;overflow-wrap:anywhere;}
.real {font-weight:600;}
@media(max-width:640px) {.gallery {grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}}
@media(hover:hover) and (pointer:fine) {button {transition:transform 160ms cubic-bezier(.23,1,.32,1);}button:hover {border-color:#859b91;}button:active {transform:scale(.97);}}
@media(prefers-reduced-motion:reduce) {button {transition:none!important;transform:none!important;}}
"""
JS = """
export default function({data,parentElement,setTriggerValue}) {
  const root=parentElement.querySelector('.gallery');
  root.replaceChildren();
  for (const item of data.items) {
    const figure=document.createElement('figure');
    const button=document.createElement('button');
    button.type='button';
    button.setAttribute('aria-label',item.label);
    button.setAttribute('aria-pressed',String(item.index===data.selected));
    const image=document.createElement('img');
    image.src=item.image;
    image.alt=item.label;
    button.append(image);
    if (item.index===data.selected) {
      const badge=document.createElement('span');badge.className='badge';badge.textContent='✓';badge.setAttribute('aria-hidden','true');button.append(badge);
    }
    button.onclick=()=>setTriggerValue('selected',{index:item.index,token:data.token});
    const caption=document.createElement('figcaption');
    const real=document.createElement('div');real.className='real';real.textContent=`Imagen ${item.index+1} · Real: ${item.actual}`;
    const predicted=document.createElement('div');predicted.textContent=`Pred.: ${item.predicted} · ${item.confidence}`;
    caption.append(real,predicted);figure.append(button,caption);root.append(figure);
  }
  return ()=>root.replaceChildren();
}
"""
_thumbnail_component = st.components.v2.component('test_image_gallery', html='<div class="gallery" role="group" aria-label="Seleccionar imagen de test"></div>', css=CSS, js=JS)


def gallery_payload(data, probabilities, start, end, selected, token):
    if not 0 <= end - start <= 10:
        raise ValueError('La galería admite hasta10 imágenes por página.')
    from .data import display_image
    from .interpretability import resize_pixels
    items = []
    for sample in range(start, end):
        source = data.test[sample]
        predicted = int(probabilities[sample].argmax())
        actual_name = data.names[int(data.labels[source])]
        predicted_name = data.names[predicted]
        confidence = f'{probabilities[sample,predicted]:.1%}'
        buffer = io.BytesIO()
        Image.fromarray(resize_pixels(display_image(data.images[source]), 160)).save(buffer,format='PNG')
        items.append(dict(index=sample, actual=actual_name, predicted=predicted_name, confidence=confidence,
                          label=f'Seleccionar imagen {sample+1}. Real: {actual_name}. Predicción: {predicted_name}, {confidence}.',
                          image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')))
    return dict(items=items, selected=selected, token=token)


def apply_selection(state, event, token, start, end):
    """Ignore stale cross-page/run events, booleans, and indices outside this page."""
    if not isinstance(event,dict) or event.get('token') != token:
        return False
    index = event.get('index')
    if type(index) is not int or not start <= index < end or index == state['sample']:
        return False
    state['sample'] = index
    return True


def gallery_widget(payload, key):
    return _thumbnail_component(data=payload, key=key, on_selected_change=lambda: None).selected
