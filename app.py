"""A local scientific notebook, backed by real CPU training."""
import io
import json
import copy
import html
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from cnn_lab.model import Config, Block, Conv, CNN, DenseLayer, ACTIVATIONS, DATASET_SPECS, dataset_spec
from cnn_lab.data import prepare, augment, display_image
from cnn_lab.inference import classify_latest
from cnn_lab.training import train
from cnn_lab.cross_validation import cross_validate
from cnn_lab.visualization import architecture_figure, neuron_figure
from cnn_lab.metrics import evaluate
from cnn_lab.explain import explain, feature_maps
from cnn_lab.live import Monitor,TERMINAL

st.set_page_config(page_title="CNN · Cuaderno experimental",layout="wide")
st.markdown("<style>"+Path("ui/workbench.css").read_text()+"</style>",unsafe_allow_html=True)
st.caption("CNN / CUADERNO EXPERIMENTAL · CPU LOCAL")
st.title("Laboratorio de redes convolucionales")
st.caption("Configurá → entrená → contrastá tus hipótesis. Datos, métricas y atribuciones en un mismo cuaderno.")

def section(title,description=""):
    st.markdown(f'<div class="lab-section"><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p></div>',unsafe_allow_html=True)

def summary(items):
    rows=''.join(f'<div><dt>{html.escape(str(label))}</dt><dd>{html.escape(str(value))}</dd></div>' for label,value in items)
    st.markdown(f'<dl class="run-summary">{rows}</dl>',unsafe_allow_html=True)

def history_figure(history,keys,best_epoch,stopped_early=False):
    fig=go.Figure()
    for key in keys:
        fig.add_trace(go.Scatter(x=[row["epoch"] for row in history],y=[row[key] for row in history],mode="lines+markers",name=key))
    last=history[-1]["epoch"]
    fig.add_vline(x=best_epoch,line_dash="dash",line_color="#087e83",annotation_text="Mejor checkpoint"+(" · parada" if stopped_early and last==best_epoch else ""))
    if stopped_early and last!=best_epoch:
        fig.add_vline(x=last,line_dash="dot",line_color="#a75d42",annotation_text="Parada por paciencia",annotation_position="bottom right")
    fig.update_layout(height=290,margin=dict(l=20,r=20,t=35,b=35),xaxis_title="Época",xaxis=dict(dtick=1),legend=dict(orientation="h",y=-.2),paper_bgcolor="#F7F5EF",plot_bgcolor="#F7F5EF")
    return fig

if "runs" not in st.session_state: st.session_state.runs=[]
if "base" not in st.session_state: st.session_state.base=Config().dict()
if st.session_state.get("view")=="live":
    st.sidebar.empty()
    st.sidebar.caption("Ejecución en vivo · editor oculto hasta finalizar o informar un error.")
    st.header("Entrenamiento en vivo")
    snapshot=st.session_state["live_training"]
    frozen=snapshot["config"]
    st.caption(f"Red activa: {frozen['dataset']} · {dataset_spec(frozen['dataset'])[2]} clases · lr {frozen['lr']:.5f}. Último modelo completado no cambia si esta ejecución falla.")
    monitor=Monitor("live_training",frozen)
    pending=st.session_state.pop("pending_training",None)
    if pending is not None:
        try:
            run=train(Config.from_dict(pending),monitor.batch,on_event=monitor.event)
            st.session_state.completed_runs=st.session_state.get("completed_runs",len(st.session_state.runs))+1
            st.session_state.runs.append(run);st.session_state.runs=st.session_state.runs[-6:]
            monitor.event({"phase":"early_stopped" if run["stopped_early"] else "completed","message":f"Entrenamiento finalizado · {run['epochs_completed']} de {run['max_epochs']} épocas. "+("Parada por paciencia. " if run['stop_reason']=='patience' else "Máximo alcanzado. ")+f"Mejor checkpoint: época {run['best_epoch']}.","epochs_completed":run['epochs_completed'],"best_epoch":run['best_epoch']})
        except Exception as error:monitor.event({"phase":"error","message":str(error)})
    elif snapshot["phase"] not in TERMINAL:
        monitor.event({"phase":"error","message":"La ejecución fue interrumpida por una recarga/cambio de archivo. No se reinicia automáticamente; la última CNN completada se conserva. Volvé a configuración para un intento explícito. Si había una descarga antigua, puede seguir en curso y no se sobrescribirá."})
    if st.session_state["live_training"]["phase"] in TERMINAL:
        st.caption("Clasificador cerrado: la CNN sólo reconoce sus clases entrenadas. Consultá Test / Probar imagen al volver.")
        if st.button("Volver a configuración",width="stretch"):
            st.session_state.view="config"
            st.rerun()
    else:st.caption("Ejecución en curso; los controles de edición están ocultos. No hay pausa/reanudación. Una recarga no vuelve a lanzar el trabajo automáticamente.")
    st.stop()

base = Config.from_dict(st.session_state.base)

with st.sidebar.container():
    st.header("Presets de configuración")
    preset=st.selectbox("Configuración inicial",["CNN base","Sin pooling","Promedio global","Sobreajuste","Con augmentation","Animales CIFAR-10"])
    if st.button("Cargar configuración"):
        c=Config()
        if preset=="Sin pooling":
            for b in c.blocks: b.pooling="none"
        elif preset=="Promedio global": c.head="gap"; c.dense=[]
        elif preset=="Sobreajuste": c.samples=300; c.epochs=25; c.dense=[DenseLayer(128,dropout=0),DenseLayer(64,dropout=0)]
        elif preset=="Con augmentation": c.augment=True
        elif preset=="Animales CIFAR-10": c.dataset="CIFAR10Animals"; c.samples=3000; c.epochs=10
        st.session_state.base=c.dict()
        # Clear editable widget state so the loaded preset takes effect.
        for key in list(st.session_state):
            if key.startswith("cfg_"): del st.session_state[key]
        st.rerun()
    st.caption("Recetas de configuración, NO redes preentrenadas.")
    with st.expander("Qué carga cada preset"):
        st.caption("Cargar reemplaza arquitectura, hiperparámetros y dataset del editor; no tus resultados anteriores. Los pesos se inicializan al azar al entrenar.")
        st.markdown("**CNN base:** Digits, 2 bloques (16→32), max-pool, densa64/ReLU/dropout0.1, Adam0.003, 8 épocas.\n\n**Sin pooling:** base sin pooling. **Promedio global:** base con GAP y sin ocultas. **Sobreajuste:** Digits300, densas128→64 sin dropout, 25 épocas. **Con augmentation:** base + affine. **Animales CIFAR-10:** RGB, 3000 muestras, 10 épocas, resto de la base.")
    upload=st.file_uploader("Importar configuración JSON",type="json")
    if upload and st.button("Aplicar JSON"):
        try:
            c=Config.from_dict(json.load(upload)); CNN(c)
            st.session_state.base=c.dict()
            for key in list(st.session_state):
                if key.startswith("cfg_"): del st.session_state[key]
            st.rerun()
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e: st.error(f"JSON inválido: {e}")
    st.divider()
    st.caption("Digits funciona sin internet. MNIST, FashionMNIST y CIFAR-10 se descargan al preparar datos o entrenar.")
    st.caption("Entrenamiento síncrono: progreso real por batch; no hay pausa/reanudación. No cambies controles mientras entrena.")

training_tab,arch_tab,data_tab,explain_tab,test_tab,upload_tab,study_tab=st.tabs(["01 · Entrenamiento","02 · Arquitectura","03 · Datos","04 · Interpretabilidad","05 · Test","06 · Probar imagen","07 · Guía de estudio"])

def select(label,options,value,key):
    return st.selectbox(label,options,index=options.index(value),key="cfg_"+key)
def number(label,lo,hi,value,key,step=1,format=None,help=None):
    return st.number_input(label,min_value=lo,max_value=hi,value=value,step=step,key="cfg_"+key,format=format,help=help)
def check(label,value,key):
    return st.checkbox(label,value=value,key="cfg_"+key)

with arch_tab:
    st.header("Bloques, mapas y decisiones")
    st.caption("Cada convolución → BatchNorm opcional → activación. Pooling al final del bloque. La salida tiene un logit por clase (6 animales o 10 clases), no probabilidades.")
    section("Estructura global","Cantidad de bloques, reducción espacial y profundidad de la MLP.")
    cols=st.columns(3)
    with cols[0]: count=number("Bloques convolucionales",1,3,len(base.blocks),"blocks")
    with cols[1]: head=select("Reducción final",["flatten","gap"],base.head,"head")
    with cols[2]: hidden_count=number("Capas densas ocultas · profundidad",0,5,len(base.dense),"hidden_count")
    blocks=[]
    for i in range(count):
        b=base.blocks[i] if i<len(base.blocks) else Block()
        with st.expander(f"Bloque convolucional {i+1}",expanded=True):
            n=number("Cantidad de convoluciones en este bloque",1,3,len(b.layers),f"{i}_n",help="Cada convolución tiene su propio conjunto de filtros, seguido de BN opcional y activación.")
            layers=[]
            for j in range(n):
                l=b.layers[j] if j<len(b.layers) else Conv(b.layers[-1].channels)
                section(f"Convolución {j+1} · filtros","Cada filtro recorre todos los canales de entrada y produce un mapa de salida.")
                cols=st.columns(3)
                with cols[0]: channels=number("Cantidad de filtros / mapas de salida",1,64,l.channels,f"{i}_{j}_ch",help="16 filtros producen 16 canales de salida. No es el tamaño espacial del filtro.")
                with cols[1]: kernel=number("Lado del filtro · k × k",1,7,l.kernel,f"{i}_{j}_k",help="Escribí el lado entero de un filtro CUADRADO: 3 significa 3×3, no tres filtros. Admite1–7, incluidos tamaños pares.")
                with cols[2]: stride=number("Paso de convolución · stride",1,2,l.stride,f"{i}_{j}_s",help="Avance del filtro en píxeles: 1 o 2.")
                cols=st.columns(2)
                with cols[0]: padding=number("Relleno por borde · padding",0,3,l.padding,f"{i}_{j}_p",help="Píxeles de cero añadidos a cada borde. Rango0–3.")
                with cols[1]: dilation=number("Separación del kernel · dilatación",1,2,l.dilation,f"{i}_{j}_d",help="1: posiciones contiguas; 2: deja una separación entre pesos.")
                st.caption(f"{channels} filtros espaciales de {kernel}×{kernel} → {channels} mapas de salida. Stride {stride}, padding {padding}, dilatación {dilation}. El número de canales de entrada lo fija la capa anterior.")
                layers.append(Conv(channels,kernel,stride,padding,dilation))
            section("Después de cada convolución","Transformaciones de sus mapas; no agregan filtros convolucionales.")
            cols=st.columns(2)
            with cols[0]: bn=check("Aplicar BatchNorm a los mapas",b.batchnorm,f"{i}_bn")
            with cols[1]: activation=select("Activación de los mapas",list(ACTIVATIONS),b.activation,f"{i}_act")
            section("Reducción espacial al finalizar el bloque","Pooling resume ventanas espaciales; mantiene la cantidad de canales.")
            cols=st.columns(2)
            with cols[0]: pool=select("Tipo de pooling",["none","max","avg"],b.pooling,f"{i}_pool")
            ps=b.pool_size
            if pool!="none":
                with cols[1]: ps=number("Lado de ventana · p × p",2,4,b.pool_size,f"{i}_ps",help="2 significa ventana2×2. Stride=lado de ventana; sin padding. Se aplica una vez al final del bloque.")
                st.caption(f"{pool} pooling: ventana {ps}×{ps}, stride {ps}, sin padding. Reduce alto/ancho; conserva {channels} canales del último mapa convolucional.")
            else: st.caption("Sin pooling: no se resume ni reduce el mapa al finalizar este bloque.")
            blocks.append(Block(layers,bn,activation,pool,ps))
    st.subheader("MLP · capas densas ocultas y salida")
    st.caption("Una MLP puede tener varias capas ocultas. Cada densa tiene sus propios pesos, activación y dropout. La salida final es otra capa densa: un logit por clase, sin softmax antes de CrossEntropyLoss.")
    dense=[]
    for i in range(hidden_count):
        layer=base.dense[i] if i<len(base.dense) else DenseLayer()
        with st.expander(f"Densa oculta {i+1}",expanded=True):
            cols=st.columns(4)
            with cols[0]: units=number("Neuronas / unidades",4,256,layer.units,f"dense_units_{i}")
            with cols[1]: act=select("Activación de esta capa",list(ACTIVATIONS),layer.activation,f"dense_activation_{i}")
            with cols[2]: drop=number("Dropout de esta capa",0.,.7,layer.dropout,f"dense_dropout_{i}",.05,format="%.2f")
            with cols[3]: bias=check("Bias / sesgo",layer.bias,f"dense_bias_{i}")
            dense.append(DenseLayer(units,act,drop,bias))
    if not dense: st.caption("Sin ocultas: Flatten/GAP → capa densa de salida. Es un clasificador lineal sobre las características convolucionales.")

with data_tab:
    st.header("Datos pequeños, preguntas grandes")
    section("Dataset y presupuesto","Seleccioná datos pequeños para comparar hipótesis sin esperar entrenamientos largos.")
    cols=st.columns(2)
    def dataset_changed():
        if st.session_state.cfg_dataset == "CIFAR10Animals": st.session_state.cfg_samples = 3000
    with cols[0]: dataset=st.selectbox("Dataset",list(DATASET_SPECS),index=list(DATASET_SPECS).index(base.dataset),key="cfg_dataset",format_func=lambda name:"CIFAR-10 · solo animales" if name=="CIFAR10Animals" else name,on_change=dataset_changed)
    with cols[1]: samples=number("Máximo de muestras",200,10000,base.samples,"samples",100)
    section("Preprocesado de train","Las estadísticas se aprenden sólo en train; augmentation nunca se aplica a val/test.")
    cols=st.columns(2)
    with cols[0]: normalize=check("Normalizar con estadísticas de train",base.normalize,"normalize")
    with cols[1]: augmentation=check("Augmentation sólo en train",base.augment,"augment")
    st.markdown("""| Dataset | Escala | Qué permite aprender |
|---|---|---|
| **Digits** | 1797 imágenes de 8×8, offline | Iteración rápida, kernels, pooling y métricas |
| **MNIST** | 28×28, subconjunto configurable | Bordes, formas y campos receptivos más legibles |
| **FashionMNIST** | 28×28, subconjunto configurable | Clases más ambiguas; errores y generalización |
| **CIFAR-10 · solo animales** | RGB 32×32, 6 clases | Pájaro, gato, ciervo, perro, rana y caballo; fotos más difíciles |""")
    if dataset == "CIFAR10Animals":
        st.caption("Se filtran IDs originales 2–7 antes del muestreo: no se entrenan vehículos. El archivo fuente completo (~163 MB) se descarga una vez y queda en data/. 3000 muestras seleccionadas ≈500 por clase ANTES de reservar 20% de validación: ≈400 train y 100 val por clase. Test se toma sólo del test oficial de animales.")
    st.caption("Digits: 70% train / 15% validación / 15% test, estratificados y disjuntos. MNIST/Fashion/CIFAR: test oficial independiente; 80/20 dentro del subconjunto oficial de entrenamiento. Nunca se calcula media/desvío sobre test.")

with training_tab:
    st.header("Entrenamiento")
    controls,launch=st.columns([1.55,1],gap="large")
    with controls:
        section("Plan de entrenamiento","Una época recorre train; cada batch produce una actualización de pesos.")
        cols=st.columns(2)
        with cols[0]: epochs=number("Máximo de épocas",1,30,base.epochs,"epochs",help="1–30. Con parada temprana puede terminar antes.")
        with cols[1]: batch_size=select("Batch size · muestras",[16,32,64,128,256],base.batch_size,"batch_size")
        section("Optimización","Elegí cómo se actualizan los parámetros y la magnitud de cada paso.")
        cols=st.columns(2)
        with cols[0]: optimizer=select("Optimizador",["Adam","AdamW","SGD"],base.optimizer,"optimizer")
        with cols[1]: lr=number("Learning rate",.00001,.1,base.lr,"lr",.001,format="%.5f",help="0.00001–0.1. El valor se muestra con cinco decimales; 0.003 no es cero.")
        momentum=base.momentum
        if optimizer=="SGD": momentum=number("Momentum · SGD",0.,.99,base.momentum,"momentum",.05,format="%.2f",help="0 desactiva la acumulación de dirección; rango 0–0.99.")
        section("Regularización y reproducibilidad","Weight decay limita magnitudes de pesos; la semilla fija inicialización y orden de batches.")
        cols=st.columns(2)
        with cols[0]: wd=number("Weight decay",0.,.1,base.weight_decay,"wd",.001,format="%.5f")
        with cols[1]: seed=number("Semilla",0,2**32-1,base.seed,"seed")
        section("Parada temprana · opcional","Vigila loss de validación. Guardar el mejor checkpoint y decidir cuándo detenerse son dos reglas distintas.")
        early=check("Activar parada temprana",base.early_stopping,"early_stopping")
        patience,min_delta=base.patience,base.min_delta
        if early:
            cols=st.columns(2)
            with cols[0]: patience=number("Paciencia · épocas",1,30,base.patience,"patience",help="Detiene tras N épocas consecutivas sin mejora significativa. La primera fija la referencia.")
            with cols[1]: min_delta=number("Mejora mínima · loss",0.,10.,float(base.min_delta),"min_delta",.0001,format="%.5f",help="La reducción debe ser estrictamente MAYOR a min_delta. Igualdad no reinicia paciencia.")
    with launch:
        launch_panel=st.container(key="launch_panel")

config=Config(blocks=blocks,head=head,dense=dense,dataset=dataset,samples=samples,seed=seed,epochs=epochs,batch_size=batch_size,optimizer=optimizer,lr=lr,weight_decay=wd,momentum=momentum,augment=augmentation,normalize=normalize,early_stopping=early,patience=patience,min_delta=min_delta)
valid=True
invalid_reason=""
try:
    preview=CNN(config)
except (ValueError,RuntimeError) as e:
    valid=False
    invalid_reason=str(e)
    with arch_tab: st.error(str(e))
with arch_tab:
    if valid:
        st.metric("Parámetros entrenables",f"{sum(p.numel() for p in preview.parameters()):,}")
        st.dataframe(pd.DataFrame(preview.rows).fillna("—").astype({"RF teórico":str,"salto":str}),hide_index=True,width="stretch")
        st.subheader("Arquitectura · nodos y conexiones")
        diagram_view=st.radio("Vista del diagrama",["Bloques CNN + MLP","Operaciones detalladas","Neuronas de la MLP"],horizontal=True,key="architecture_view")
        figure=neuron_figure(preview) if diagram_view=="Neuronas de la MLP" else architecture_figure(preview,detailed=diagram_view=="Operaciones detalladas")
        st.caption("Arrastrá para mover, usá la rueda para zoom y pasá el cursor sobre un nodo para inspeccionarlo. La barra superior permite restaurar la vista y descargar PNG. Los diagramas son esquemas funcionales, no posiciones físicas de memoria/GPU.")
        if figure.layout.height>650:
            st.caption("Diagrama vertical: desplazá este panel para recorrer todas las capas sin achicar sus etiquetas.")
            with st.container(height=650,border=False):
                st.plotly_chart(figure,width="stretch",config={"scrollZoom":True,"displayModeBar":True,"displaylogo":False},key="architecture_plot")
        else:
            st.plotly_chart(figure,width="stretch",config={"scrollZoom":True,"displayModeBar":True,"displaylogo":False},key="architecture_plot")
        if diagram_view=="Neuronas de la MLP":
            st.caption("Cada punto representa una unidad real seleccionada de esa capa (índice en tooltip). Mostramos hasta 12 por capa oculta/vector, con el total real; mostramos TODAS las clases de salida. Las líneas son conexiones densas entre unidades mostradas: existen también conexiones a unidades omitidas. Activación y dropout no agregan neuronas. La CNN se resume en su vector de características; sus convoluciones NO son fully connected. Softmax se usa sólo al mostrar probabilidades, no como capa antes de CrossEntropyLoss.")
        else:
            st.caption("Nodos = bloques u operaciones, no neuronas individuales. Conexiones = flujo de datos de arriba hacia abajo. Colores: entrada gris · conv teal · activación terracota · BN ocre · dropout mostaza · pooling verde · vector azul gris · densa azul · logits rosa oscuro. Cambiá a Neuronas de la MLP para ver conectividad densa.")
        st.caption("RF teórico mide extensión en píxeles de entrada; incluye padding y puede superar la imagen. No equivale a sensibilidad efectiva.")
    st.download_button("Exportar configuración JSON",json.dumps(config.dict(),indent=2),"cnn-config.json","application/json")

with data_tab:
    section("Inspección antes de entrenar","Comprobá clases, particiones y transformaciones sobre ejemplos reales.")
    if st.button("Preparar y visualizar datos",disabled=not valid,width="stretch"):
        try:
            st.session_state.pop("data_preparation",None)
            data_monitor=Monitor("data_preparation",config.dict())
            data_monitor.event({"phase":"preparing","message":"Solicitud recibida: preparando dataset antes de generar muestras"})
            st.session_state.preview_data=prepare(config,on_event=data_monitor.event)
            st.session_state.preview_config=config.dict()
            data_monitor.event({"phase":"completed","message":"Dataset listo. Las muestras se muestran debajo."})
        except Exception as e:
            data_monitor.event({"phase":"error","message":f"No se pudo preparar el dataset: {e}. Elegí Digits para trabajar offline."})
    if "preview_data" in st.session_state:
        d=st.session_state.preview_data
        st.caption(f"Última preparación completada · {st.session_state.get('preview_config',{}).get('dataset','dataset previo')} (puede diferir del editor): {len(d.train)} train / {len(d.val)} val / {len(d.test)} test · μ={np.asarray(d.mean).reshape(-1).round(4).tolist()}, σ={np.asarray(d.std).reshape(-1).round(4).tolist()}")
        raw=d.images[d.train[:6]]
        changed=augment(raw,torch.Generator().manual_seed(seed))
        cols=st.columns(6)
        for i,col in enumerate(cols):
            col.image(display_image(raw[i]),caption=f"Original · {d.names[int(d.labels[d.train[i]])]}",clamp=True)
            col.image(display_image(changed[i]),caption="Affine pequeño",clamp=True)
        st.caption("Vista ilustrativa de augmentation (rotación ±7.5°, traslación ±7.5%). Sólo se aplica al entrenar si está activado. Se transforma antes de normalizar; BatchNorm actúa dentro de la red, no reemplaza este paso.")

with launch_panel:
    st.subheader("Tu próximo experimento")
    size,channels,classes=dataset_spec(dataset)
    summary([("Datos",dataset),("Entrada",f"{channels} × {size} × {size} · {classes} clases"),("Arquitectura",f"{len(blocks)} bloques · {len(dense)} densas ocultas"),("Optimización",f"{optimizer} · lr {lr:.5f}"),("Plan",f"batch {batch_size} · hasta {epochs} épocas"),("Parada",f"paciencia {patience} · Δ>{min_delta:.5f}" if early else "Desactivada")])
    if valid: st.caption(f"{sum(p.numel() for p in preview.parameters()):,} parámetros · checkpoint por menor loss de validación.")
    else: st.error(f"Revisá Arquitectura: {invalid_reason}")
    def queue_training():
        import time
        frozen=copy.deepcopy(config.dict())
        st.session_state.base=frozen
        st.session_state.live_training={"phase":"queued","config":frozen,"started":time.monotonic(),"history":[],"batches":[]}
        st.session_state.pending_training=frozen
        st.session_state.view="live"
    if st.button("Entrenar CNN",type="primary",disabled=not valid,width="stretch"):
        queue_training()
        st.rerun()
    st.caption("Digits: sin descargas." if dataset=="Digits" else "La primera preparación descarga el dataset y lo conserva localmente. CIFAR-10: archivo completo ~163 MB.")
    st.caption("CPU local · entrenamiento síncrono, sin pausa/reanudación. No cambies controles mientras entrena.")

with training_tab:
    section("Seguimiento y resultados","Los resultados usan la configuración congelada de cada ejecución, no los controles que edites después.")
    if "live_training" in st.session_state:
        st.caption("El último monitor queda guardado. No inicia otro entrenamiento.")
        if st.button("Ver último entrenamiento en vivo"):
            st.session_state.view="live";st.rerun()
    if not st.session_state.runs:st.caption("Aún no hay ejecuciones; Entrenar CNN abre una vista dedicada antes de preparar datos.")

run=None
if st.session_state.runs:
    with training_tab:
        run_labels=[f"{i+1} · {r['config']['dataset']} / {r['config']['optimizer']}" for i,r in enumerate(st.session_state.runs)]
        selected=st.selectbox("Experimento para inspeccionar",list(range(len(run_labels))),index=len(run_labels)-1,format_func=lambda i:run_labels[i])
        run=st.session_state.runs[selected]
        if Config.from_dict(run["config"]).dict()!=config.dict(): st.warning("Los controles actuales no coinciden con este experimento. Los resultados y mapas usan su configuración congelada, no los controles editados.")
        st.caption(f"Completadas {run.get('epochs_completed',len(run['history']))} de {run.get('max_epochs',run['config']['epochs'])} épocas · mejor checkpoint {run['best_epoch']} · motivo {"paciencia" if run.get('stop_reason')=='patience' else "máximo de épocas"}")
        st.json(run["config"],expanded=False)
        cols=st.columns(2)
        h=pd.DataFrame(run["history"]).set_index("epoch")
        cols[0].plotly_chart(history_figure(run["history"],["train_loss","val_loss"],run["best_epoch"],run.get("stopped_early",False)),width="stretch")
        cols[1].plotly_chart(history_figure(run["history"],["train_accuracy","val_accuracy"],run["best_epoch"],run.get("stopped_early",False)),width="stretch")
        st.caption("Train se mide durante actualizaciones, con dropout/augmentation activos; validación se mide al final en eval. No son condiciones idénticas.")
        with st.expander("Traza real batch a batch"):
            st.line_chart(pd.DataFrame(run["batches"]).set_index("step")[["loss"]]); st.dataframe(run["batches"],hide_index=True)
        comparison=[{"experimento":i+1,"dataset":r["config"]["dataset"],"optimizador":r["config"]["optimizer"],"mejor época":r["best_epoch"],"mín loss val":min(x["val_loss"] for x in r["history"]),"segundos":r["elapsed"]} for i,r in enumerate(st.session_state.runs)]
        st.dataframe(comparison,hide_index=True)
        st.download_button("Descargar logs CSV",pd.DataFrame(run["batches"]).to_csv(index=False),"batches.csv","text/csv")
        st.download_button("Descargar épocas CSV",pd.DataFrame(run["history"]).to_csv(index=False),"epochs.csv","text/csv")
        buffer=io.BytesIO()
        torch.save({"format_version":4,"input_shape":list(run["data"].images.shape[1:]),"source_indices":run["data"].source_indices,"state_dict":run["model"].state_dict(),"config":Config.from_dict(run["config"]).dict(),"mean":run["data"].mean,"std":run["data"].std,"classes":run["data"].names,"best_epoch":run["best_epoch"],"epochs_completed":run.get("epochs_completed",len(run["history"])),"max_epochs":run.get("max_epochs",run["config"]["epochs"]),"stopped_early":run.get("stopped_early",False),"stop_reason":run.get("stop_reason","max_epochs"),"split_indices":{"train":run["data"].train.tolist(),"val":run["data"].val.tolist(),"test":run["data"].test.tolist()},"history":run["history"]},buffer)
        st.download_button("Descargar checkpoint + metadatos",buffer.getvalue(),"cnn-best.pt","application/octet-stream")
        st.caption("Historial: últimos 6 experimentos de esta sesión en memoria. Descargá artefactos antes de cerrar/reiniciar. Checkpoint para inferencia, no para reanudar optimizer.")

with training_tab:
    st.divider()
    section("Validación K-fold estratificada","Comparación independiente del split: acción separada del entrenamiento para inferencia.")
    st.write("Repartí el conjunto de desarrollo (train + validación originales) en K folds. Cada muestra valida exactamente una vez. Test queda separado y nunca se evalúa en esta acción.")
    cols=st.columns(2)
    with cols[0]: cv_k=st.selectbox("Cantidad de folds K",[2,3,5],index=1,key="cv_k")
    with cols[1]: cv_epochs=st.number_input("Épocas por fold",1,30,min(epochs,3),key="cv_epochs")
    st.caption(f"Coste: {cv_k} entrenamientos independientes × {cv_epochs} épocas= hasta {cv_k*cv_epochs} épocas en CPU. Parada temprana: {"activada, cada fold puede terminar antes" if early else "desactivada"}. Se carga el dataset una vez; en datasets no cacheados puede descargarse. Cada fold crea pesos y optimizador nuevos. No reemplaza la última CNN para Probar imagen; no guarda modelos de folds ni crea un ensemble.")
    cv_progress=st.empty(); cv_live=st.empty(); cv_chart=st.empty()
    if st.button("Ejecutar validación K-fold",disabled=not valid):
        def cv_callback(fold,trace,history,batches,total):
            cv_progress.progress(((fold-1)*cv_epochs*total+(trace["epoch"]-1)*total+trace["batch"])/(cv_k*cv_epochs*total))
            cv_live.caption(f"Fold {fold}/{cv_k} · época {trace['epoch']}/{cv_epochs} · batch {trace['batch']}/{total} · loss {trace['loss']:.4f}")
            if history and (trace.get("epoch_complete") or trace["batch"]%5==0):
                cv_chart.line_chart(pd.DataFrame(history).set_index("epoch")[["train_loss","val_loss"]])
        try:
            st.session_state.cv_result=cross_validate(config,cv_k,cv_epochs,cv_callback)
            cv_progress.empty()
            st.success("K-fold finalizado. Test no fue evaluado; los entrenamientos normales anteriores siguen intactos.")
        except Exception as error: st.error(f"K-fold detenido: {error}")
    if "cv_result" in st.session_state:
        cv=st.session_state.cv_result
        if cv["config"]!={**config.dict(),"epochs":cv_epochs} or cv["k"]!=cv_k:
            st.warning("Los controles actuales no coinciden con esta validación K-fold. Se muestran los resultados de su configuración congelada.")
        st.caption(f"Resultado guardado: K={cv['k']} · {cv['epochs']} épocas/fold · {cv['elapsed']:.1f}s · {len(cv['heldout_test_indices'])} ejemplos test reservados sin evaluar")
        columns=["fold","seed","train_count","val_count","accuracy","precision","recall","f1","val_loss","best_epoch","epochs_completed","max_epochs","stopped_early","stop_reason","elapsed"]
        table=pd.DataFrame(cv["folds"]).reindex(columns=columns)
        st.dataframe(table,hide_index=True,width="stretch")
        summary=pd.DataFrame(cv["aggregate"]).T.rename(columns={"mean":"media","std":"desvío entre folds"})
        st.dataframe(summary,width="stretch")
        st.caption("Precision/recall/F1 macro. Desvío estándar muestral entre folds (ddof=1); no es un intervalo de confianza ni una medida de incertidumbre por imagen. Loss corresponde al mejor checkpoint de cada fold.")
        with st.expander("Curvas por fold, traza y configuración congelada"):
            st.json(cv["config"],expanded=False)
            for fold in cv["folds"]:
                st.markdown(f"**Fold {fold['fold']} · seed {fold['seed']}**")
                cols=st.columns(2)
                frame=pd.DataFrame(fold["history"]).set_index("epoch")
                cols[0].plotly_chart(history_figure(fold["history"],["train_loss","val_loss"],fold["best_epoch"],fold.get("stopped_early",False)),width="stretch")
                cols[1].plotly_chart(history_figure(fold["history"],["train_accuracy","val_accuracy"],fold["best_epoch"],fold.get("stopped_early",False)),width="stretch")
                st.line_chart(pd.DataFrame(fold["batches"]).set_index("step")[["loss"]])
        st.download_button("Descargar resultados K-fold CSV",table.to_csv(index=False),"kfold-resultados.csv","text/csv")
        st.download_button("Descargar K-fold completo JSON",json.dumps(cv,indent=2),"kfold-config-logs.json","application/json")
    st.caption("Cada fold calcula normalización sólo con su train; augmentation sólo allí. Seleccionamos checkpoint con validación del mismo fold: es una validación práctica, no nested CV ni una estimación totalmente libre de selección. Usá estos resultados para comparar configuraciones, luego entrená una CNN normal para inferencia y reservá test para la evaluación final.")

with test_tab:
    st.header("Generalización, no memorización")
    if run is None: st.info("Entrená primero. Test permanece fuera de las actualizaciones y de la selección del checkpoint.")
    else:
        d=run["data"]; probabilities=run["probabilities"]; labels=d.labels[d.test].numpy(); m=evaluate(labels,probabilities)
        st.caption(f"Evaluación del mejor checkpoint (época {run['best_epoch']}) sobre {len(labels)} ejemplos held-out.")
        cols=st.columns(5)
        for col,key in zip(cols,["accuracy","precision","recall","f1","auc"]): col.metric(key.upper()+ (" macro" if key not in ("accuracy",) else ""),f"{m[key]:.3f}" if m[key] is not None else "Indefinido")
        section("Rendimiento por clase","Precision, recall y F1 muestran errores que accuracy puede ocultar.")
        st.dataframe(pd.DataFrame(m["per_class"],index=d.names),width="stretch")
        section("Matriz de confusión","Filas: clase real. Columnas: predicción.")
        normalized=st.checkbox("Normalizar matriz por clase real")
        cm=m["confusion"].astype(float)
        if normalized: cm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)!=0)
        fig=go.Figure(go.Heatmap(z=cm,x=d.names,y=d.names,colorscale="Teal",text=cm.round(2),texttemplate="%{text}"))
        fig.update_layout(xaxis_title="Predicción",yaxis_title="Clase real",yaxis_autorange="reversed")
        st.plotly_chart(fig,width="stretch")
        section("ROC · one-vs-rest","Curvas de umbral sobre probabilidades; no sobre etiquetas discretas.")
        fig=go.Figure()
        for i,(fpr,tpr,score) in m["curves"].items(): fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{d.names[i]} · AUC {score:.3f}"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Azar",line=dict(dash="dash",color="gray")))
        fig.update_layout(xaxis_title="False positive rate",yaxis_title="True positive rate")
        st.plotly_chart(fig,width="stretch")
        st.caption("ROC one-vs-rest: umbrales sobre probabilidades, no sobre etiquetas. AUC macro es promedio de AUC por clase; clases sin positivos/negativos se omiten y su ROC es indefinida.")
        errors=np.flatnonzero(probabilities.argmax(1)!=labels)
        st.subheader(f"Galería de errores · {len(errors)}")
        cols=st.columns(6)
        for index,i in enumerate(errors[:18]):
            cols[index%6].image(display_image(d.images[d.test[i]]),caption=f"Real {d.names[labels[i]]} → {d.names[probabilities[i].argmax()]} ({probabilities[i].max():.0%})",clamp=True)
        st.warning("Si ajustás repetidamente la arquitectura mirando test, deja de ser una evaluación independiente. Decidí con validación; reservá test para el resultado final.")

with explain_tab:
    st.header("Qué ve la red y a qué es sensible")
    if run is None: st.caption("Los filtros, mapas y atribuciones estarán disponibles tras entrenar.")
    else:
        model=run["model"]; d=run["data"]
        section("Muestra, capa y objetivo","Elegí qué entrada y clase querés analizar antes de interpretar el mapa.")
        cols=st.columns(3)
        with cols[0]: sample=st.number_input("Índice de muestra test",0,len(d.test)-1,0)
        with cols[1]: layer=st.selectbox("Capa convolucional",model.conv_names())
        with cols[2]: target=st.selectbox("Clase objetivo",list(range(len(d.names))),format_func=lambda i:d.names[i])
        index=d.test[sample]; x=d.normalized([index]); maps=feature_maps(model,x,layer)
        st.caption(f"Etiqueta real: {d.names[int(d.labels[index])]} · mapas: {maps.shape[0]} × {maps.shape[1]} × {maps.shape[2]}")
        section("Método de atribución","Grad-CAM explica un logit; el gradiente de neurona mide sensibilidad local de una unidad.")
        method=st.radio("Atribución",["Grad-CAM","Gradiente de neurona"],horizontal=True)
        channel=0; row=0; col=0
        if method=="Gradiente de neurona":
            cols=st.columns(3)
            channel=cols[0].number_input("Canal",0,maps.shape[0]-1,0)
            row=cols[1].number_input("Fila",0,maps.shape[1]-1,0)
            col=cols[2].number_input("Columna",0,maps.shape[2]-1,0)
        result=explain(model,x,layer,target,channel,row,col,"gradcam" if method=="Grad-CAM" else "neuron")
        cols=st.columns([1,2])
        cols[0].image(display_image(d.images[index]),caption="Entrada original",clamp=True)
        original=display_image(d.images[index])
        fig=go.Figure(go.Image(z=(original*255).astype(np.uint8))) if original.ndim==3 else go.Figure(go.Heatmap(z=original,colorscale="Greys",showscale=False))
        fig.add_trace(go.Heatmap(z=result["heat"],colorscale="YlOrRd",opacity=.55,zmin=0,zmax=1))
        fig.update_layout(yaxis_autorange="reversed",yaxis_scaleanchor="x",height=360)
        cols[1].plotly_chart(fig,width="stretch")
        if result["constant"]: st.warning("Mapa constante/cero: no hay contraste de atribución en este punto. No implica que todos los píxeles sean igualmente importantes.")
        st.bar_chart(pd.DataFrame({"probabilidad":result["probabilities"]},index=d.names))
        st.caption("Grad-CAM usa gradientes del logit de la clase elegida; el gradiente de neurona usa la salida pre-BN/pre-activación de esta convolución. Es sensibilidad local, no prueba causal. Atribuciones 8×8 son muy gruesas.")
        with st.expander("Mapas de características (hasta 16 canales)",expanded=True):
            cols=st.columns(8)
            for i,feature in enumerate(maps[:16]):
                lo,hi=feature.min(),feature.max(); image=(feature-lo)/(hi-lo) if hi>lo else np.zeros_like(feature)
                cols[i%8].image(image,caption=f"Canal {i}",clamp=True)
            st.caption("Cada mapa se escala por separado: colores similares no significan amplitudes iguales.")
        with st.expander("Filtros aprendidos (hasta 16)"):
            weights=model.features.get_submodule(layer).weight.detach().numpy()
            input_channel=st.number_input("Canal de entrada del filtro",0,weights.shape[1]-1,0)
            cols=st.columns(8)
            for i,w in enumerate(weights[:16,input_channel]):
                span=max(float(abs(w).max()),1e-9)
                cols[i%8].image((w/span+1)/2,caption=f"Filtro {i}",clamp=True)
            st.caption("Gris medio = 0; claro = positivo; oscuro = negativo. Escala simétrica por filtro.")

with upload_tab:
    st.header("Clasificá una imagen nueva")
    st.caption("Siempre usa el último entrenamiento completado y su mejor checkpoint de validación. No usa el experimento antiguo seleccionado ni los controles actuales. No entrena con la imagen subida.")
    if not st.session_state.runs:
        st.info("Primero entrená una red. Para fotos de animales elegí el preset Animales CIFAR-10 y entrenalo.")
    else:
        latest=st.session_state.runs[-1]
        st.caption(f"Red activa: entrenamiento #{st.session_state.get('completed_runs',len(st.session_state.runs))} · {latest['config']['dataset']} · mejor checkpoint época {latest['best_epoch']} · {len(latest['data'].names)} clases")
        section("Probar sin entrenar","PNG, JPEG o WebP · hasta 10 MB y 16 MP. Se usa el último checkpoint completado.")
        uploaded=st.file_uploader("Imagen PNG, JPEG o WebP · hasta 10 MB / 16 MP",type=["png","jpg","jpeg","webp"],key="inference_upload")
        expected=st.selectbox("Clase real esperada (opcional)",["Desconocida / fuera de las clases"]+latest["data"].names,key="expected_upload")
        if uploaded:
            try:
                result=classify_latest(st.session_state.runs,uploaded.getvalue())
                cols=st.columns(2)
                cols[0].image(result["original"],caption="Original · orientación EXIF aplicada; transparencia sobre blanco",width="stretch")
                cols[1].image(display_image(result["input"]),caption=f"Entrada efectiva · {list(result['input'].shape)} (C × H × W), luego normalización congelada de train",clamp=True,width=256)
                predicted=result["names"][result["prediction"]]
                st.subheader(f"Predicción: {predicted}")
                probabilities=pd.DataFrame({"clase":result["names"],"probabilidad softmax":result["probabilities"]}).sort_values("probabilidad softmax",ascending=False)
                st.dataframe(probabilities,hide_index=True,width="stretch")
                st.bar_chart(probabilities.set_index("clase"),sort=False)
                if expected != "Desconocida / fuera de las clases":
                    if expected==predicted: st.success("Coincide con la clase esperada.")
                    else: st.warning(f"No coincide: esperabas {expected}, predijo {predicted}.")
            except ValueError as error: st.error(str(error))
        st.warning("Clasificador cerrado: siempre elige una clase entrenada, incluso si la foto no contiene esos animales. Softmax NO es probabilidad calibrada de acertar ni detector de desconocidos. Las fotos nuevas, fondos y encuadres pueden diferir mucho de CIFAR-10; un gato grande o un animal ajeno puede ser mal clasificado.")
        st.caption("Preprocesado: convertir a RGB o gris según la red, redimensionar bilineal sin recorte a su resolución, dividir por 255 y aplicar media/desvío de su train. No hay augmentation ni ajustes con esta imagen. En Digits se respeta la escala [0,1]; conviene fondo negro y trazos claros.")

with study_tab:
    st.header("Guía de estudio")
    st.caption("Una pregunta → una variable → evidencia. Primero el recorrido corto; después fórmulas y límites.")
    guide=Path("GUIA_ESTUDIO.md").read_text()
    intro,details=guide.split("## Cómo atraviesa la red una imagen",1)
    st.markdown(intro)
    with st.expander("Conceptos, fórmulas y experimentos",expanded=False):
        st.markdown("## Cómo atraviesa la red una imagen"+details)
    st.download_button("Descargar guía de estudio",Path("GUIA_ESTUDIO.md").read_text(),"guia-cnn.md","text/markdown")
