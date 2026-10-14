"""Synchronous Streamlit monitor backed only by observed training/data events."""
import copy
import time
import plotly.graph_objects as go
import streamlit as st

PHASES={'queued':'Solicitud recibida','preparing':'Preparando datos','cache_check':'Verificando caché','cached':'Leyendo caché verificada','downloading':'Descargando dataset','verifying':'Verificando integridad','extracting':'Extrayendo datos','reading':'Leyendo datos','prepared':'Datos preparados','initializing':'Inicializando CNN','training':'Entrenando','validation':'Validando','test':'Evaluando test','completed':'Completado','early_stopped':'Parada temprana','error':'Error'}
TERMINAL={'completed','early_stopped','error'}

class Monitor:
    def __init__(self,key,config):
        self.key=key;self.config=config;self.last_draw=0;self.render_number=0
        self.status=st.empty();self.download=st.empty();self.stats=st.empty();self.progress=st.empty();self.chart=st.empty();self.batch_chart=st.empty()
        if key not in st.session_state:st.session_state[key]={'phase':'queued','config':copy.deepcopy(config),'started':time.monotonic(),'history':[],'batches':[]}
        self.render()

    def event(self,event):
        state=st.session_state[self.key];state.update(event)
        self.render(force=True)

    def batch(self,trace,history,batches,total):
        state=st.session_state[self.key]
        state.update(trace);state['phase']='training';state['history']=history;state['batches']=batches;state['total_batches']=total
        if trace.get('epoch_complete'):
            state['epochs_completed']=len(history)
            state['last_val_loss']=history[-1]['val_loss'];state['last_val_accuracy']=history[-1]['val_accuracy']
            state['best_epoch']=min(history,key=lambda row:row['val_loss'])['epoch']
        self.render(force=bool(trace.get('epoch_complete')))

    def render(self,force=True):
        now=time.monotonic()
        if not force and now-self.last_draw<.25:return
        self.last_draw=now;self.render_number+=1
        state=st.session_state[self.key];phase=state['phase'];elapsed=now-state['started']
        with self.status.container():
            st.subheader(PHASES.get(phase,phase))
            message=state.get('message','La solicitud fue recibida; aún no hay batches.')
            if phase=='error':st.error(message)
            elif phase in TERMINAL:st.success(message)
            else:st.caption(message)
            st.caption(f"Dataset: {self.config['dataset']} · configuración congelada · CPU · {elapsed:.1f}s")
        if phase in {'downloading','verifying','extracting','cache_check','cached','reading','prepared','preparing'}:
            with self.download.container():
                received=state.get('downloaded',0);total=state.get('total')
                if phase=='downloading':
                    if total:st.progress(min(received/total,1),text=f"{received/1024**2:.1f} / {total/1024**2:.1f} MB")
                    else:st.caption(f"{received/1024**2:.1f} MB recibidos · tamaño total desconocido, sin porcentaje inventado")
                    speed=state.get('speed')
                    if speed:
                        eta=f" · ETA orientativa {max(total-received,0)/speed:.0f}s" if total else ""
                        st.caption(f"{speed/1024:.1f} KB/s{eta} · límite: 20s por lectura y ~5 min por archivo")
                else:st.caption('Este paso precede al entrenamiento; el progreso de batches aún no comenzó.')
        else:self.download.empty()
        if 'step' in state:
            with self.stats.container():
                cols=st.columns(3)
                cols[0].metric('Época / máximo',f"{state['epoch']} / {self.config['epochs']}")
                cols[1].metric('Batch de esta época',f"{state['batch']} / {state['total_batches']}")
                cols[2].metric('Batches totales vistos',state['step'])
                cols=st.columns(3)
                cols[0].metric('Loss del batch',f"{state['loss']:.4f}")
                cols[1].metric('Loss train · media actual',f"{state['train_loss_running']:.4f}")
                cols[2].metric('Accuracy train · actual',f"{state['train_accuracy_running']:.1%}")
                val_loss=f"{state['last_val_loss']:.4f}" if "last_val_loss" in state else "todavía no disponible"
                val_accuracy=f"{state['last_val_accuracy']:.1%}" if "last_val_accuracy" in state else "todavía no disponible"
                st.caption(f"Muestras vistas en esta época: {state['samples_seen']} · última loss val: {val_loss} · accuracy val: {val_accuracy}")
                if self.config.get('early_stopping'):st.caption(f"Paciencia: {state.get('patience_count',0)} / {self.config['patience']} · mejor checkpoint: {state.get('best_epoch','pendiente')}")
            fraction=((state['epoch']-1)*state['total_batches']+state['batch'])/(self.config['epochs']*state['total_batches'])
            self.progress.progress(min(fraction,1),text='Batches ejecutados respecto del máximo configurado (la parada temprana puede dejarlo incompleto)')
        history=state.get('history',[])
        if history:
            fig=go.Figure()
            for column,name,color in [('train_loss','Train · media por época','#087e83'),('val_loss','Validación · final de época','#a75d42')]:
                fig.add_trace(go.Scatter(x=[row['epoch'] for row in history],y=[row[column] for row in history],mode='lines+markers',name=name,line=dict(color=color)))
            fig.update_layout(height=290,xaxis_title='Época',yaxis_title='Cross entropy',xaxis=dict(dtick=1),margin=dict(l=30,r=20,t=20,b=40),legend=dict(orientation='h',y=-.25))
            self.chart.plotly_chart(fig,width='stretch',key=f'{self.key}_epoch_{self.render_number}')
        else:self.chart.caption('Loss de validación aparecerá tras la primera validación. No se representa como cero.')
        batches=state.get('batches',[])
        if batches:
            fig=go.Figure(go.Scatter(x=[row['step'] for row in batches[-500:]],y=[row['loss'] for row in batches[-500:]],mode='lines',name='Train · batch'))
            fig.update_layout(height=180,xaxis_title='Batch acumulado (últimos500)',yaxis_title='Loss batch',margin=dict(l=30,r=20,t=10,b=35))
            self.batch_chart.plotly_chart(fig,width='stretch',key=f'{self.key}_batch_{self.render_number}')
