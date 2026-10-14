"""Deliberately high min_delta demonstrates stopping mechanics, not discovered overfitting."""
from cnn_lab.model import Config
from cnn_lab.training import train
if __name__=='__main__':
    run=train(Config(samples=600,epochs=8,early_stopping=True,patience=2,min_delta=10.))
    assert run['epochs_completed']==3 and run['stopped_early']
    assert run['best_epoch']==min(run['history'],key=lambda row:row['val_loss'])['epoch']
    print({'epochs_completed':run['epochs_completed'],'max_epochs':run['max_epochs'],'best_epoch':run['best_epoch'],'stop_reason':run['stop_reason'],'stopped_early':run['stopped_early'],'validation_losses':[round(r['val_loss'],5) for r in run['history']],'seconds':run['elapsed'],'note':'min_delta alto forzado para comprobar mecánica, no evidencia de sobreajuste'})
