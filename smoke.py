"""Real CPU training smoke, separate from the quick unit suite."""
import numpy as np
from cnn_lab.model import Config
from cnn_lab.training import train
from cnn_lab.metrics import evaluate

if __name__=='__main__':
    run=train(Config(epochs=5))
    history=run['history']
    assert np.isfinite([r['train_loss'] for r in history]).all()
    assert history[-1]['train_loss']<history[0]['train_loss']
    report=evaluate(run['data'].labels[run['data'].test],run['probabilities'])
    print({'first_train_loss':history[0]['train_loss'],'last_train_loss':history[-1]['train_loss'],'best_epoch':run['best_epoch'],'test_accuracy':report['accuracy'],'test_f1':report['f1'],'seconds':run['elapsed'],'batches':len(run['batches'])})
