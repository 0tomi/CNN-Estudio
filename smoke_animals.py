"""Explicit integration smoke: downloads the official archive once and trains on animals only."""
import io
import numpy as np
from PIL import Image
from cnn_lab.model import Config
from cnn_lab.training import train
from cnn_lab.metrics import evaluate
from cnn_lab.data import display_image
from cnn_lab.inference import classify_latest

if __name__ == '__main__':
    run = train(Config(dataset="CIFAR10Animals", samples=600, epochs=3))
    report = evaluate(run['data'].labels[run['data'].test],run['probabilities'])
    raw = display_image(run['data'].images[run['data'].test[0]])
    encoded = io.BytesIO()
    Image.fromarray(np.rint(raw*255).astype('uint8')).save(encoded,format='PNG')
    result = classify_latest([run],encoded.getvalue())
    assert result['probabilities'].shape == (6,)
    assert np.isfinite(result['probabilities']).all()
    assert np.isclose(result['probabilities'].sum(),1)
    assert np.allclose(result['probabilities'],run['probabilities'][0],atol=1e-6)
    print({'dataset':run['config']['dataset'],'train':len(run['data'].train),'validation':len(run['data'].val),'official_test_subset':len(run['data'].test),'names':result['names'],'best_epoch':run['best_epoch'],'test_accuracy':report['accuracy'],'test_f1_macro':report['f1'],'prediction_for_uploaded_heldout':result['names'][result['prediction']],'expected':run['data'].names[int(run['data'].labels[run['data'].test[0]])],'finite':True,'seconds':run['elapsed']})
