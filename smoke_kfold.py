"""Three independent small CPU folds on offline Digits; never evaluate heldout test."""
import numpy as np
from cnn_lab.model import Config
from cnn_lab.cross_validation import cross_validate

if __name__ == '__main__':
    result=cross_validate(Config(samples=600,epochs=2),k=3)
    validation=np.concatenate([fold['val_indices'] for fold in result['folds']])
    assert len(validation)==len(np.unique(validation))
    assert not set(validation)&set(result['heldout_test_indices'])
    assert all(np.isfinite(fold['val_loss']) for fold in result['folds'])
    print({'k':result['k'],'epochs_per_fold':result['epochs'],'development_samples_validated_once':len(validation),'untouched_test_samples':len(result['heldout_test_indices']),'accuracy_by_fold':[round(fold['accuracy'],4) for fold in result['folds']],'f1_macro_by_fold':[round(fold['f1'],4) for fold in result['folds']],'aggregate':result['aggregate'],'std_ddof':result['std_ddof'],'seconds':result['elapsed']})
