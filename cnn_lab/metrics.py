import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_curve,auc

def evaluate(labels,probabilities):
    labels = np.asarray(labels); probabilities=np.asarray(probabilities)
    classes = probabilities.shape[1] if probabilities.ndim == 2 else 0
    if classes < 2 or probabilities.shape != (len(labels),classes) or not len(labels) or np.any(labels < 0) or np.any(labels >= classes) or np.any(probabilities < 0) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1,atol=1e-5):
        raise ValueError("Se requieren probabilidades finitas de todas las clases que sumen 1.")
    predictions = probabilities.argmax(1)
    p,r,f,s = precision_recall_fscore_support(labels,predictions,labels=np.arange(classes),zero_division=0)
    curves={}; aucs=[]
    for i in range(classes):
        binary = labels==i
        if binary.any() and (~binary).any():
            fpr,tpr,_ = roc_curve(binary,probabilities[:,i])
            score = auc(fpr,tpr); aucs.append(score); curves[i]=(fpr,tpr,score)
    macro = precision_recall_fscore_support(labels,predictions,labels=np.arange(classes),average="macro",zero_division=0)
    return {"accuracy":accuracy_score(labels,predictions),"precision":macro[0],"recall":macro[1],"f1":macro[2],"auc":np.mean(aucs) if aucs else None,"per_class":{"precision":p,"recall":r,"F1":f,"soporte":s},"confusion":confusion_matrix(labels,predictions,labels=np.arange(classes)),"curves":curves}
