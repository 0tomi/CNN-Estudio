import copy
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import CNN
from .data import prepare, augment

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

@torch.no_grad()
def predict(model,data,indices):
    model.eval()
    logits = torch.cat([model(batch) for batch in data.normalized(indices).split(256)])
    return logits.softmax(1).numpy(),logits

def train(config, callback=None, *, data=None, evaluate_test=True):
    seed_all(config.seed)
    data = prepare(config) if data is None else data
    model = CNN(config,data.images.shape[-1],data.images.shape[1],len(data.names))
    if len(data.train) % config.batch_size == 1 and any("_bn" in row["capa"] and row["salida"].endswith("× 1 × 1") for row in model.rows):
        raise ValueError("BatchNorm en mapas 1×1 no puede entrenar con el último batch de una muestra. Cambiá batch size/cantidad de muestras, quitá BatchNorm de ese bloque o reducí pooling.")
    kwargs = dict(lr=config.lr,weight_decay=config.weight_decay)
    if config.optimizer == "SGD": kwargs["momentum"] = config.momentum
    optimizer = getattr(torch.optim,config.optimizer)(model.parameters(),**kwargs)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(TensorDataset(data.images[data.train],data.labels[data.train]),batch_size=config.batch_size,shuffle=True,generator=generator)
    history,batches = [],[]
    best_loss,best_epoch,best = float("inf"),0,None
    start = time.monotonic()
    for epoch in range(1,config.epochs+1):
        model.train(); total=correct=seen=0
        for batch,(x,y) in enumerate(loader,1):
            if config.augment: x = augment(x)
            x = (x-data.mean)/data.std
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = loss_fn(logits,y)
            if not torch.isfinite(loss): raise ValueError("Loss no finita. Reducí learning rate.")
            loss.backward(); optimizer.step()
            total += loss.item()*len(y); correct += int((logits.argmax(1)==y).sum()); seen += len(y)
            trace = {"epoch":epoch,"batch":batch,"step":len(batches)+1,"loss":loss.item(),"accuracy":float((logits.argmax(1)==y).float().mean()),"elapsed":time.monotonic()-start}
            batches.append(trace)
            if callback: callback(trace,history,batches,len(loader))
        _,logits = predict(model,data,data.val)
        val_loss = float(loss_fn(logits,data.labels[data.val]))
        row = {"epoch":epoch,"train_loss":total/seen,"val_loss":val_loss,"train_accuracy":correct/seen,"val_accuracy":float((logits.argmax(1)==data.labels[data.val]).float().mean())}
        history.append(row)
        if val_loss < best_loss:
            best_loss,best_epoch,best = val_loss,epoch,copy.deepcopy(model.state_dict())
        if callback: callback({**trace,"epoch_complete":True},history,batches,len(loader))
    model.load_state_dict(best); model.eval()
    probabilities,_ = predict(model,data,data.test if evaluate_test else data.val)
    return {"config":copy.deepcopy(config.dict()),"model":model,"data":data,"history":history,"batches":batches,"probabilities":probabilities,"best_epoch":best_epoch,"elapsed":time.monotonic()-start,"evaluation_split":"test" if evaluate_test else "validation"}
