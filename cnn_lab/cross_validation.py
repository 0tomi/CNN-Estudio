"""Independent stratified folds over development data; never evaluate held-out test."""
import copy
from dataclasses import replace
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .data import prepare, training_statistics
from .training import train, seed_all
from .model import validate
from .metrics import evaluate


def make_folds(data, k, seed, normalize=True):
    if k not in (2, 3, 5):
        raise ValueError("Elegí 2, 3 o 5 folds.")
    development = np.sort(np.concatenate([data.train, data.val]))
    if len(np.unique(development)) != len(development) or np.intersect1d(development, data.test).size:
        raise ValueError("Las particiones de desarrollo/test deben ser disjuntas.")
    labels = data.labels[development].numpy()
    counts = np.bincount(labels, minlength=len(data.names))
    if counts.min() < k:
        raise ValueError(f"K={k} requiere al menos {k} muestras por clase en desarrollo; mínimo actual {counts.min()}. Aumentá muestras o reducí K.")
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_local, val_local in splitter.split(development, labels):
        training, validation = development[train_local], development[val_local]
        mean, std = training_statistics(data.images, training, normalize)
        yield replace(data, train=training, val=validation, mean=mean, std=std)


def cross_validate(config, k=3, epochs=None, callback=None, *, data=None):
    config = copy.deepcopy(config)
    if epochs is not None:
        config.epochs = epochs
    validate(config)
    seed_all(config.seed)
    data = prepare(config) if data is None else data
    # Validate every partition before the first training and reuse this one loaded dataset.
    folds = list(make_folds(data, k, config.seed, config.normalize))
    results = []
    start = time.monotonic()
    for index, fold_data in enumerate(folds, 1):
        fold_config = copy.deepcopy(config)
        fold_config.seed = (config.seed + index - 1) % 2**32
        def notify(trace, history, batches, total, fold=index):
            if callback:
                callback(fold, trace, history, batches, total)
        # Fresh train() invocation creates a fresh model, optimizer and seed for every fold.
        try:
            run = train(fold_config, notify, data=fold_data, evaluate_test=False)
        except ValueError as error:
            raise ValueError(f"Fold {index}/{k}: {error}") from error
        metrics = evaluate(fold_data.labels[fold_data.val], run["probabilities"])
        best = run["history"][run["best_epoch"] - 1]
        results.append({"fold":index, "seed":fold_config.seed, "train_count":len(fold_data.train), "val_count":len(fold_data.val),
                        "accuracy":metrics["accuracy"], "precision":metrics["precision"], "recall":metrics["recall"], "f1":metrics["f1"],
                        "val_loss":best["val_loss"], "best_epoch":run["best_epoch"], "elapsed":run["elapsed"],
                        "train_indices":fold_data.train.tolist(), "val_indices":fold_data.val.tolist(),
                        "mean":np.asarray(fold_data.mean).reshape(-1).tolist(), "std":np.asarray(fold_data.std).reshape(-1).tolist(),
                        "history":run["history"], "batches":run["batches"]})
        # No fold models/optimizers/probabilities are retained in the session result.
        del run
    columns = ["accuracy", "precision", "recall", "f1", "val_loss"]
    table = pd.DataFrame(results)[columns]
    aggregate = {name:{"mean":float(table[name].mean()), "std":float(table[name].std(ddof=1))} for name in columns}
    return {"config":config.dict(), "k":k, "epochs":config.epochs, "folds":results,
            "aggregate":aggregate, "std_ddof":1, "heldout_test_indices":data.test.tolist(),
            "elapsed":time.monotonic()-start}
