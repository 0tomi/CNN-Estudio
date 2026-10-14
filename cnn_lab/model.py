"""Configurable CNN with explicit shape and receptive-field bookkeeping."""
from dataclasses import dataclass, field, asdict, InitVar
import math
import torch
from torch import nn

MAX_CONV_BLOCKS = 10

@dataclass
class Conv:
    channels: int = 16
    kernel: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1

@dataclass
class Block:
    layers: list[Conv] = field(default_factory=lambda: [Conv()])
    batchnorm: bool = False
    activation: str = "relu"
    pooling: str = "max"
    pool_size: int = 2

@dataclass
class DenseLayer:
    units: int = 64
    activation: str = "relu"
    dropout: float = 0.1
    bias: bool = True

@dataclass
class Config:
    blocks: list[Block] = field(default_factory=lambda: [Block(), Block([Conv(32)])])
    head: str = "flatten"
    dense: list[DenseLayer] | None = None
    dataset: str = "Digits"
    samples: int = 1797
    seed: int = 42
    epochs: int = 8
    batch_size: int = 64
    optimizer: str = "Adam"
    lr: float = 0.003
    weight_decay: float = 0.0
    momentum: float = 0.9
    augment: bool = False
    normalize: bool = True
    early_stopping: bool = False
    patience: int = 3
    min_delta: float = 0.0001

    # Input-only compatibility: canonical exports contain only per-layer dense settings.
    hidden: InitVar[list[int] | None] = None
    activation: InitVar[str | None] = None
    dropout: InitVar[float | None] = None

    def __post_init__(self, hidden, activation, dropout):
        if self.dense is None:
            widths = [64] if hidden is None else hidden
            self.dense = [DenseLayer(width, "relu" if activation is None else activation, .1 if dropout is None else dropout) for width in widths]
        elif any(value is not None for value in (hidden, activation, dropout)):
            raise ValueError("Usá dense por capa o configuración MLP antigua, no ambas a la vez.")

    def dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["blocks"] = [Block(**{**b, "layers": [Conv(**c) for c in b["layers"]]}) for b in data["blocks"]]
        if "dense" in data and data["dense"] is not None:
            data["dense"] = [DenseLayer(**layer) for layer in data["dense"]]
            for key in ("hidden", "activation", "dropout"):
                data.pop(key, None)
        return cls(**data)

DATASET_SPECS = {"Digits": (8, 1, 10), "MNIST": (28, 1, 10), "FashionMNIST": (28, 1, 10), "CIFAR10Animals": (32, 3, 6)}

def dataset_spec(name):
    return DATASET_SPECS[name]

ACTIVATIONS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

def validate(c):
    if not 1 <= len(c.blocks) <= MAX_CONV_BLOCKS or not isinstance(c.dense, list) or not 0 <= len(c.dense) <= 5:
        raise ValueError(f"Usá 1–{MAX_CONV_BLOCKS} bloques y 0–5 capas densas ocultas.")
    if c.dataset not in DATASET_SPECS or c.head not in ("flatten", "gap"):
        raise ValueError("Dataset o salida no reconocido.")
    if c.optimizer not in ("Adam", "AdamW", "SGD"):
        raise ValueError("Activación u optimizador no reconocido.")
    if not (1 <= c.epochs <= 30 and c.batch_size in (16,32,64,128,256) and 200 <= c.samples <= 10000):
        raise ValueError("Límites: 1–30 épocas, batch 16/32/64/128/256 y 200–10000 muestras.")
    if not (0 < c.lr <= .1 and 0 <= c.weight_decay <= .1 and 0 <= c.momentum < 1):
        raise ValueError("Hiperparámetros fuera de rango seguro.")
    if not isinstance(c.seed, int) or not 0 <= c.seed < 2**32:
        raise ValueError("Semilla o neuronas inválidas.")
    if not isinstance(c.early_stopping, bool) or not isinstance(c.patience, int) or isinstance(c.patience, bool) or not 1 <= c.patience <= 30 or not isinstance(c.min_delta, (int, float)) or isinstance(c.min_delta, bool) or not math.isfinite(c.min_delta) or not 0 <= c.min_delta <= 10:
        raise ValueError("Parada temprana inválida: activar booleano, paciencia entera 1–30 y min_delta finito 0–10.")
    for layer in c.dense:
        if not isinstance(layer, DenseLayer) or not isinstance(layer.units, int) or isinstance(layer.units, bool) or not 4 <= layer.units <= 256 or not isinstance(layer.activation, str) or layer.activation not in ACTIVATIONS or not isinstance(layer.bias, bool) or not isinstance(layer.dropout, (int, float)) or isinstance(layer.dropout, bool) or not 0 <= layer.dropout <= .7:
            raise ValueError("Capa densa inválida: 4–256 neuronas, activación válida, dropout 0–0.7 y bias booleano.")
    for b in c.blocks:
        if not 1 <= len(b.layers) <= 3 or b.activation not in ACTIVATIONS or b.pooling not in ("none","max","avg") or not isinstance(b.pool_size,int) or isinstance(b.pool_size,bool) or not 2 <= b.pool_size <= 4:
            raise ValueError("Configuración de bloque inválida.")
        for l in b.layers:
            if not all(isinstance(value,int) and not isinstance(value,bool) for value in (l.channels,l.kernel,l.stride,l.padding,l.dilation)) or not (1 <= l.channels <= 64 and 1 <= l.kernel <= 7 and l.stride in (1,2) and 0 <= l.padding <= 3 and l.dilation in (1,2)):
                raise ValueError("Convolución inválida: filtros 1–64; lado kernel entero 1–7; stride/dilatación 1/2; padding 0–3.")

class CNN(nn.Module):
    def __init__(self, config: Config, size=None, in_channels=None, num_classes=None):
        super().__init__()
        validate(config)
        default_size, default_channels, default_classes = dataset_spec(config.dataset)
        size = default_size if size is None else size
        in_channels = default_channels if in_channels is None else in_channels
        num_classes = default_classes if num_classes is None else num_classes
        self.input_channels, self.num_classes, self.input_size = in_channels, num_classes, size
        self.config = config
        self.rows = []
        modules = []
        ch, spatial, rf, jump = in_channels, size, 1, 1
        def record(name, module, outch, outsize, receptive, step):
            modules.append((name,module))
            self.rows.append({"capa":name,"salida":f"{outch} × {outsize} × {outsize}","parámetros":sum(p.numel() for p in module.parameters()),"RF teórico":receptive,"salto":step})
        for i,b in enumerate(config.blocks):
            for j,l in enumerate(b.layers):
                out = math.floor((spatial+2*l.padding-l.dilation*(l.kernel-1)-1)/l.stride+1)
                if out < 1:
                    raise ValueError(f"Bloque {i+1}, conv {j+1}: tamaño espacial {out}. Reducí pooling/stride/kernel o aumentá padding.")
                rf += (l.kernel-1)*l.dilation*jump
                jump *= l.stride
                record(f"b{i+1}_conv{j+1}",nn.Conv2d(ch,l.channels,l.kernel,l.stride,l.padding,dilation=l.dilation),l.channels,out,rf,jump)
                ch,spatial = l.channels,out
                if b.batchnorm:
                    record(f"b{i+1}_bn{j+1}",nn.BatchNorm2d(ch),ch,spatial,rf,jump)
                record(f"b{i+1}_act{j+1}",ACTIVATIONS[b.activation](),ch,spatial,rf,jump)
            if b.pooling != "none":
                out = spatial//b.pool_size
                if out < 1:
                    raise ValueError(f"Pooling del bloque {i+1} colapsa la imagen. Quitá pooling o reducí bloques.")
                rf += (b.pool_size-1)*jump
                jump *= b.pool_size
                pool = nn.MaxPool2d if b.pooling == "max" else nn.AvgPool2d
                record(f"b{i+1}_pool",pool(b.pool_size),ch,out,rf,jump)
                spatial = out
        from collections import OrderedDict
        self.features = nn.Sequential(OrderedDict(modules))
        head = []
        if config.head == "gap":
            head += [nn.AdaptiveAvgPool2d(1)]
            self.rows.append({"capa":"promedio global","salida":f"{ch} × 1 × 1","parámetros":0,"RF teórico":rf+(spatial-1)*jump,"salto":jump})
        head += [nn.Flatten()]
        self.rows.append({"capa":"flatten","salida":str(ch if config.head == "gap" else ch*spatial*spatial),"parámetros":0,"RF teórico":"—","salto":"—"})
        n = ch if config.head == "gap" else ch*spatial*spatial
        for index, layer in enumerate(config.dense, 1):
            linear = nn.Linear(n, layer.units, bias=layer.bias)
            head += [linear, ACTIVATIONS[layer.activation](), nn.Dropout(layer.dropout)]
            self.rows.extend([
                {"capa":f"densa oculta {index}","entrada":str(n),"salida":str(layer.units),"parámetros":sum(p.numel() for p in linear.parameters()),"RF teórico":"global","salto":"—"},
                {"capa":f"{layer.activation} MLP {index}","salida":str(layer.units),"parámetros":0,"RF teórico":"global","salto":"—"},
                {"capa":f"dropout {index} p={layer.dropout}","salida":str(layer.units),"parámetros":0,"RF teórico":"global","salto":"—"},
            ])
            n = layer.units
        output = nn.Linear(n, num_classes)
        head.append(output)
        self.rows.append({"capa":"salida · logits","entrada":str(n),"salida":str(num_classes),"parámetros":sum(p.numel() for p in output.parameters()),"RF teórico":"global","salto":"—"})
        self.classifier = nn.Sequential(*head)
        if sum(p.numel() for p in self.parameters()) > 2_000_000:
            raise ValueError("La red supera 2 millones de parámetros; reducí canales/MLP o usá GAP.")

    def forward(self,x):
        return self.classifier(self.features(x))

    def conv_names(self):
        return [name for name,module in self.features.named_children() if isinstance(module,nn.Conv2d)]
