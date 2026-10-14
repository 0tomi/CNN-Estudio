"""Configurable CNN with explicit shape and receptive-field bookkeeping."""
from dataclasses import dataclass, field, asdict
import math
import torch
from torch import nn

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
class Config:
    blocks: list[Block] = field(default_factory=lambda: [Block(), Block([Conv(32)])])
    head: str = "flatten"
    hidden: list[int] = field(default_factory=lambda: [64])
    activation: str = "relu"
    dropout: float = 0.1
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

    def dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["blocks"] = [Block(**{**b, "layers": [Conv(**c) for c in b["layers"]]}) for b in data["blocks"]]
        return cls(**data)

DATASET_SPECS = {"Digits": (8, 1, 10), "MNIST": (28, 1, 10), "FashionMNIST": (28, 1, 10), "CIFAR10Animals": (32, 3, 6)}

def dataset_spec(name):
    return DATASET_SPECS[name]

ACTIVATIONS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

def validate(c):
    if not 1 <= len(c.blocks) <= 3 or not 0 <= len(c.hidden) <= 2:
        raise ValueError("Usá 1–3 bloques y 0–2 capas ocultas.")
    if c.dataset not in DATASET_SPECS or c.head not in ("flatten", "gap"):
        raise ValueError("Dataset o salida no reconocido.")
    if c.activation not in ACTIVATIONS or c.optimizer not in ("Adam", "AdamW", "SGD"):
        raise ValueError("Activación u optimizador no reconocido.")
    if not (1 <= c.epochs <= 30 and c.batch_size in (16,32,64,128,256) and 200 <= c.samples <= 10000):
        raise ValueError("Límites: 1–30 épocas, batch 16/32/64/128/256 y 200–10000 muestras.")
    if not (0 < c.lr <= .1 and 0 <= c.dropout <= .7 and 0 <= c.weight_decay <= .1 and 0 <= c.momentum < 1):
        raise ValueError("Hiperparámetros fuera de rango seguro.")
    if not isinstance(c.seed, int) or not 0 <= c.seed < 2**32 or any(not isinstance(n,int) or not 4 <= n <= 256 for n in c.hidden):
        raise ValueError("Semilla o neuronas inválidas.")
    for b in c.blocks:
        if not 1 <= len(b.layers) <= 3 or b.activation not in ACTIVATIONS or b.pooling not in ("none","max","avg") or b.pool_size not in (2,3):
            raise ValueError("Configuración de bloque inválida.")
        for l in b.layers:
            if not (1 <= l.channels <= 64 and l.kernel in (1,3,5) and l.stride in (1,2) and 0 <= l.padding <= 2 and l.dilation in (1,2)):
                raise ValueError("Convolución inválida: canales 1–64; kernel 1/3/5; stride/dilatación 1/2; padding 0–2.")

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
        for index,width in enumerate([*config.hidden,num_classes]):
            linear = nn.Linear(n,width)
            head.append(linear)
            self.rows.append({"capa":"logits" if index==len(config.hidden) else "lineal","salida":str(width),"parámetros":sum(p.numel() for p in linear.parameters()),"RF teórico":"global","salto":"—"})
            n = width
            if index < len(config.hidden):
                head += [ACTIVATIONS[config.activation](),nn.Dropout(config.dropout)]
                self.rows.extend([{ "capa":config.activation+" MLP", "salida":str(width),"parámetros":0,"RF teórico":"global","salto":"—"},{"capa":f"dropout p={config.dropout}","salida":str(width),"parámetros":0,"RF teórico":"global","salto":"—"}])
        self.classifier = nn.Sequential(*head)
        if sum(p.numel() for p in self.parameters()) > 2_000_000:
            raise ValueError("La red supera 2 millones de parámetros; reducí canales/MLP o usá GAP.")

    def forward(self,x):
        return self.classifier(self.features(x))

    def conv_names(self):
        return [name for name,module in self.features.named_children() if isinstance(module,nn.Conv2d)]
