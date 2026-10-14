import torch

def scale_map(tensor):
    tensor = tensor.detach().float()
    if not torch.isfinite(tensor).all():
        raise ValueError("Mapa no finito; revisá el modelo.")
    tensor = tensor - tensor.min()
    maximum = tensor.max()
    return (tensor/maximum if maximum > 1e-12 else torch.zeros_like(tensor)).numpy(),bool(maximum <= 1e-12)

def explain(model,x,layer,target=0,channel=0,row=0,col=0,method="gradcam"):
    modes = {module: module.training for module in model.modules()}
    captured=[]
    module = model.features.get_submodule(layer)
    handle = module.register_forward_hook(lambda _m,_i,out: captured.append(out))
    try:
        model.eval()
        x = x.detach().clone().requires_grad_(True)
        logits = model(x)
        features = captured[0]
        if method == "gradcam":
            gradient = torch.autograd.grad(logits[0,target],features)[0]
            heat = (gradient.mean((2,3),keepdim=True)*features).sum(1).relu()
            heat = torch.nn.functional.interpolate(heat.unsqueeze(1),x.shape[-2:],mode="bilinear",align_corners=False)[0,0]
        elif method == "neuron":
            gradient = torch.autograd.grad(features[0,channel,row,col],x)[0]
            heat = gradient[0].abs().sum(0)
        else: raise ValueError("Método desconocido.")
        heat,constant = scale_map(heat)
        return {"heat":heat,"constant":constant,"features":features.detach()[0].numpy(),"probabilities":logits.detach().softmax(1)[0].numpy()}
    finally:
        handle.remove()
        for module, training in modes.items(): module.training = training

@torch.no_grad()
def feature_maps(model,x,layer):
    captured=[]; modes={module: module.training for module in model.modules()}
    handle=model.features.get_submodule(layer).register_forward_hook(lambda _m,_i,o:captured.append(o.detach()))
    try:
        model.eval(); model(x)
        return captured[0][0].numpy()
    finally:
        handle.remove()
        for module, training in modes.items(): module.training = training


def gradcam_all(model, x, target):
    """Capture every convolution in one forward and one gradient traversal."""
    modes = {module: module.training for module in model.modules()}
    captured, handles = {}, []
    try:
        model.eval()
        for name in model.conv_names():
            handles.append(model.features.get_submodule(name).register_forward_hook(
                lambda _module, _inputs, output, name=name: captured.__setitem__(name, output)))
        source = x.detach().clone().requires_grad_(True)
        logits = model(source)
        if not 0 <= target < logits.shape[1]:
            raise ValueError("La clase objetivo no pertenece a esta red.")
        gradients = torch.autograd.grad(logits[0, target], tuple(captured.values()), allow_unused=True)
        layers = {}
        for (name, features), gradient in zip(captured.items(), gradients):
            if gradient is None:
                layers[name] = {"error": "Esta capa no participa en el logit elegido."}
                continue
            heat = (gradient.mean((2, 3), keepdim=True) * features).sum(1).relu()
            heat = torch.nn.functional.interpolate(heat.unsqueeze(1), source.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            try:
                normalized, constant = scale_map(heat)
                layers[name] = {"heat": normalized, "constant": constant,
                                "features": features.detach()[0].cpu().numpy()}
            except ValueError as error:
                layers[name] = {"error": str(error)}
        return {"layers": layers, "probabilities": logits.detach().softmax(1)[0].cpu().numpy()}
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training
