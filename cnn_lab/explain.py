import torch

def scale_map(tensor):
    tensor = tensor.detach().float()
    if not torch.isfinite(tensor).all():
        raise ValueError("Mapa no finito; revisá el modelo.")
    tensor = tensor - tensor.min()
    maximum = tensor.max()
    return (tensor/maximum if maximum > 1e-12 else torch.zeros_like(tensor)).numpy(),bool(maximum <= 1e-12)

def explain(model,x,layer,target=0,channel=0,row=0,col=0,method="gradcam"):
    training = model.training
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
        handle.remove(); model.train(training)

@torch.no_grad()
def feature_maps(model,x,layer):
    captured=[]; training=model.training
    handle=model.features.get_submodule(layer).register_forward_hook(lambda _m,_i,o:captured.append(o.detach()))
    try:
        model.eval(); model(x)
        return captured[0][0].numpy()
    finally:
        handle.remove(); model.train(training)
