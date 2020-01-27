def get_number_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)