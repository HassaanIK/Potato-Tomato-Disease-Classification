def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# num_params = count_parameters(model)
# print(f"Number of parameters in the model: {num_params}")