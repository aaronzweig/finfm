import torch

def get_energy_statistics_global(energy_model, dataloader, low=.05, high=.95, noise_sigma=0.0, sigma_scalar=None):
    all_E = []
    for x, y in dataloader:
        if sigma_scalar is None:
            all_E.append(energy_model.get_energy(x, y).detach().cpu())
        else:
            sigma = torch.ones(x.shape[0],1).to(x.device) * sigma_scalar
            all_E.append(energy_model.forward_energy(x, sigma, y).detach().cpu())
    E_cat = torch.cat(all_E)
    return torch.quantile(E_cat, low).item(), torch.quantile(E_cat, high).item()

# def get_energy_weights(energy_model, dataloader, sigma_scalar=None):
#     all_E = []
#     all_y = []
#     for x, y in dataloader:
#         if sigma_scalar is None:
#             all_E.append(energy_model.get_energy(x, y).detach().cpu())
#             all_y.append(y)
#         else:
#             sigma = torch.ones(batch.shape[0],1).to(x.device) * sigma_scalar
#             all_E.append(energy_model.forward_energy(x, sigma, y).detach().cpu())
#             all_y.append(y)
#     E_cat = torch.cat(all_E)
#     y_cat = torch.cat(all_y)
    
#     return E_cat


def get_new_weights(energy_model, dataloader, sigma_scalar=None, low=0.05, high=0.95, weight_beta=1.0):
    all_E = []
    all_y = []
    for x, y in dataloader:
        if sigma_scalar is None:
            all_E.append(energy_model.get_energy(x, y).detach().cpu())
            all_y.append(y)
        else:
            sigma = torch.ones(x.shape[0],1).to(x.device) * sigma_scalar
            all_E.append(energy_model.forward_energy(x, sigma, y).detach().cpu())
            all_y.append(y)
    E_cat = torch.cat(all_E)
    y_cat = torch.cat(all_y)

    n_clusters = torch.max(y_cat) + 1
    low_qs = []
    high_qs = []
    for i in range(n_clusters):
        low_qs.append(torch.quantile(E_cat[y_cat == i], low).item())
        high_qs.append(torch.quantile(E_cat[y_cat == i], high).item())

    weights = E_cat
    for i in range(n_clusters):
        index = y_cat == i
        weights[index] = torch.clamp(weights[index], min=low_qs[i], max=high_qs[i]) - high_qs[i]
        weights[index] = torch.exp(weight_beta * weights[index])
        #Mixture distribution over clusters
        weights[index] /= torch.sum(index) * torch.sum(weights[index])
    
    return weights