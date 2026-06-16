import torch
from ..architectures.genomic_model_zero import GenomicModelDeepStarrTwo


def load_deep_starr_second(checkpoint_path: str, device):

    model = GenomicModelDeepStarrTwo()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model