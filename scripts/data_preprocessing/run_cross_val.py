# scripts/run_cross_val.py
import os
import sys
import torch
import logging

# Ustawienie logowania na konsolę
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Imporujemy zaimplementowane wcześniej moduły
## Zakładamy, że kod z poprzednich kroków umieściłeś w pliku cross_validation_core.py
from PromotorOptimizer.loaders.add_metrics import run_cross_validation_pipeline

# ----------------------------------------------------------------------
# MOCK ARCHITECTURES - Podmień te definicje na swoje rzeczywiste klasy
# ----------------------------------------------------------------------
## Modele muszą zwracać tuple: (hidden_space, ratio) zgodnie z konstrukcją modeli
class MockDeepstar(torch.nn.Module):
    def forward(self, x): return None, torch.randn(x.shape[0], 1)

class MockSecondDeepstarr(torch.nn.Module):
    def forward(self, x): return None, torch.randn(x.shape[0], 1)

class MockModelOriginal(torch.nn.Module):
    def forward(self, x): return None, torch.randn(x.shape[0], 1)


if __name__ == "__main__":
    # Definicja relatywnych ścieżek z poziomu CWD: ~/repositories/final
    INPUT_DIR = "data/results_final"
    OUTPUT_DIR = "data/results_final_cross_validated"
    
    # Krok 1: Instancjonowanie architektur i ładowanie wag
    ## Wstaw tutaj rzeczywiste ścieżki do plików .pt/.pth swoich modeli
    logger.info("Initializing neural network architectures...")
    
    deepstar = MockDeepstar()
    # deepstar.load_state_dict(torch.load("weights/deepstar.pt"))
    
    second_deepstarr = MockSecondDeepstarr()
    # second_deepstarr.load_state_dict(torch.load("weights/second_deepstarr.pt"))
    
    model_original = MockModelOriginal()
    # model_original.load_state_dict(torch.load("weights/model_original.pt"))
    
    # Słownik mapujący nazwy na obiekty modeli
    models_dict = {
        "Deepstar": deepstar,
        "Second Deepstarr": second_deepstarr,
        "Model Original": model_original
    }
    
    # Krok 2: Wyznaczenie dostępności karty graficznej
    device_target = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target compute device isolated: {device_target}")
    
    # Krok 3: Uruchomienie zunifikowanego potoku orchestracji
    run_cross_validation_pipeline(
        json_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        models=models_dict,
        device=device_target,
        batch_size=64
    )
    
    logger.info("Cross-validation process completed successfully.")