import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

def plot_metrics(history: Dict[str, List[float]], save_path: Path):
    """
    Plot training metrics (Loss and WER/CER) and save to file.
    
    Args:
        history: Dictionary containing lists of metrics. 
                 Expected keys: 'train_loss', 'val_loss', 'val_wer', 'val_cer'.
                 Lists should be of same length (epochs).
        save_path: Path to save the plot image.
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    
    if 'val_wer' in history:
        ax1.plot(epochs, history['val_wer'], label='Val WER', linestyle='--', marker='x')
        
    ax2.set_title('Validation Metrics (WER/CER)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Error Rate')
    
    if 'val_wer' in history:
        ax2.plot(epochs, history['val_wer'], label='Val WER', marker='o')
    if 'val_cer' in history:
        ax2.plot(epochs, history['val_cer'], label='Val CER', marker='o')
        
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
