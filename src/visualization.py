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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        x = range(1, len(history['train_loss']) + 1)
        ax1.plot(x, history['train_loss'], label='Train Loss', marker='o')
        
    if 'val_loss' in history and len(history['val_loss']) > 0:
        x = range(1, len(history['val_loss']) + 1)
        ax1.plot(x, history['val_loss'], label='Val Loss', marker='o')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Steps/Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Metrics (WER/CER)
    ax2.set_title('Validation Metrics (WER/CER)')
    ax2.set_xlabel('Steps/Epochs')
    ax2.set_ylabel('Error Rate')
    
    if 'val_wer' in history and len(history['val_wer']) > 0:
        x = range(1, len(history['val_wer']) + 1)
        ax2.plot(x, history['val_wer'], label='Val WER', marker='o')
        
    if 'val_cer' in history and len(history['val_cer']) > 0:
        x = range(1, len(history['val_cer']) + 1)
        ax2.plot(x, history['val_cer'], label='Val CER', marker='o')
        
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
