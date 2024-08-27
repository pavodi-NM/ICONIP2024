"This file will define all plots I intend to use"

from typing import Optional, List
import matplotlib.pyplot as plt 
from matplotlib.ticker import ScalarFormatter


def twin_plot(loss, error, optimizer: Optional[str] = None, yscale: Optional[str] = None, format:Optional[bool] = False) -> None:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(loss, 'g-')
    ax2.plot(error, 'b-')

    ax1.set_xlabel('Epochs')

    if yscale is not None:
        ax1.set_ylabel(f'Training loss - {yscale} scale', color='g')
        ax2.set_ylabel(f'Test error - {yscale} scale', color = 'b')
    else:
        ax1.set_ylabel('Training loss', color='g')
        ax2.set_ylabel('Test error', color = 'b')

    ax1.set_ylim([min(loss), max(loss)])
    ax2.set_ylim([min(error), max(error)])

    if yscale is not None:
        ax1.set_yscale(yscale)
        ax2.set_yscale(yscale)

    if format:
        formatter = ScalarFormatter(useMathText=format)
        formatter.set_scientific(format)

    plt.title(f"Training loss and Test error : {optimizer}")
    #plt.legend()
    plt.show()
    

def single_plot(loss, error, optimizer: Optional[str] = None, yscale: Optional[str] = None, format: Optional[bool] = False, use_ylim: Optional[bool] = False) -> None:
    fig, ax = plt.subplots()
    
    ax.plot(loss, 'g-', label='Training Loss')
    ax.plot(error, 'b-', label='Test Error')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    
    # Set the limits for y-axis based on the minimum and maximum of both loss and error
    if use_ylim:
        all_values = loss + error
        ax.set_ylim([min(all_values), max(all_values)])
    
    if yscale is not None:
        ax.set_yscale(yscale)
        ax.set_ylabel(f'Error ({yscale} scale)')

    if format:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        ax.yaxis.set_major_formatter(formatter)

    plt.title(f"Training Loss and Test Error: {optimizer}")
    plt.legend()
    plt.show()