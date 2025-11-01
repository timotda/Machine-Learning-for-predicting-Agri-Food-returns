import matplotlib.pyplot as plt
import pandas as pd

def loss_graph(data, title, file_name):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(data['Epoch'], data['Train_Loss'], label='Training loss', color='red')
    ax.plot(data['Epoch'], data['Validation_Loss'], label='Validation loss', color='blue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{file_name}.pdf', dpi=150)
    plt.show()

# Loss graph for model with all features
loss_AF = pd.read_csv('training_losses_AF.csv')
loss_graph(loss_AF, 'Loss Curves - All Features', 'loss_graph_AF')

# Loss graph for model with only important features
loss_IF = pd.read_csv('training_losses_IF.csv')
loss_graph(loss_IF, 'Loss Curves - Important Features', 'loss_graph_IF')
