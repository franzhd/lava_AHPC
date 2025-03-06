
import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc

def create_raster_plot(data, name=None, figsize=None):
    plt.clf()
    if figsize is None:
        channels, timesteps = data.shape
        x_size = timesteps / 2
        y_size = channels / 10 if channels > 100 else channels 
        figsize = (x_size, y_size)
    else:
        figsize = figsize

    plt.figure(figsize=figsize)
    print(data.shape)
    for channel in range(data.shape[0]):
        spike_times = np.where(data[channel])[0]
        plt.vlines(spike_times, channel + 0.5, channel + 1.5)

    if name:
        plt.title(f"{name} Raster Plot")
    else:
        plt.title('Raster Plot')

    plt.yticks(np.arange(0, data.shape[0] + 1), [f'N {i}' for i in range(0, data.shape[0]+1)])
    plt.xlabel('Time (ms)')
    plt.ylabel('Channels')
    plt.ylim(0, data.shape[0])
    plt.xlim(0, data.shape[1])
    plt.xticks(ticks=np.arange(data.shape[1]), labels=np.arange(data.shape[1]))
    plt.grid(axis='x', color='black', linestyle='--', linewidth=0.2)
    plt.show()


# Generate the raster plot

def plot_signals(matrix, name=None, figsize=None):
    """
    Plots the signal for each channel in the provided matrix.
    
    Parameters:
    matrix (np.ndarray): A 2D array where the first dimension represents channels
                         and the second dimension represents timesteps.
    """
    channels, timesteps = matrix.shape

    if figsize is None:
        x_size = timesteps / 2
        y_size = int(channels / 10) if channels > 100 else channels 
        figsize = (x_size, y_size)
    else:
        figsize = figsize

    plt.figure(figsize=figsize)


    for i in range(channels):
        plt.plot(matrix[i], label=f'Channel {i+1}')

    if name:
        plt.title(f"{name} Signal Plot")
    else:
        plt.title('Signal Plot')

    plt.xlabel('Timestep')
    plt.ylabel('Signal Value')
    plt.xlim(0, matrix.shape[1])
    plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(matrix.shape[1]))
    plt.grid(axis='x', color='black', linestyle='--', linewidth=0.2)
    plt.legend(ncol=12, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white', framealpha=1)
    plt.show()

def plot_signals_subplot(matrix1, matrix2, name1=None, name2=None, figsize=None):
    """
    Plots the signal for each channel in the provided matrices side by side.
    
    Parameters:
    matrix1 (np.ndarray): A 2D array where the first dimension represents channels
                        and the second dimension represents timesteps.
    matrix2 (np.ndarray): A 2D array where the first dimension represents channels
                        and the second dimension represents timesteps.
    """
    channels1, timesteps1 = matrix1.shape
    channels2, timesteps2 = matrix2.shape

    if figsize is None:
        x_size = max(timesteps1, timesteps2) 
        y_size = max(channels1, channels2) / 20 if max(channels1, channels2) > 100 else max(channels1, channels2)
        figsize = (x_size, y_size)
    else:
        figsize = figsize

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    colorblind_colormap = cc.cm['CET_CBL2']
    for i in range(channels1):
        axs[0].plot(matrix1[i], label=f'Channel {i+1}')
    for i in range(channels2):
        axs[1].plot(matrix2[i], label=f'Channel {i+1}')
    
    if name1:
        axs[0].set_title(f"{name1} Signal Plot")
    else:
        axs[0].set_title('Signal Plot 1')
    
    if name2:
        axs[1].set_title(f"{name2} Signal Plot")
    else:
        axs[1].set_title('Signal Plot 2')

    for ax in axs:
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Signal Value')
        ax.set_xlim(0, max(timesteps1, timesteps2))
        ax.set_xticks(ticks=np.arange(max(timesteps1, timesteps2)), labels=np.arange(max(timesteps1, timesteps2)))
        ax.grid(axis='x', color='black', linestyle='--', linewidth=0.2)
        # ax.legend(ncol=12, loc='upper center', bbox_to_anchor=(0.5, -0.05), facecolor='white', framealpha=1)
    
    plt.show()

def create_raster_plot_subplot(data1, data2, name1=None, name2=None, figsize=None):
    """
    Creates raster plots for the provided data side by side.
    
    Parameters:
    data1 (np.ndarray): A 2D array where the first dimension represents channels
                        and the second dimension represents timesteps.
    data2 (np.ndarray): A 2D array where the first dimension represents channels
                        and the second dimension represents timesteps.
    """
    if figsize is None:
        channels1, timesteps1 = data1.shape
        channels2, timesteps2 = data2.shape
        x_size = int(max(timesteps1, timesteps2) / 1.5  )
        y_size = max(channels1, channels2) / 20 if max(channels1, channels2) > 100 else max(channels1, channels2) / 2
        figsize = (x_size, y_size)
    else:
        figsize = figsize

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    for channel in range(data1.shape[0]):
        spike_times = np.where(data1[channel])[0]
        axs[0].vlines(spike_times, channel + 0.5, channel + 1.5)
    
    for channel in range(data2.shape[0]):
        spike_times = np.where(data2[channel])[0]
        axs[1].vlines(spike_times, channel + 0.5, channel + 1.5)

    if name1:
        axs[0].set_title(f"{name1} Raster Plot")
    else:
        axs[0].set_title('Raster Plot 1')
    
    if name2:
        axs[1].set_title(f"{name2} Raster Plot")
    else:
        axs[1].set_title('Raster Plot 2')

    for ax in axs:
        ax.set_yticks(np.arange(0, max(data1.shape[0], data2.shape[0]) + 1), [f'N {i}' for i in range(0, max(data1.shape[0], data2.shape[0]) + 1)])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Channels')
        ax.set_ylim(0, max(data1.shape[0], data2.shape[0]))
        ax.set_xlim(0, max(data1.shape[1], data2.shape[1]))
        ax.set_xticks(ticks=np.arange(max(data1.shape[1], data2.shape[1])), labels=np.arange(max(data1.shape[1], data2.shape[1])))
        ax.grid(axis='x', color='black', linestyle='--', linewidth=0.3)
    
    plt.show()

