import matplotlib.pyplot as plt

def plot_signals(time, signals, labels=None, title='Simulated Signals'):
    plt.figure()
    if signals.ndim == 1:
        plt.plot(time, signals)
    else:
        for i in range(signals.shape[1]):
            plt.plot(time, signals[:,i], label=labels[i] if labels else f'Component {i+1}')
        plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()