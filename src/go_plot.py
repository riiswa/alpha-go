import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

__ticks = ['Filler', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'Filler']


def plot_play_probabilities(board, probabilities: np.ndarray, cmap="viridis"):
    """
    Plot the Go board with play probabilities.
    You need to call ```plot.show``` afterwards because this only plots on the current Axes object.

    Parameters
    -----------
    - **board**: the Goban board or a np.ndarray of shape (9, 9)
    - **probabilities**: a np.ndarray of probabilities of shape (82), probabilities[81] being the pass probability
    - **cmap**: (*optional*) the color map to use
    """
    ax = plt.gca()
    
    if hasattr(board, "_board"):
        def get_token(x, y):
            return board._board[board._BOARDSIZE * y + x]
    else:
        def get_token(x, y):
            return board[y, x]

    pass_probability: float = probabilities[-1]
    probabilities: np.ndarray = probabilities[:-1].reshape((9, 9))

    ax.matshow(probabilities, vmin=0, vmax=1, cmap=cmap)

    white_pts_x = []
    white_pts_y = []
    black_pts_x = []
    black_pts_y = []
    for y in range(9):
        for x in range(9):
            token = get_token(x, y)
            if token == 1:
                black_pts_x.append(x)
                black_pts_y.append(y)
            elif token == 2:
                white_pts_x.append(x)
                white_pts_y.append(y)

    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

    ax.plot(black_pts_x, black_pts_y, 'o', color='black', markersize=15)
    ax.plot(white_pts_x, white_pts_y, 'wo', markersize=15)
    ax.grid(True, which='major', axis='both', color='black')
    ax.set_title(f"Pass probability:{pass_probability}")
    ax.set_xticklabels(__ticks)
    ax.set_yticklabels(list(reversed(range(0, 11))))
