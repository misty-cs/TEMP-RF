"""Shared figure style for the paper.

One place to set type, size and colour so every figure in the paper matches.

Type: Times New Roman at 8 pt, matching IEEE conference body text, so figure
labels read at the same size as the surrounding prose rather than shrinking
under \\includegraphics scaling. Figures are therefore drawn at their final
printed size --- 3.5 in for one column, 7.16 in for two --- and included at
\\linewidth with no rescaling.

Colour: the categorical slots are used in fixed order and never cycled; a
method keeps its colour across every figure. Magnitude uses one hue, light to
dark. Text never takes a series colour.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── categorical slots, fixed order ──────────────────────────────────────────
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7')
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]

# one method, one colour, everywhere
METHOD_COLOR = {
    'Softmax':          '#8a8a86',
    'OpenMax':          ORANGE,
    'Static prototype': AQUA,
    'Cosine $k$-NN':    YELLOW,
    'Single-centroid':  MAGENTA,
    'Multi-prototype':  BLUE,
}

# sequential ramp (blue, light -> dark) for magnitude
SEQUENTIAL = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

INK        = '#0b0b0b'    # primary text
INK_2      = '#52514e'    # secondary text
GRID       = '#d8d7d2'
CHANCE     = '#9b9b96'    # reference lines

COL_W  = 3.45   # inches, single IEEE column
COL_2W = 7.16   # inches, full width


def use():
    mpl.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':  'stix',
        'font.size':          8,
        'axes.labelsize':     8,
        'axes.titlesize':     8,
        'xtick.labelsize':    7.5,
        'ytick.labelsize':    7.5,
        'legend.fontsize':    7.5,
        'axes.edgecolor':     GRID,
        'axes.labelcolor':    INK,
        'axes.linewidth':     0.6,
        'axes.grid':          True,
        'axes.axisbelow':     True,
        'grid.color':         GRID,
        'grid.linewidth':     0.5,
        'grid.alpha':         0.9,
        'xtick.color':        INK_2,
        'ytick.color':        INK_2,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.major.size':   2.5,
        'ytick.major.size':   2.5,
        'text.color':         INK,
        'legend.frameon':     False,
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.01,
        'pdf.fonttype':       42,     # embed TrueType, not Type 3
        'ps.fonttype':        42,
    })


def despine(ax, left=True, bottom=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def chance_line(ax, value, label='chance', x=0.99, ha='right', data_x=False):
    """A reference line for the chance rate, labelled once, never in a series colour.

    `x` is in axes fraction unless `data_x`, in which case it is a data
    coordinate --- useful for dropping the label into a gap between bars.
    """
    ax.axhline(value, color=CHANCE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    trans = ax.transData if data_x else ax.get_yaxis_transform()
    ax.text(x, value, f' {label} ', transform=trans, ha=ha, va='bottom',
            fontsize=6.5, color=CHANCE)


def save(fig, path):
    fig.savefig(path)
    fig.savefig(str(path).replace('.pdf', '.png'), dpi=300)
    print(f'  wrote {path}')
