import math

# --- Parameters ---
# Percentages entered as decimals: 0.10% = 0.001, 1% = 0.01
ALPHA =  0.0       # baseline drift (decimal)
BU    =  1.0       # up-market beta (multiplier)
BD    =  1.0       # down-market beta (multiplier)
S0    =  0.001     # residual std dev vs index (decimal, e.g. 0.001 = 0.10%)
MED   =  0.0       # median daily return (decimal)
SOWN  =  0.002     # stock's own daily std dev (decimal, e.g. 0.002 = 0.20%)

MV = [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]
K  = 50
LN2 = math.log(2)

# --- Core logic ---

def softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0)

def split_market(rm):
    """Split market return into positive / negative components via softplus."""
    p =  (softplus(K * rm)  - LN2) / K
    n = -(softplus(-K * rm) - LN2) / K
    return p, n

def expected_return(rm):
    p, n = split_market(rm)
    return ALPHA + BU * p + BD * n

def sign(x):
    return (1 if x > 0 else -1 if x < 0 else 0)

def calc(rs, rm):
    ex = expected_return(rm)
    ar = rs - ex
    zi = ar / max(S0, 1e-8)
    zo = (rs - MED) / max(SOWN, 1e-8)
    return {
        "rs": rs, "rm": rm, "ex": ex, "ar": ar,
        "zi": zi, "zo": zo,
        "idx_score": sign(zi) * zi * zi,
        "own_score": sign(zo) * zo * zo,
        "combined":  sign(zi) * abs(zi) * abs(zo),
    }

# --- Build and print matrix with rich + colors ---
from rich.console import Console
from rich.table import Table

def fmt_pct(v):
    return f"{v*100:+.0f}%" if v != 0 else "0%"

def color_val(x):
    v = x["combined"]
    if v > 0:
        return f"[green]{v:+.1f}[/green]"
    elif v < 0:
        return f"[red]{v:+.1f}[/red]"
    return f"[white]{v:+.1f}[/white]"

rows = [[calc(rs, rm) for rm in MV] for rs in MV]

col_headers = [fmt_pct(m) for m in MV]
row_headers = [fmt_pct(m) for m in MV]

console = Console()

table = Table(title="Combined Impact Matrix", show_lines=True)
table.add_column("Stock\\Index", justify="right", style="bold cyan")

for h in col_headers:
    table.add_column(h, justify="right", style="bold")

for r_label, row in zip(row_headers, rows):
    vals = [color_val(x) for x in row]
    table.add_row(r_label, *vals)

console.print(table)