import math
from rich.console import Console
from rich.table import Table

# --- Parameters ---
ALPHA =  0.0
BU    =  1.0
BD    =  1.0
S0    =  0.001
MED   =  0.005
SOWN  =  0.020

MV = [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]
K  = 50
LN2 = math.log(2)

def softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0)

def split_market(rm):
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
    return {"zi": zi, "zo": zo}

# --- Scoring options ---

def score_current(zi, zo):
    """Current: sign(zi) * |zi| * |zo|"""
    return sign(zi) * abs(zi) * abs(zo)

def score_a(zi, zo):
    """A) Pure SAR: sign(zi) * zi²"""
    return sign(zi) * zi * zi

def score_d(zi, zo):
    """D) SAR + 1σ amplifier: sign(zi) * zi² * max(1, |zo|)"""
    return sign(zi) * zi * zi * max(1, abs(zo))

def score_e(zi, zo):
    """E) Own-history only: sign(zo) * zo²"""
    return sign(zo) * zo * zo

OPTIONS = [
    ("Current: sign(zi)·|zi|·|zo|", score_current),
    ("A) Pure SAR: sign(zi)·zi²", score_a),
    ("D) SAR + 1sd amplifier: sign(zi)·zi²·max(1,|zo|)", score_d),
    ("E) Own-history only: sign(zo)·zo²", score_e),
]

def fmt_pct(v):
    return f"{v*100:+.0f}%" if v != 0 else "0%"

console = Console()

for title, score_fn in OPTIONS:
    table = Table(title=title, show_lines=True)
    table.add_column("Stock\\Index", justify="right", style="bold cyan")
    for m in MV:
        table.add_column(fmt_pct(m), justify="right", style="bold")

    for rs in MV:
        vals = []
        for rm in MV:
            c = calc(rs, rm)
            v = score_fn(c["zi"], c["zo"])
            if v > 0:
                vals.append(f"[green]{v:+.0f}[/green]")
            elif v < 0:
                vals.append(f"[red]{v:-.0f}[/red]")
            else:
                vals.append(f"{v:.0f}")
        table.add_row(fmt_pct(rs), *vals)

    console.print(table)
    console.print()
