import math
from rich.console import Console
from rich.table import Table

# --- Parameters ---
ALPHA =  0.0
BU    =  1.0
BD    =  1.0
S0    =  0.001
MED   =  0.0003
SOWN  =  0.020

# Volume parameters
VRATIO    = 2.0     # today's volume / average volume (1.0 = normal day)
VOL_STD   = 0.8     # std dev of ln(daily volume), estimated from stock's history

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
    zv = math.log(VRATIO) / max(VOL_STD, 1e-8)  # volume z-score in log space
    return {"zi": zi, "zo": zo, "zv": zv}

# --- Scoring options ---

def score_a(zi, zo, zv):
    """A) Pure SAR: sign(zi) * zi^2"""
    return sign(zi) * zi * zi

def score_d(zi, zo, zv):
    """D) SAR + own-history amplifier: sign(zi) * zi^2 * max(1, |zo|)"""
    return sign(zi) * zi * zi * max(1, abs(zo))

def score_e(zi, zo, zv):
    """E) Own-history only: sign(zo) * zo^2"""
    return sign(zo) * zo * zo

def score_ev(zi, zo, zv):
    """Ev) Mahalanobis: sign(zo) * sqrt(zo^2 + zv^2)"""
    return sign(zo) * math.sqrt(zo * zo + zv * zv)

def score_dv(zi, zo, zv):
    """Dv) Mahalanobis: sign(zi) * sqrt(zi^2*max(1,|zo|)^2 + zv^2)"""
    d_price = zi * zi * max(1, abs(zo))  # D formula as price component
    return sign(zi) * math.sqrt(d_price * d_price + zv * zv)

ZV_DISPLAY = math.log(VRATIO) / max(VOL_STD, 1e-8)

OPTIONS = [
    ("A) Pure SAR: sign(zi)*zi^2", score_a, ".0f"),
    ("D) SAR + own-history: sign(zi)*zi^2*max(1,|zo|)", score_d, ".0f"),
    ("E) Own-history only: sign(zo)*zo^2", score_e, ".1f"),
    (f"Ev) Mahalanobis: sign(zo)*sqrt(zo^2+zv^2)  [zv={ZV_DISPLAY:.2f}]", score_ev, ".1f"),
    (f"Dv) Mahalanobis: sign(zi)*sqrt(D^2+zv^2)  [zv={ZV_DISPLAY:.2f}]", score_dv, ".0f"),
]

def fmt_pct(v):
    return f"{v*100:+.0f}%" if v != 0 else "0%"

console = Console()

for title, score_fn, fmt in OPTIONS:
    table = Table(title=title, show_lines=True)
    table.add_column("Stock\\Index", justify="right", style="bold cyan")
    for m in MV:
        table.add_column(fmt_pct(m), justify="right", style="bold")

    for rs in MV:
        vals = []
        for rm in MV:
            c = calc(rs, rm)
            v = score_fn(c["zi"], c["zo"], c["zv"])
            if v > 0:
                vals.append(f"[green]{v:+{fmt}}[/green]")
            elif v < 0:
                vals.append(f"[red]{v:-{fmt}}[/red]")
            else:
                vals.append(f"{v:{fmt}}")
        table.add_row(fmt_pct(rs), *vals)

    console.print(table)
    console.print()

# Volume as separate metric
console.print(f"[bold]Volume (separate signal)[/bold]: vratio={VRATIO}, "
              f"ln(vratio)={math.log(VRATIO):+.2f}, "
              f"zv={ZV_DISPLAY:+.2f} (zv = ln(vratio) / vol_std)")
console.print(f"  Interpretation: zv>0 above-avg volume, zv<0 below-avg, "
              f"|zv|>2 unusual volume")
