"""Draw the implementation-aligned FedPASA framework (PNG only).

Run: python make_framework_figure.py [--output PATH] [--dpi 600]
No model training or experimental-result generation is performed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE = "#263449", "#677589", "#8996A6"
BLUE, TEAL, AMBER = "#3A6B9E", "#287D79", "#A7752D"

def render(output: Path, dpi: int = 600) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "mathtext.fontset": "dejavusans", "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.04, 4.10), dpi=dpi)
    ax = fig.add_axes((.006, .008, .988, .984))
    ax.set(xlim=(-1, 181), ylim=(-12, 94))
    ax.axis("off")
    texts = []

    def text(x, y, s, size=8, color=INK, weight="normal", ha="center"):
        t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va="center", zorder=6)
        texts.append(t)
        return t

    def box(x, y, w, h, fill="white", edge=LINE, lw=.7, radius=1.1):
        ax.add_patch(FancyBboxPatch((x,y), w,h,
            boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
            edgecolor=edge, linewidth=lw, zorder=1))

    def line(points, color=INK, lw=.85, dashed=False):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=lw,
                linestyle=(0,(3,2.4)) if dashed else "-",
                solid_capstyle="round", zorder=3)

    def arrow(points, color=INK, lw=.85, dashed=False):
        if len(points)>2:
            line(points[:-1], color, lw, dashed)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1],
            arrowstyle="-|>", mutation_scale=7.3, shrinkA=0, shrinkB=0,
            linewidth=lw, color=color,
            linestyle=(0,(3,2.4)) if dashed else "-", zorder=4))

    def node(x,y,s,color=INK):
        ax.add_patch(Circle((x,y),2.55, facecolor="white",
                            edgecolor=color, linewidth=.8, zorder=4))
        text(x,y,s,9,color)

    def panel(x,w,title,subtitle,fill,accent):
        box(x,25,w,38.5,fill,accent,.65,1.7)
        text(x+2.3,60.8,title,8.3,accent,"bold","left")
        text(x+2.3,56.6,subtitle,7,MUTED,ha="left")

    # The source initialization and parameter exchange are outside feature flow.
    text(1,91.1,"(a) Source initialization",8.7,weight="bold",ha="left")
    box(1,75.5,58,11.5,"#F4F6F9","#C8D0DA")
    text(30,83.2,"Source interactions + frozen features",7.3)
    text(30,78.7,r"BPR + FedAvg  $\rightarrow\ \theta^{\mathrm{src}}\ \rightarrow\ A_h$",8)
    text(68,91.1,"(b) Federated optimization",8.7,weight="bold",ha="left")
    box(68,75.5,47,11.5,"#F4F6F9","#C8D0DA")
    text(91.5,83.3,"Server / FedAvg",8.1,weight="bold")
    text(91.5,78.7,r"$\Theta=\{\theta_h,\theta_l,\phi\}$",8.3)
    box(148,75.5,31,11.5,"#F4F6F9","#C8D0DA")
    text(163.5,83.3,"Target clients",8.1,weight="bold")
    text(163.5,78.7,"Local BPR updates",7.2)
    arrow([(116,83.4),(146.8,83.4)],BLUE,dashed=True)
    text(131.4,87,r"broadcast $\Theta$",7.1,BLUE)
    arrow([(146.8,78.8),(116,78.8)],BLUE,dashed=True)
    text(131.4,75.1,r"upload $\Theta_c$",7.1,BLUE)

    # Target-client boundary and three reliability controls.
    box(0,6.5,180,64.5,"#FCFDFE","#B9C3CF",radius=2)
    text(3,67.9,"(c) Target client c",8.8,weight="bold",ha="left")
    text(178,67.9,"Shared trainable modules; private user state",7,MUTED,ha="right")
    panel(29,44,"01  Item / PGDR","Target training frequency","#EDF3FA",BLUE)
    panel(78,46,"02  Client / FiLM","Local semantic profile","#ECF5F2",TEAL)
    panel(129,48,"03  Domain / AutoAnchor","Density-dependent prior weight","#FAF3E8",AMBER)
    text(13,58.1,"Item text",8,weight="bold")
    box(2,41.5,22,11,"#F0F2F5","#A5AFBB")
    text(13,49,"Frozen",7.8)
    text(13,44.9,"encoder",7.8)
    arrow([(13,55.2),(13,53)],MUTED)
    arrow([(13,41.5),(13,35.7)],MUTED)
    text(13,32.8,r"Prior $x_i$",8.4)

    # Frequency-weighted head/tail mixture.
    text(51,51.9,r"$\mathrm{freq}(i)\ \rightarrow\ q'_i$",8.2,BLUE)
    box(32,39,17,9.6,edge=BLUE)
    text(40.5,45.4,r"$A_h(x_i)$",8.5)
    text(40.5,41.2,"source init.",6.9,MUTED)
    box(53,39,17,9.6,edge=BLUE)
    text(61.5,45.4,r"$A_l(x_i)$",8.5)
    text(61.5,41.2,"random init.",6.9,MUTED)
    arrow([(23,33),(26,33),(26,44),(31,44)],BLUE)
    arrow([(26,44),(26,49.9),(61.5,49.9),(61.5,48.7)],BLUE)
    node(61.5,29.7,r"$\Sigma$",BLUE)
    arrow([(40.5,39),(40.5,33.8),(57.8,33.8),(59.6,31.6)],BLUE)
    text(49.9,36,r"$1-q'_i$",7.3,BLUE)
    arrow([(61.5,39),(61.5,32.3)],BLUE)
    text(67.2,35.4,r"$q'_i$",7.5,BLUE)
    arrow([(64.1,29.7),(75.7,29.7),(75.7,39.1),(81.5,39.1)])
    text(70,27.0,r"$h_i$",8.2)

    # FiLM precedes the alpha residual mixture in the experiment implementation.
    box(81,47.8,40,5.8,edge="#91BCB8",radius=.8)
    text(101,50.7,r"History $\rightarrow p_c \rightarrow g_\phi$",7.7,TEAL)
    arrow([(100,47.7),(100,43.7)],TEAL)
    text(112,45.5,r"$\gamma_c,\beta_c$",7.8,TEAL)
    box(82,34.8,36,8.6,edge=TEAL)
    text(100,39.1,r"$\mathrm{FiLM}(h_i;\gamma_c,\beta_c)$",8,TEAL)
    node(100,28.2,r"$\Sigma$",TEAL)
    arrow([(100,34.7),(100,30.8)],TEAL)
    text(107.1,32.6,r"$\alpha$",7.7,TEAL)
    arrow([(40.5,33.8),(40.5,21.8),(100,21.8),(100,25.6)],BLUE)
    text(69,23.4,r"Head residual: $(1-\alpha)A_h(x_i)$",7.2,BLUE)
    box(109,25.7,12.3,5.1,edge=TEAL,radius=.8)
    text(115.15,28.25,"norm",7.1,TEAL)
    arrow([(102.7,28.2),(108.4,28.2)],TEAL)
    arrow([(121.4,28.2),(126.8,28.2),(126.8,41.2),(132.4,41.2)])
    text(121.7,23.0,r"$z_{c,i}$",8)

    for x,y in [(26,44),(40.5,33.8)]:
        ax.add_patch(Circle((x,y),.45,facecolor=BLUE,edgecolor="none",zorder=5))

    # Default AutoAnchor uses a deterministic catalog-level density gate.
    box(132,47.8,42,5.8,edge="#C8AD83",radius=.8)
    text(153,50.7,r"$\eta_T=|\mathcal{U}_T|/|\mathcal{I}_T|\ \rightarrow\rho_{\mathrm{eff}}$",8,AMBER)
    arrow([(153,47.7),(153,44.4)],AMBER)
    box(133,33.6,40,10.5,edge=AMBER)
    text(153,41.3,"Prior interpolation",7.9,AMBER,"bold")
    text(153,36.8,r"$(1-\rho_{\mathrm{eff}})z_{c,i}+\rho_{\mathrm{eff}}x_i$",8)
    box(142.8,25.7,20.4,5.1,edge=AMBER,radius=.8)
    text(153,28.25,r"$\ell_2$ norm",7.3,AMBER)
    arrow([(153,33.5),(153,31)],AMBER)
    arrow([(13,29.5),(13,16.6),(130.4,16.6),(130.4,36.8),(132.4,36.8)],MUTED)
    text(70,18.2,r"Frozen semantic prior $x_i$",7.3,MUTED)
    arrow([(153,25.6),(153,13.3)],AMBER)
    text(167.2,21.3,r"$z^{\mathrm{anchor}}_{c,i}$",8.3,AMBER)
    text(143.2,10.2,r"$e_u\ \rightarrow\ s(u,i)=e_u^{\top}z^{\mathrm{anchor}}_{c,i}\ \rightarrow\ \mathcal{L}_{\mathrm{BPR}}$",8.2)
    text(3,10.2,r"Stays local: histories, profiles $p_c$, user embeddings $e_u$",7.1,MUTED,ha="left")
    # The adapters' internal residual is distinct from the head-path bypass.
    box(0,-11,106,15,"#F4F6F9","#C8D0DA")
    text(2,-.3,"Inside each adapter",7.6,INK,"bold","left")
    text(2,-5.5,r"$A_h$ and $A_l$: two residual blocks",6.9,MUTED,ha="left")
    text(48,-5,r"$v$",8)
    box(56,-8,13,6,edge=LINE,radius=.8)
    text(62.5,-5,r"$F_\ell$",8)
    node(80,-5,r"$+$")
    arrow([(50,-5),(55.4,-5)])
    arrow([(69,-5),(77.3,-5)])
    arrow([(52,-5),(52,1.7),(80,1.7),(80,-2.3)])
    text(87.5,1.1,r"$\times 2$",7.5,MUTED)
    arrow([(82.7,-5),(88,-5)])
    box(89,-8,14,6,edge=LINE,radius=.8)
    text(96,-5,"norm",7)
    ax.add_patch(Circle((52,-5),.4,facecolor=INK,edgecolor="none",zorder=5))
    box(110,-11,70,15,"#F4F6F9","#C8D0DA")
    text(145,-.1,"After federation",7.7,INK,"bold")
    text(145,-4.8,"40 local proximal steps",7.3)
    text(145,-8.7,"No further parameter exchange",7.0,MUTED)

    # Check for clipped text at the final physical size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for t in texts:
        e = t.get_window_extent(renderer)
        if not (bounds.x0<=e.x0<=e.x1<=bounds.x1 and bounds.y0<=e.y0<=e.y1<=bounds.y1):
            raise RuntimeError(f"Text leaves canvas: {t.get_text()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=dpi,metadata={"Software":"Matplotlib / FedPASA framework"})
    plt.close(fig)
    print(f"Saved {output.name}: 7.04 x 4.10 in, {dpi} dpi")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"paper_www"/"figures"/"framework.png")
    p.add_argument("--dpi",type=int,default=600)
    args=p.parse_args()
    if args.output.suffix.lower()!=".png":
        p.error("The framework is exported only as PNG.")
    render(args.output.resolve(),args.dpi)
