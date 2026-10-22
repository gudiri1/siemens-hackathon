"""Plot a processed anatomy: centerlines coloured by branch (AP and lateral projections) with
branch labels, the default start, and the lumen radius along every branch.

    python scripts/plot_anatomy.py --model 0248_H_AOCERE_CAS [--out results/anatomy.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stroke_nav.data.anatomy import Anatomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a = Anatomy.load(args.model, args.variant)
    out = Path(args.out or f"results/{args.model}_anatomy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(16, 9))
    ax_ap = fig.add_subplot(1, 3, 1)
    ax_lat = fig.add_subplot(1, 3, 2)
    ax_r = fig.add_subplot(1, 3, 3)
    for b, name in enumerate(a.branch_names):
        nodes = a.branch_nodes(name)
        p = a.nodes[nodes]
        c = cmap(b % 20)
        # AP view: patient's left on the right of the image (RAS: -x to the right)
        ax_ap.plot(-p[:, 0], p[:, 2], color=c, lw=2)
        ax_lat.plot(p[:, 1], p[:, 2], color=c, lw=2)
        mid = p[len(p) // 2]
        ax_ap.text(-mid[0], mid[2], name, fontsize=7, color=c)
        ax_r.plot(a.arc[nodes] - a.arc[nodes].min(), a.radius[nodes], color=c, lw=1.2, label=name)
    s = a.nodes[a.default_start()]
    for ax, (i, j, sx) in ((ax_ap, (0, 2, -1)), (ax_lat, (1, 2, 1))):
        ax.plot(sx * s[i], s[j], "o", color="#2563eb", ms=9, label="default start")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    j = a.junctions()
    ax_ap.plot(-a.nodes[j, 0], a.nodes[j, 2], "k.", ms=4, label="junctions")
    ax_ap.set_title("AP view (patient left on right)")
    ax_ap.set_xlabel("-x (mm)")
    ax_ap.set_ylabel("z (mm)")
    ax_ap.legend(loc="lower left", fontsize=8)
    ax_lat.set_title("Lateral view")
    ax_lat.set_xlabel("y (mm, anterior)")
    ax_r.set_title("Lumen radius along each branch")
    ax_r.set_xlabel("arc length along branch (mm)")
    ax_r.set_ylabel("radius (mm)")
    ax_r.set_yscale("log")
    ax_r.grid(alpha=0.3)
    ax_r.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{a.name}: {len(a.branch_names)} branches, {len(a.nodes)} nodes, {len(j)} junctions")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    for name in a.branch_names:
        nodes = a.branch_nodes(name)
        print(f"  {name:<26} length {a.arc[nodes].max() - a.arc[nodes].min():6.1f} mm   "
              f"median radius {np.median(a.radius[nodes]):5.2f} mm")


if __name__ == "__main__":
    main()
