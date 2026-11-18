"""Figures for the DRCP revision (GBD 2023, T1D, South Asia, ages under 20).
No titles inside figures; legends are in the manuscript. Vector PDF plus PNG/JPG at 1000 dpi,
about 190 mm wide (Elsevier full-page width). Lettering at least 7 pt."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image
from common import *
from trend import joinpoint, YEARS

OUT = "figs"
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#222222", "text.color": "#222222", "pdf.fonttype": 42,
})
COL = {"<5 years": "#009E73", "5-9 years": "#0072B2", "10-14 years": "#C98300",
       "15-19 years": "#A8327E", "<20 years": "#222222"}
MARK = {"<5 years": "^", "5-9 years": "o", "10-14 years": "s", "15-19 years": "D", "<20 years": None}
LS = {a: "-" for a in AGES}; LS["<20 years"] = (0, (4, 2))
SEXCOL = {"Male": "#3B6FB6", "Female": "#D55E00"}
NAME = {"<5 years": "Under 5 years", "5-9 years": "5–9 years", "10-14 years": "10–14 years",
        "15-19 years": "15–19 years", "<20 years": "Under 20 years"}

b, pop = load()


def num(v, dec):
    return f"{v:,.{dec}f}".replace("-", "−")


def yfmt(dec):
    return FuncFormatter(lambda v, _: num(v, dec))


def letter(ax, s, x=-0.17):
    ax.text(x, 1.05, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def xaxis(ax):
    ax.set_xlim(1999.3, 2023.7)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{OUT}/{name}.png", dpi=1000, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    Image.MAX_IMAGE_PIXELS = None
    Image.open(f"{OUT}/{name}.png").convert("RGB").save(f"{OUT}/{name}.jpg", quality=95, subsampling=0, dpi=(1000, 1000))
    plt.close(fig)


def rate_panel(ax, measure, ylabel, dec, loc="South Asia", legend=False):
    for a in AGES:
        s = series(b, measure, a, loc=loc)
        ax.fill_between(s.year, s.lower, s.upper, color=COL[a], alpha=0.09, lw=0)
        ax.plot(s.year, s.val, color=COL[a], ls=LS[a], lw=1.1, marker=MARK[a], ms=2.2,
                markevery=3, label=NAME[a])
    ax.set_ylabel(ylabel); xaxis(ax); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(yfmt(dec)); ax.yaxis.set_major_locator(MaxNLocator(5))


# ---------------- Figure 1: rates by age ----------------
def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(194 * MM, 74 * MM))
    rate_panel(axs[0], "Deaths", "Death rate per 100,000", 1)
    rate_panel(axs[1], "Prevalence", "Prevalence per 100,000", 0)
    rate_panel(axs[2], "Incidence", "Incidence per 100,000", 1)
    for ax, l in zip(axs, "abc"):
        letter(ax, l); ax.set_xlabel("Year")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.2)
    save(fig, "Figure_1")


# ---------------- Figure 2: joinpoint for death rates ----------------
def fig2():
    fig, axs = plt.subplots(2, 3, figsize=(194 * MM, 118 * MM))
    axs = axs.ravel()
    for i, a in enumerate(AGES):
        ax = axs[i]
        s = series(b, "Deaths", a)
        j = joinpoint(s.val.values)
        ax.plot(s.year, s.val, ls="none", marker="o", ms=2.6, color=COL[a], alpha=0.45, mec="none")
        ax.plot(YEARS, j["fitted"], color=COL[a], lw=1.4)
        for t in j["taus"]:
            k = int(t - 2000)
            ax.plot(t, j["fitted"][k], marker="o", ms=4.2, mfc="white", mec=COL[a], mew=1.1, zorder=4)
        lo, hi = s.val.min(), s.val.max(); pad = (hi - lo)
        ax.set_ylim(lo - 0.26 * pad, hi + 0.30 * pad)
        segs = j["ar1"]["segs"]
        for n_, (s0, s1, apc, l, h, p) in enumerate(segs):
            txt = f"{'+' if apc > 0 else ''}{num(apc, 1)}%{'*' if p < 0.05 else ''}"
            if s1 - s0 <= 3 and n_ == len(segs) - 1:      # short final segment: below the line end
                yv = j["fitted"][-1]
                ax.text(2023.6, yv - 0.06 * pad, txt, ha="right", va="top", fontsize=7, color="#222222")
            else:
                xm = (s0 + s1) / 2
                yv = np.interp(xm, YEARS, j["fitted"])
                ax.text(xm, yv + 0.08 * pad, txt, ha="center", va="bottom", fontsize=7, color="#222222")
        ax.set_title(NAME[a], fontsize=7.5, loc="left", pad=3)
        xaxis(ax); ax.yaxis.set_major_formatter(yfmt(2)); ax.yaxis.set_major_locator(MaxNLocator(4))
        if i % 3 == 0:
            ax.set_ylabel("Death rate per 100,000")
        if i >= 2:
            ax.set_xlabel("Year")
        letter(ax, "abcde"[i], x=-0.2)
    ax = axs[5]; ax.axis("off")
    ax.plot([], [], ls="none", marker="o", ms=2.6, color="#777777", alpha=0.6, label="Annual GBD estimate")
    ax.plot([], [], color="#777777", lw=1.4, label="Fitted joinpoint model")
    ax.plot([], [], ls="none", marker="o", ms=4.2, mfc="white", mec="#777777", mew=1.1, label="Joinpoint")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.95))
    ax.text(0.02, 0.42, "Labels show the annual percentage\nchange for each segment.\n* p < 0.05 (Prais–Winsten)",
            transform=ax.transAxes, fontsize=7, va="top")
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, "Figure_2")


# ---------------- Figure 3: death rates by sex ----------------
def fig3():
    fig, axs = plt.subplots(2, 3, figsize=(194 * MM, 112 * MM))
    axs = axs.ravel()
    for i, a in enumerate(AGES):
        ax = axs[i]
        for sex in ["Female", "Male"]:
            s = series(b, "Deaths", a, sex=sex)
            ax.fill_between(s.year, s.lower, s.upper, color=SEXCOL[sex], alpha=0.14, lw=0)
            ax.plot(s.year, s.val, color=SEXCOL[sex], lw=1.2, label=sex,
                    marker="o" if sex == "Female" else "s", ms=2.2, markevery=3)
        ax.set_title(NAME[a], fontsize=7.5, loc="left", pad=3)
        ax.set_ylim(bottom=0); xaxis(ax)
        ax.yaxis.set_major_formatter(yfmt(2)); ax.yaxis.set_major_locator(MaxNLocator(4))
        if i % 3 == 0:
            ax.set_ylabel("Death rate per 100,000")
        if i >= 2:
            ax.set_xlabel("Year")
        letter(ax, "abcde"[i], x=-0.2)
    ax = axs[5]; ax.axis("off")
    h, lab = axs[0].get_legend_handles_labels()
    ax.legend(h, lab, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.95))
    ax.text(0.02, 0.55, "Shaded bands show 95%\nuncertainty intervals.", transform=ax.transAxes, fontsize=7, va="top")
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, "Figure_3")


# ---------------- Appendix Fig A.1: absolute numbers (stacked, non-overlapping) ----------------
def figA1():
    fig, axs = plt.subplots(1, 2, figsize=(194 * MM, 76 * MM))
    for ax, m, lab, scale, dec in [(axs[0], "Deaths", "Deaths", 1, 0),
                                   (axs[1], "Prevalence", "Prevalent cases (thousands)", 1000, 0)]:
        base = np.zeros(24)
        for a in BANDS:
            v = series(b, m, a, metric="Number").val.values / scale
            ax.fill_between(YEARS, base, base + v, color=COL[a], alpha=0.75, lw=0, label=NAME[a])
            base = base + v
        tot = series(b, m, "<20 years", metric="Number")
        ax.plot(tot.year, tot.val / scale, color="#222222", lw=1.0, ls=LS["<20 years"], label="Under 20 years (total)")
        ax.set_ylabel(lab); ax.set_xlabel("Year"); xaxis(ax); ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(yfmt(dec)); ax.yaxis.set_major_locator(MaxNLocator(5))
    letter(axs[0], "a"); letter(axs[1], "b")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.5)
    save(fig, "Figure_A1")


# ---------------- Appendix Fig A.2: country death rates by age ----------------
def figA2():
    fig, axs = plt.subplots(2, 3, figsize=(194 * MM, 112 * MM))
    axs = axs.ravel()
    for i, loc in enumerate(LOCS[1:]):
        ax = axs[i]
        for a in BANDS:
            s = series(b, "Deaths", a, loc=loc)
            ax.plot(s.year, s.val, color=COL[a], lw=1.1, marker=MARK[a], ms=2.0, markevery=3, label=NAME[a])
        ax.set_title(loc, fontsize=7.5, loc="left", pad=3)
        ax.set_ylim(bottom=0); xaxis(ax)
        ax.yaxis.set_major_formatter(yfmt(2)); ax.yaxis.set_major_locator(MaxNLocator(4))
        if i % 3 == 0:
            ax.set_ylabel("Death rate per 100,000")
        if i >= 2:
            ax.set_xlabel("Year")
        letter(ax, "abcde"[i], x=-0.2)
    ax = axs[5]; ax.axis("off")
    h, l = axs[0].get_legend_handles_labels()
    ax.legend(h, l, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.95))
    ax.text(0.02, 0.45, "Vertical axes differ\nbetween panels.", transform=ax.transAxes, fontsize=7, va="top")
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, "Figure_A2")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3(); figA1(); figA2()
    print("done")
