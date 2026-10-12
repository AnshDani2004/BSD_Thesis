"""Figures and markdown tables for Extension 2. Reads results/<phase>/*.csv only.
Usage (from final_thesis_work/): MPLBACKEND=Agg python3 ext2/make_figures.py confirmatory
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = sys.argv[1] if len(sys.argv) > 1 else 'confirmatory'
RES = os.path.join(HERE, 'results', PHASE)

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
DET = [('hmm_ret', 'Return-only HMM', '#2a78d6', 'o'), ('volhmm', 'Vol-HMM', '#eb6834', 's'),
       ('volhmm_cusum', 'Vol-HMM + CUSUM', '#1baf7a', '^'), ('bocpd_rl', 'BOCPD (run length)', '#eda100', 'D'),
       ('bocpd_mean', 'BOCPD (segment mean)', '#e87ba4', 'v'), ('jm', 'Jump model', '#008300', 'P')]
M_COL = {1.0: '#86b6ef', 2.0: '#3987e5', 3.0: '#1c5cab', 5.0: '#0d366b'}   # ordinal blue ramp

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': AXIS,
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2})


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def frontier_panel(ax, rows, aii, shock, D, legend=False):
    for d, lab, col, mk in DET:
        pts = sorted([(float(r['fa_per_1000']), float(r['mean_censored'])) for r in rows
                      if r['aii'] == aii and r['shock'] == shock and r['D'] == D and r['detector'] == d
                      and r['level'] != 'native'])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1, label=lab)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(['1', '2', '5', '10'])
    ax.set_ylim(bottom=0)
    if legend:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)


def main():
    fr = load('a_frontier.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, aii in zip(axes, ['0.95', '0.97', '0.99']):
        frontier_panel(ax, fr, aii, 'thesis', '125', legend=(aii == '0.95'))
        ax.set_title(f'A_ii = {aii}', color=INK, fontsize=10, loc='left')
        ax.set_xlabel('realised false alarms per 1,000 pre-break steps (log scale)')
    axes[0].set_ylabel('mean detection delay, steps (censored at 50)')
    fig.suptitle(f'Detection frontier, thesis shock (Student-t(3), location -0.10, scale 0.30) lasting to the end; '
                 f'{PHASE} seeds, NOT YET REPLICATED', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_frontier.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for i, shock in enumerate(['small', 'thesis', 'large']):
        for j, D in enumerate(['5', '20', '125']):
            ax = axes[i, j]
            frontier_panel(ax, fr, '0.95', shock, D, legend=(i == 0 and j == 0))
            ax.set_title(f'shock {shock}, D = {D} (censored at {min(int(D), 50)})', color=INK, fontsize=9, loc='left')
            if i == 2:
                ax.set_xlabel('false alarms per 1,000 pre-break steps')
            if j == 0:
                ax.set_ylabel('mean censored delay, steps')
    fig.suptitle(f'Detection frontier by shock cell, A_ii = 0.95; {PHASE} seeds, NOT YET REPLICATED',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_frontier_shocks.png'), dpi=150)
    plt.close(fig)

    gb = load('b_gap.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for i, v in enumerate(['V1_code', 'V2_pure']):
        for j, dg in enumerate(['diffusive', 'jump']):
            ax = axes[i, j]
            for m, col in M_COL.items():
                pts = sorted([(int(r['n_rebalance']), float(r['breach_rate'])) for r in gb
                              if r['dgp'] == dg and r['variant'] == v and r['peak'] == 'peak_step'
                              and r['model'] == 'bounded' and float(r['m']) == m])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker='o', markersize=6,
                        markeredgecolor=SURFACE, label=f'm = {m:g}')
                ax.annotate(f'm = {m:g}', (pts[-1][0], pts[-1][1]), xytext=(4, 0), textcoords='offset points',
                            fontsize=7, color=INK2, va='center')
            ax.set_xscale('log')
            ax.set_xticks([1, 2, 5, 10, 50])
            ax.set_xticklabels(['1', '2', '5', '10', '50'])
            ax.set_ylim(0, 1)
            ax.set_title(f'{v}, {dg} DGP', color=INK, fontsize=10, loc='left')
            if i == 1:
                ax.set_xlabel('rebalances per step n (peak tracked at step ends)')
            if j == 0:
                ax.set_ylabel('share of paths breaching the 20% floor')
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f'CPPI floor-breach rate, bounded returns; {PHASE} seeds, NOT YET REPLICATED',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_gap.png'), dpi=150)
    plt.close(fig)

    # markdown table (table view of the figures)
    lines = ['| A_ii | detector | target | threshold | realised FA /1000 | mean delay | median | P90 | never detected |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in fr:
        if r['shock'] == 'thesis' and r['D'] == '125':
            lines.append(f"| {r['aii']} | {r['detector']} | {r['level']} | {float(r['threshold']):.4g} | "
                         f"{float(r['fa_per_1000']):.2f} | {float(r['mean_censored']):.2f} | "
                         f"{float(r['median_censored']):.1f} | {float(r['p90_censored']):.1f} | "
                         f"{float(r['share_never_detected']):.3f} |")
    with open(os.path.join(RES, 'table_frontier_thesis_D125.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('figures written to', RES)


if __name__ == '__main__':
    main()
