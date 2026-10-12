"""Digest of the numbers quoted in RESULTS.md, read only from results/<phase>/ files.
Usage (from final_thesis_work/): python3 ext2/summarize.py confirmatory > ext2/results/confirmatory/digest.json
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = sys.argv[1] if len(sys.argv) > 1 else 'confirmatory'
RES = os.path.join(HERE, 'results', PHASE)


def rows(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def f(x, nd=4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


out = {'phase': PHASE}
out['primary'] = json.load(open(os.path.join(RES, 'primary.json')))
fr = rows('a_frontier.csv')
out['a_frontier_thesis_D125'] = [
    {k: f(r[k]) for k in ('aii', 'detector', 'level', 'fa_per_1000', 'onsets_per_1000', 'mean_censored',
                          'median_censored', 'p90_censored', 'share_never_detected',
                          'placebo_share_alarm_in_125_175_no_shock')}
    for r in fr if r['shock'] == 'thesis' and r['D'] == '125']
# realised false-alarm range per target (all cells share pre-break paths)
fa = {}
for r in fr:
    fa.setdefault(r['level'], []).append(float(r['fa_per_1000']))
out['a_realised_fa_range'] = {k: [round(min(v), 3), round(max(v), 3)] for k, v in fa.items()}
ct = rows('a_contrasts.csv')
summ = {}
for r in ct:
    if r['level'] == 'native':
        continue
    s = summ.setdefault(r['contrast'], {'cells': 0, 'a_faster_sig_q05': 0, 'a_slower_sig_q05': 0, 'min': 1e9, 'max': -1e9})
    s['cells'] += 1
    md = float(r['mean_diff'])
    s['min'] = round(min(s['min'], md), 3)
    s['max'] = round(max(s['max'], md), 3)
    sig = (r['q_bh_secondary'] != '' and float(r['q_bh_secondary']) < 0.05) or r['primary'] == 'True'
    if sig and md < 0:
        s['a_faster_sig_q05'] += 1
    if sig and md > 0:
        s['a_slower_sig_q05'] += 1
out['a_contrast_summary_all_matched_cells'] = summ
out['a_contrasts_primary_cell_all'] = [{k: f(r[k]) for k in ('contrast', 'mean_diff', 'ci_lo', 'ci_hi', 'p_signflip',
                                                              'share_a_faster', 'share_tied')}
                                       for r in ct if r['aii'] == '0.95' and r['level'] == '5' and r['shock'] == 'thesis'
                                       and r['D'] == '125']
dd = rows('a_downstream.csv')
out['a_downstream_thesis_D125_aii095'] = [
    {k: f(r[k]) for k in ('detector', 'level', 'ruin_rate', 'practical_ruin_rate', 'median_terminal_wealth', 'mtw_ci_lo',
                          'mtw_ci_hi', 'breach_rate', 'nofloor_median_terminal_wealth', 'unbounded_ruin_rate',
                          'unbounded_practical_ruin_rate', 'unbounded_median_terminal_wealth', 'unbounded_breach_rate')}
    for r in dd if r['shock'] == 'thesis' and r['D'] == '125' and r['aii'] in ('0.95', '') and r['level'] in ('5', 'native', '')]
out['a_downstream_max_ruin_bounded'] = max(float(r['ruin_rate']) for r in dd)
best = {}
for r in dd:
    key = f"{r['shock']}/D{r['D']}"
    best.setdefault(key, []).append((float(r['median_terminal_wealth']), r['detector'], r['aii'], r['level']))
out['a_downstream_top3_by_median_wealth_per_shock_cell'] = {k: sorted(v, reverse=True)[:3] for k, v in best.items()}
gb = rows('b_gap.csv')
out['b_grid'] = [{k: f(r[k]) for k in ('dgp', 'variant', 'peak', 'model', 'm', 'n_rebalance', 'breach_rate', 'breach_ci_lo',
                                       'breach_ci_hi', 'breach_fine_rate', 'share_breaches_lambda_at_min',
                                       'median_terminal_wealth', 'ruin_rate')}
                 for r in gb if r['peak'] in ('peak_step', '') or r['n_rebalance'] in ('1', '50')]
ca = rows('c_agents.csv')
out['c_agents'] = [{k: f(r[k]) for k in ('scenario', 'model', 'agent', 'ruin_rate', 'ruin_ci_lo', 'ruin_ci_hi',
                                         'practical_ruin_rate', 'median_terminal_wealth', 'mtw_ci_lo', 'mtw_ci_hi',
                                         'breach_rate')} for r in ca]
out['run_info'] = json.load(open(os.path.join(RES, 'run_info.json')))
print(json.dumps(out, indent=1))
