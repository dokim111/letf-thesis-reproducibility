"""Count overlapping evaluation windows in existing stored forecasts.

No model is estimated and no input file is modified. The original stock file
is read only for the trading-date calendar. Example:
  python check_overlap_windows.py /path/to/3_통계학 --output overlap_windows.json
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stats_dir', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('overlap_windows.json'))
    args = parser.parse_args()
    base = args.stats_dir.resolve()
    data = base / 'krx_stock_daily_long_20000104_20261113.csv'
    stored = base / 'symmetric_extension_results/forecasts_log.csv.gz'
    before = {str(p): digest(p) for p in (data, stored)}
    calendar = pd.read_csv(data, usecols=['stock', 'date'])
    forecasts = pd.read_csv(stored)
    horizon, listing = 20, 20260527
    groups = []
    model_calendars_equal = True

    for stock, raw_dates in calendar.groupby('stock'):
        dates = np.sort(raw_dates.date.unique())
        date_to_index = {int(date): i for i, date in enumerate(dates)}
        selected = forecasts[(forecasts.stock == stock) &
                             (forecasts.h == horizon) &
                             (forecasts.date >= listing)]
        for beta, beta_rows in selected.groupby('beta'):
            rows = beta_rows[beta_rows.model == 'M1'].sort_values('date').copy()
            assert not rows.date.duplicated().any()
            comparable = ['date', 'end_date', 'event_affected']
            expected = rows[comparable].reset_index(drop=True)
            for _, model_rows in beta_rows.groupby('model'):
                same = model_rows.sort_values('date')[comparable].reset_index(drop=True).equals(expected)
                model_calendars_equal &= same
                assert same
            rows['t'] = rows.date.map(date_to_index)
            end_t = rows.end_date.map(date_to_index)
            assert (end_t.to_numpy() - rows.t.to_numpy() == horizon).all()
            assert (np.diff(rows.t.to_numpy()) == 1).all()
            first_t = int(rows.t.iloc[0])
            excluded = rows[rows.event_affected]
            remaining = rows[~rows.event_affected]
            ts = remaining.t.to_numpy()
            blocks = np.split(ts, np.flatnonzero(np.diff(ts) > 1) + 1)

            # Phase is fixed before event exclusion, as in the manuscript.
            phase_counts = [int(np.sum((ts - first_t) % horizon == k))
                            for k in range(horizon)]
            # Equal-length target intervals [t+1, t+h] are optimally selected
            # by earliest-finish greedy scheduling. A next origin t+h is valid.
            picked, next_origin = [], -1
            for t in ts:
                if t >= next_origin:
                    picked.append(int(t))
                    next_origin = int(t + horizon)
            interval = lambda t: {
                'origin': int(dates[t]),
                'target_first': int(dates[t + 1]),
                'target_last': int(dates[t + horizon]),
            }
            groups.append({
                'stock': stock, 'beta': int(beta), 'h': horizon,
                'all_post_origins': len(rows),
                'event_excluded': len(excluded),
                'remaining_origins': len(remaining),
                'remaining_over_h_arithmetic_ratio': len(remaining) / horizon,
                'phase_anchor_origin': int(rows.date.iloc[0]),
                'phase_counts': phase_counts,
                'phase_count_distribution': {
                    str(n): phase_counts.count(n) for n in sorted(set(phase_counts))
                },
                'max_disjoint_target_windows': len(picked),
                'one_maximal_selection': [interval(t) for t in picked],
                'remaining_origin_blocks': [{
                    'n': len(block),
                    'origin_first': int(dates[block[0]]),
                    'origin_last': int(dates[block[-1]]),
                    'target_union_first': int(dates[block[0] + 1]),
                    'target_union_last': int(dates[block[-1] + horizon]),
                } for block in blocks],
            })

    common_keys = ['all_post_origins', 'event_excluded', 'remaining_origins',
                   'phase_counts', 'max_disjoint_target_windows',
                   'remaining_origin_blocks', 'one_maximal_selection']
    common = all(all(g[k] == groups[0][k] for k in common_keys) for g in groups)
    assert len(groups) == 4 and common
    assert all(g['remaining_origins'] == 54 and
               g['phase_count_distribution'] == {'2': 6, '3': 14} and
               g['max_disjoint_target_windows'] == 4 and
               [b['n'] for b in g['remaining_origin_blocks']] == [25, 29]
               for g in groups)
    after = {str(p): digest(p) for p in (data, stored)}
    assert before == after
    report = {
        'source_stats_dir': str(base),
        'raw_calendar_path': str(data),
        'stored_forecasts_path': str(stored),
        'input_sha256': before,
        'inputs_unchanged': before == after,
        'refitted': False,
        'raw_prices_used_for_model_fitting': False,
        'model_date_grids_equal': bool(model_calendars_equal),
        'both_stocks_and_betas_share_calendar_counts': bool(common),
        'groups': groups,
        'interpretation': {
            'fixed_phase_count': '2 or 3 observations per phase; 14 phases have 3 and 6 phases have 2.',
            'global_maximum_disjoint_windows': 4,
            'not_an_effective_sample_size': True,
            'disjoint_targets_do_not_imply_independent_forecast_errors': True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'output': str(args.output.resolve()),
        'groups_checked': len(groups),
        'remaining': 54, 'phase_distribution': {'2': 6, '3': 14},
        'max_disjoint': 4, 'blocks': [25, 29], 'inputs_unchanged': True,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
