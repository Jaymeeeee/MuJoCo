"""Plot exact frozen v2 evaluation positions and any recorded trial outcomes.

Run with --ind-output RUN --ood-output RUN --output DIRECTORY. Before results
exist, all points are shown as pending. A complete result set adds success-rate
labels; partial results never present a final rate. No model or simulator loads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml

from mujoco_tools.paths import relocated_evaluation_path


SPLIT_COLORS = {'ind': '#276aa5', 'ood': '#8155a0'}
PENDING = {'pending', 'started_incomplete'}
PHYSICAL_FAILURES = {'timeout', 'contact_violation', 'task_success_with_contact_violations'}
ERRORS = {'runtime_error', 'interrupted', 'viewer_closed', 'infrastructure_failure', 'artifact_error'}


def load_split(output, split):
    """Read one frozen manifest and the newest available summary snapshot."""
    output = relocated_evaluation_path(output).resolve()
    manifest = yaml.safe_load((output / 'configs/positions_config.yaml').read_text(encoding='utf8'))
    rows = manifest['positions']
    if manifest['split'] != split or [row['name'] for row in rows] != [f'pos{i}' for i in range(100)]:
        raise ValueError(f'{output}: expected the {split} manifest with pos0 through pos99')
    for side in ('source', 'target'):
        points = np.asarray([row[f'{side}_xy'] for row in rows], dtype=float)
        if points.shape != (100, 2) or not np.isfinite(points).all():
            raise ValueError(f'{output}: invalid {side} XY positions')
    categories = {row['name']: 'pending' for row in rows}
    candidates = [output / name for name in ('summary.json', 'progress.json') if (output / name).exists()]
    summary = None
    if candidates:
        path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        summary = json.loads(path.read_text(encoding='utf8'))
        if summary.get('evaluation_split') != split or summary.get('episodes_requested') != 100:
            raise ValueError(f'{path}: result identity differs from the frozen split')
        expected = {row['name']: row for row in rows}
        seen = set()
        for result in summary['results']:
            name = result['position']
            if name not in expected or name in seen:
                raise ValueError(f'{path}: unknown or duplicate result position {name}')
            seen.add(name)
            for key in ('seed', 'source_xy', 'target_xy'):
                if result[key] != expected[name][key]:
                    raise ValueError(f'{path}: {name}/{key} differs from frozen geometry')
            status = result['status']
            if status in PENDING:
                category = 'pending'
            elif status in ERRORS:
                category = 'error'
            elif result['success'] is True and status == 'success':
                category = 'success'
            elif result['success'] is False and status in PHYSICAL_FAILURES:
                category = 'failure'
            else:
                raise ValueError(f'{path}: inconsistent outcome for {name}: {status}')
            categories[name] = category
        if len(seen) != 100:
            raise ValueError(f'{path}: summary must cover all 100 requested trials')
        counts = Counter(categories.values())
        if summary.get('successes') != counts['success'] or bool(summary.get('complete')) != (counts['pending'] == 0):
            raise ValueError(f'{path}: aggregate counts differ from per-trial outcomes')
    return dict(manifest=manifest, categories=categories, counts=Counter(categories.values()), summary=summary)


def _status_label(data):
    counts = data['counts']
    if counts['pending']:
        return (f"Recorded {100 - counts['pending']}/100; successes {counts['success']}; "
                f"pending {counts['pending']}; errors {counts['error']}")
    return (f"Success {counts['success']}/100 ({counts['success']:.1f}%); "
            f"physical failures {counts['failure']}; errors {counts['error']}")


def plot_comparison(ind_output, ood_output, output, *, name='comparison', policy_label='ACT v2 RGB-D'):
    """Write PNG/PDF files under the explicitly supplied output directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FormatStrFormatter

    data = {'ind': load_split(ind_output, 'ind'), 'ood': load_split(ood_output, 'ood')}
    if data['ind']['manifest']['regions'] != data['ood']['manifest']['regions']:
        raise ValueError('InD and OOD must use the same source/target region definitions')
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError('--name must be a plain filename stem')
    destination = Path(output).expanduser().resolve()
    paths = [destination / f'{name}.{extension}' for extension in ('png', 'pdf')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Plot files already exist; choose a new --name or --output')
    figure, axes = plt.subplots(2, 2, figsize=(11, 10))
    try:
        for row_index, split in enumerate(('ind', 'ood')):
            split_data = data[split]
            manifest = split_data['manifest']
            categories = np.array([split_data['categories'][row['name']] for row in manifest['positions']])
            for column, side in enumerate(('source', 'target')):
                axis = axes[row_index, column]
                region = manifest['regions'][side]
                outer, inner = region['outer_bounds_m'], region['ood_bounds_m']
                center = np.asarray(region['center_xy'])
                axis.add_patch(Rectangle((outer['x'][0], outer['y'][0]),
                                         outer['x'][1] - outer['x'][0], outer['y'][1] - outer['y'][0],
                                         facecolor=SPLIT_COLORS[split] if split == 'ind' else '#c7ced5',
                                         edgecolor='#64717e', linewidth=1, alpha=.11, zorder=0))
                axis.add_patch(Rectangle((inner['x'][0], inner['y'][0]),
                                         inner['x'][1] - inner['x'][0], inner['y'][1] - inner['y'][0],
                                         facecolor='white' if split == 'ind' else '#eee7f4',
                                         edgecolor='#637181', linestyle='--', linewidth=1.1, zorder=1))
                axis.plot([outer['x'][0], outer['x'][1], outer['x'][1], outer['x'][0], outer['x'][0]],
                          [outer['y'][0], outer['y'][0], outer['y'][1], outer['y'][1], outer['y'][0]],
                          color='#82909e', linewidth=.9, zorder=2)
                points = np.asarray([row[f'{side}_xy'] for row in manifest['positions']])
                for category, marker, color, size in (
                        ('pending', 'o', SPLIT_COLORS[split], 19),
                        ('success', 'o', '#27824e', 24),
                        ('failure', 'x', '#bf4542', 26),
                        ('error', '^', '#cf851d', 33)):
                    selected = points[categories == category]
                    if len(selected):
                        style = {'facecolors': 'none', 'edgecolors': color} if category == 'pending' else {'color': color}
                        axis.scatter(selected[:, 0], selected[:, 1], s=size, marker=marker,
                                     linewidths=.9, alpha=.9,
                                     zorder=4 if category == 'success' else 3, **style)
                # Identical 13 cm spans and equal aspect preserve common metric
                # scale while displaying each panel in actual world coordinates.
                axis.set_xlim(center[0] - .065, center[0] + .065)
                axis.set_ylim(center[1] - .065, center[1] + .065)
                axis.set_aspect('equal', adjustable='box')
                axis.set_xticks(center[0] + np.array([-.05, 0, .05]))
                axis.set_yticks(center[1] + np.array([-.05, 0, .05]))
                axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axis.grid(color='#dfe4e9', linewidth=.5, zorder=0)
                axis.set_xlabel('World X (m)')
                axis.set_ylabel('World Y (m)')
                axis.set_title(f"{split.upper() if split == 'ood' else 'InD'} | {side.capitalize()}\n"
                               + _status_label(split_data), fontsize=10, loc='left')
        handles = [Line2D([], [], marker='o', markerfacecolor='none', color=SPLIT_COLORS[split],
                          linestyle='none', label=f'{split.upper()} pending') for split in ('ind', 'ood')]
        handles += [Line2D([], [], marker=marker, color=color, linestyle='none', label=label)
                    for marker, color, label in [('o', '#27824e', 'Success'), ('x', '#bf4542', 'Physical failure'),
                                                 ('^', '#cf851d', 'Runtime / artifact error')]]
        figure.suptitle(f'{policy_label} | frozen regional evaluation layouts', fontsize=15, weight='semibold')
        figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=5, frameon=False, fontsize=9)
        figure.text(.5, .025, 'Outer squares: 11 cm; central held-out squares: 4 cm. One trial per source-target pair.\n'
                     'Points show exact sampled positions; source and target panels share each trial outcome.',
                    ha='center', fontsize=9, color='#485562')
        figure.tight_layout(rect=(0, .065, 1, .90), h_pad=2.3)
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            with path.open('xb') as stream:
                figure.savefig(stream, format=path.suffix[1:], dpi=200, bbox_inches='tight', facecolor='white')
    finally:
        plt.close(figure)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ind-output', type=Path, required=True)
    parser.add_argument('--ood-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', default='comparison', help='New filename stem; existing plots are preserved')
    parser.add_argument('--policy-label', default='ACT v2 RGB-D', help='Policy name displayed in the plot title')
    args = parser.parse_args(argv)
    paths = plot_comparison(args.ind_output, args.ood_output, args.output, name=args.name,
                            policy_label=args.policy_label)
    print(json.dumps(dict(plots=[str(path) for path in paths])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
