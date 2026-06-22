from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

ROOT = Path('/home/qian/Code/FaasANNS')
OUT = ROOT / 'work_space' / 'compare' / 'image'
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ('full_search', ROOT / 'baseline/functions/full_search/test/result/run_queries_gist.csv'),
    ('full_search', ROOT / 'baseline/functions/full_search/test/result/run_queries_sift100w.csv'),
    ('two_stage', ROOT / 'baseline/functions/Two_stage_search/test/result/run_queries_gist.csv'),
    ('two_stage', ROOT / 'baseline/functions/Two_stage_search/test/result/run_queries_sift100w.csv'),
    ('faasann', ROOT / 'logs/run_queries_gist.csv'),
    ('faasann', ROOT / 'logs/run_queries_sift100w.csv'),
]

METHOD_LABELS = {
    'full_search': 'Full Search',
    'two_stage': 'Two-stage',
    'faasann': 'FaasANN',
}
DATASET_LABELS = {
    'gist': 'GIST',
    'sift100w': 'SIFT100W',
}
STATE_LABELS = {
    'cold': 'Cold start',
    'warm': 'Warm',
}
COLORS = {
    'cold': '#d95f02',
    'warm': '#1b9e77',
}


def read_rows() -> list[dict]:
    rows: list[dict] = []
    for method, path in SOURCES:
        if not path.exists():
            print(f'warning: missing {path}')
            continue
        with path.open(newline='', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                dataset = row.get('dataset') or dataset_from_name(path.name)
                cold_start_num = int(float(row.get('cold_start_num') or 0))
                latency_field = 'avg_client_ms' if row.get('avg_client_ms') not in {None, ''} else 'avg_total_ms'
                rows.append({
                    'timestamp': row.get('timestamp', ''),
                    'dataset': dataset,
                    'method': method,
                    'method_label': METHOD_LABELS[method],
                    'cold_state': 'cold' if cold_start_num > 0 else 'warm',
                    'cold_start_num': cold_start_num,
                    'latency_ms': float(row[latency_field]),
                    'latency_source': latency_field,
                    'qps': float(row['qps_client']),
                    'recall': float(row.get('recall') or 0.0),
                    'query_count': int(float(row.get('query_count') or 0)),
                    'candidate_k': row.get('candidate_k', ''),
                    'ef_search': row.get('ef_search', ''),
                    'source_file': str(path.relative_to(ROOT)),
                })
    return rows


def dataset_from_name(name: str) -> str:
    if 'gist' in name:
        return 'gist'
    if 'sift100w' in name:
        return 'sift100w'
    return 'unknown'


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['method'], row['cold_state']), []).append(row)

    out: list[dict] = []
    for (dataset, method, state), items in sorted(grouped.items()):
        out.append({
            'dataset': dataset,
            'method': method,
            'method_label': METHOD_LABELS[method],
            'cold_state': state,
            'run_count': len(items),
            'latency_ms': mean(item['latency_ms'] for item in items),
            'qps': mean(item['qps'] for item in items),
            'recall': mean(item['recall'] for item in items),
            'cold_start_num_avg': mean(item['cold_start_num'] for item in items),
            'latency_source': '+'.join(sorted({item['latency_source'] for item in items})),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if value >= 100:
        return f'{value:.0f}'
    if value >= 10:
        return f'{value:.1f}'
    return f'{value:.2f}'


def svg_bar_chart(rows: list[dict], metric: str, title: str, ylabel: str, path: Path) -> None:
    datasets = ['gist', 'sift100w']
    methods = ['full_search', 'two_stage', 'faasann']
    states = ['cold', 'warm']
    lookup = {(r['dataset'], r['method'], r['cold_state']): r for r in rows}
    values = [r[metric] for r in rows]
    max_value = max(values) if values else 1.0
    top = max_value * 1.18

    width = 1200
    height = 680
    margin_l = 86
    margin_r = 32
    margin_t = 70
    margin_b = 130
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    dataset_gap = 78
    method_gap = 22
    state_gap = 4
    group_w = (plot_w - dataset_gap) / len(datasets)
    method_w = (group_w - method_gap * (len(methods) - 1)) / len(methods)
    bar_w = min(46, (method_w - state_gap) / 2)

    def x_for(di: int, mi: int, si: int) -> float:
        dataset_x = margin_l + di * (group_w + dataset_gap)
        method_x = dataset_x + mi * (method_w + method_gap)
        pair_w = 2 * bar_w + state_gap
        return method_x + (method_w - pair_w) / 2 + si * (bar_w + state_gap)

    def y_for(value: float) -> float:
        return margin_t + plot_h - (value / top) * plot_h

    ticks = nice_ticks(top, 5)
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#222">{escape(title)}</text>')
    parts.append(f'<text x="22" y="{margin_t + plot_h/2}" transform="rotate(-90 22 {margin_t + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#333">{escape(ylabel)}</text>')

    for tick in ticks:
        y = y_for(tick)
        parts.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{width-margin_r}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l-10}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{escape(fmt(tick))}</text>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{width-margin_r}" y2="{margin_t+plot_h}" stroke="#333" stroke-width="1.3"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#333" stroke-width="1.3"/>')

    for di, dataset in enumerate(datasets):
        dataset_center = margin_l + di * (group_w + dataset_gap) + group_w / 2
        parts.append(f'<text x="{dataset_center:.2f}" y="{height-30}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#222">{DATASET_LABELS[dataset]}</text>')
        for mi, method in enumerate(methods):
            method_center = margin_l + di * (group_w + dataset_gap) + mi * (method_w + method_gap) + method_w / 2
            parts.append(f'<text x="{method_center:.2f}" y="{height-62}" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">{escape(METHOD_LABELS[method])}</text>')
            for si, state in enumerate(states):
                item = lookup.get((dataset, method, state))
                x = x_for(di, mi, si)
                if item is None:
                    parts.append(f'<rect x="{x:.2f}" y="{margin_t+plot_h-1}" width="{bar_w:.2f}" height="1" fill="#cccccc"/>')
                    continue
                value = float(item[metric])
                y = y_for(value)
                h = margin_t + plot_h - y
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="2" fill="{COLORS[state]}"/>')
                parts.append(f'<text x="{x+bar_w/2:.2f}" y="{max(margin_t+12, y-6):.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#222">{escape(fmt(value))}</text>')
                parts.append(f'<title>{DATASET_LABELS[dataset]} {METHOD_LABELS[method]} {STATE_LABELS[state]}: {value:.3f}</title>')

    legend_x = width - margin_r - 260
    legend_y = 50
    for i, state in enumerate(states):
        x = legend_x + i * 130
        parts.append(f'<rect x="{x}" y="{legend_y}" width="18" height="12" fill="{COLORS[state]}"/>')
        parts.append(f'<text x="{x+25}" y="{legend_y+11}" font-family="Arial" font-size="13" fill="#333">{STATE_LABELS[state]}</text>')

    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="11" fill="#666">Cold start means cold_start_num &gt; 0. Values are averages across rows in each group.</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def nice_ticks(max_value: float, count: int) -> list[float]:
    if max_value <= 0:
        return [0]
    raw = max_value / count
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1:
        step = 1 * mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    ticks = []
    value = 0.0
    while value <= max_value * 1.001:
        ticks.append(value)
        value += step
    return ticks


def main() -> None:
    raw = read_rows()
    summary = aggregate(raw)
    write_csv(OUT / 'compare_raw_rows.csv', raw)
    write_csv(OUT / 'compare_summary.csv', summary)
    svg_bar_chart(summary, 'latency_ms', 'Latency by Dataset, Method, and Cold Start State', 'Average latency (ms)', OUT / 'latency_by_method_cold_state.svg')
    svg_bar_chart(summary, 'qps', 'QPS by Dataset, Method, and Cold Start State', 'Client QPS', OUT / 'qps_by_method_cold_state.svg')
    print(f'wrote {OUT / "compare_raw_rows.csv"}')
    print(f'wrote {OUT / "compare_summary.csv"}')
    print(f'wrote {OUT / "latency_by_method_cold_state.svg"}')
    print(f'wrote {OUT / "qps_by_method_cold_state.svg"}')


if __name__ == '__main__':
    main()
