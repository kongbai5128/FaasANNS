from common import *
        
import time
import random
from collections.abc import Iterable, Callable
import scipy.signal

help_str = '''Parameters:
  o: outer distribution, can be gaussian, zipf, uniform or faasgraph (default: zipf)
  a1 (mu): (outer/low-pass) skewness for zipf, standard deviation for gaussian, discarded
  a2 (a): (inner/high-pass) skewness, 
  g: granularity, number of bins.(default: 10)
  n: number of samples
  duration: range of samples (in seconds)
  offset: offset of samples (delay in seconds)
  f: save/load file location (default: ./plan.bin)
  seed: deterministic random seed
  preview: optional PNG path for the faasgraph profile preview
  mode: 
       get_distribution: get and save the distribution
       generate: generate queries based on the distribution
'''


parameters = parameters_t()
plan = None
n_threads = None
console_debug = lambda *_, **__: None

def init(seed: int | None = None):
    global parameters
    import os
    if seed is not None:
        npseed = int(seed) % uint32_max
        randomseed = int(seed)
        np.random.seed(npseed)
        random.seed(randomseed)
        parameters.seeds = (npseed, randomseed)
        print(f'Using explicit random seed {seed}.')
        return
    if os.path.exists('seeds'):
        with open('seeds', 'r') as fp:
            try:
                seeds = tuple(int(s.strip()) for s in fp.read().split(' ') if s.strip())
                np.random.seed(seeds[0])
                random.seed(seeds[1])
                parameters.seeds = seeds
                print(f'Using random seeds {seeds}.')
                return
            except: pass
    random.seed(time.time() + 1)
    npseed = int(random.random() * time.perf_counter_ns()) % uint32_max
    np.random.seed(npseed)
    randomseed = int(np.random.random() * time.perf_counter_ns()) 
    random.seed(randomseed)
    parameters.seeds = (npseed, randomseed) # save the seeds for reproducibility

def controlled_shuffle(data, max_shift):
    n = len(data)
    if n < 3: return
    max_shift = int(min(max_shift, (n-1) // 2 - 1))
    for i in random.sample(range(1, (n-1)//2), max_shift):
        ss = data[:i]
        data[:i] = data[-i:]
        data[-i:] = ss
        
def _faasgraph_profile(bin_count: int, rng) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build the normalized multi-stage trend shown in FaaSGraph Figure 1."""
    x = (np.arange(bin_count, dtype=np.float64) + 0.5) / bin_count
    anchor_x = np.array([
        0.00, 0.04, 0.08, 0.12, 0.16, 0.19, 0.22, 0.26, 0.31,
        0.36, 0.41, 0.47, 0.50, 0.53, 0.56, 0.60, 0.64, 0.68,
        0.72, 0.77, 0.82, 0.88, 0.93, 0.97, 1.00,
    ])
    anchor_y = np.array([
        0.20, 0.29, 0.34, 0.34, 0.43, 0.53, 0.67, 0.60, 0.57,
        0.47, 0.42, 0.43, 0.56, 0.48, 0.57, 0.46, 0.35, 0.22,
        0.11, 0.055, 0.025, 0.012, 0.010, 0.025, 0.10,
    ])
    profile = np.interp(x, anchor_x, anchor_y)

    # Correlated noise keeps the curve natural without changing its day-scale shape.
    from scipy.ndimage import gaussian_filter1d

    noise = gaussian_filter1d(rng.normal(0.0, 1.0, bin_count), sigma=0.8)
    noise /= max(float(np.max(np.abs(noise))), 1e-12)
    profile += noise * (0.018 + 0.045 * profile)
    profile = np.clip(profile, 0.01, 0.74)

    spike_positions = (0.085, 0.17, 0.295)
    spike_heights = (0.95, 1.00, 0.82)
    spike_bins = tuple(
        min(bin_count - 1, max(0, int(round(position * (bin_count - 1)))))
        for position in spike_positions
    )
    for bin_index, height in zip(spike_bins, spike_heights):
        profile[bin_index] = height
    return profile, spike_bins


def _counts_from_profile(profile: np.ndarray, sample_count: int) -> np.ndarray:
    exact = profile / np.sum(profile) * sample_count
    counts = np.floor(exact).astype(np.int64)
    remainder = sample_count - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(exact - counts), kind='stable')
        counts[order[:remainder]] += 1
    return counts


def _save_faasgraph_preview(
    preview_path: str,
    timestamps: list[float],
    spike_bins: tuple[int, ...],
) -> None:
    from pathlib import Path
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(preview_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0.0, parameters.duration, parameters.granularity + 1)
    counts, _ = np.histogram(timestamps, bins=edges)
    bin_width = parameters.duration / parameters.granularity
    query_rate_qps = counts / bin_width
    peak_qps = max(float(np.max(query_rate_qps)), 1.0)
    x_minutes = (edges[:-1] + np.diff(edges) / 2.0) / 60.0

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)
    ax.plot(x_minutes, query_rate_qps, color='#1C8EAE', linewidth=1.0, label='query rate')
    ax.axhline(0.67 * peak_qps, color='#F06A6A', linewidth=1.8, label='normalized 0.67')
    ax.axhline(peak_qps, color='#888888', linestyle='--', linewidth=1.0)
    ax.axhline(0.01 * peak_qps, color='#888888', linestyle='--', linewidth=1.0)
    ax.scatter(
        [x_minutes[index] for index in spike_bins],
        [query_rate_qps[index] for index in spike_bins],
        marker='x',
        s=64,
        linewidth=1.2,
        color='#111111',
        label='spike',
        zorder=4,
    )
    ax.set_xlim(0.0, parameters.duration / 60.0)
    ax.set_ylim(0.0, peak_qps * 1.06)
    ax.set_xlabel('Elapsed time (minute)')
    ax.set_ylabel(f'Scheduled query rate (QPS, {bin_width:g}s window)')
    ax.set_title(f'55-minute FaaSGraph-like workload plan (peak {peak_qps:.1f} QPS)')
    ax.grid(axis='y', color='#E5E7EB', linewidth=0.7)
    ax.legend(loc='upper right', frameon=False, ncol=3)
    normalized_axis = ax.secondary_yaxis(
        'right',
        functions=(lambda qps: qps / peak_qps, lambda ratio: ratio * peak_qps),
    )
    normalized_axis.set_ylabel('Normalized query rate')
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f'Wrote preview to {output}')


def _gen_faasgraph_distribution(preview_path: str | None) -> None:
    global plan
    if parameters.n <= 0:
        raise ValueError('n must be positive')
    if parameters.duration <= 0:
        raise ValueError('duration must be positive')
    if parameters.granularity < 20:
        raise ValueError('faasgraph granularity must be at least 20')

    rng = np.random.default_rng(parameters.seeds[0])
    profile, spike_bins = _faasgraph_profile(parameters.granularity, rng)
    counts = _counts_from_profile(profile, parameters.n)
    bin_width = parameters.duration / parameters.granularity
    chunks = []
    for bin_index, count in enumerate(counts):
        if count <= 0:
            continue
        start = bin_index * bin_width
        chunks.append(rng.uniform(start, start + bin_width, int(count)))
    timestamps = np.sort(np.concatenate(chunks))
    timestamps[0] = 0.0
    timestamps[-1] = float(parameters.duration)
    plan = timestamps.tolist()

    with open(parameters.f, 'wb') as fp:
        pickle.dump(dump_t(parameters, plan), fp)
    print(
        f'Generated FaaSGraph profile: queries={len(plan)}, '
        f'duration={parameters.duration}s, bins={parameters.granularity}, '
        f'bin_width={bin_width:.3f}s'
    )
    if preview_path:
        _save_faasgraph_preview(preview_path, plan, spike_bins)


def gen_distribution(preview_path: str | None = None):
    global plan
    if parameters.outer == distribution_t.faasgraph:
        _gen_faasgraph_distribution(preview_path)
        return

    def get_normalized(a, b, n, r, postporcess, 
                       distribution = distribution_f[distribution_t.zipf]):
        if n == 0: return np.empty(0)
        data = distribution(a, b, n).astype(np.float64)
        data = postporcess(data, b)
        data *= r
        return data
    
    def postprocess_outer(data, b):
        data /= np.max(data)
        data = data *(1 - b) + b * (np.mean(data)+truncnorm_01rand(1, len(data))*.1) 
        return data / np.sum(data)
    weights = get_normalized(parameters.a1, 
                             parameters.b1, 
                             parameters.granularity, 
                             parameters.n, 
                             postprocess_outer, 
                             distribution = distribution_f[parameters.outer]
            )
    if parameters.s1 > 1:
        np.random.shuffle(weights)
    else:
        controlled_shuffle(weights, (len(weights) - 1 // 2 - 1) * parameters.s1)
    weights = np.round(weights).astype(np.int32)
    weights[-1] = max(parameters.n - np.sum(weights[:-1]), 0)
    console_debug(f'weights: \n{weights}')
    offsets = (np.array(range(0, parameters.granularity), dtype=np.float64)/parameters.granularity) * parameters.duration
    duration_per_segment = parameters.duration / parameters.granularity
    def postprocess(data, b):
        n = len(data)
        sample_size = int(n * (1 - b))
        if sample_size > 0:
            data = random.sample(list(data), sample_size)
        elif sample_size <= 0:
            data = []
        if (sample_size > n):
            print(f'Warning: sample size {sample_size} < n {n}')
        print(sample_size)
        if parameters.b3 > 0:
            window = int(parameters.b3)
            window = np.ones(window) / window
            data = np.convolve(data, window, mode = 'valid')
        if data:
            data = data / np.max(data)  
        data = np.append(data, (np.linspace(0, 1, n - sample_size) + truncnorm_01rand(1, n - sample_size)*parameters.b4))
        # data = np.sort(data)
        # if parameters.shfl > 0 and parameters.shfl <= 1:
        #     controlled_shuffle(data, (len(data) - 1 // 2 - 1) * parameters.shfl)
        # elif parameters.shfl > 1:
        # np.add.accumulate(data, out = data)
        return np.sort(data)
    
    plan = [k for w, off in zip(weights, offsets) for k in 
            get_normalized(parameters.a2, 
                           parameters.b2,
                           w, 
                           duration_per_segment, 
                           postprocess,
                           distribution = distribution_f[parameters.inner]
    ) + off]
    
    print(len(plan) - parameters.n)
    # console_debug(f'weights: \n{plan}')
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    # plan = savgol_filter(plan, window_length=10, polyorder=4)
    # plan = np.convolve(plan, 15, mode = 'full')
    # plan = gaussian_filter1d(plan, 10)
    with open(parameters.f, 'wb') as fp:
        pickle.dump(dump_t(parameters, plan), fp)

def generate_impl(workload : Iterable[Callable]): # submit workload functions as a list of functions
    time.sleep(parameters.offset) # sleep off the offset
    t0 = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        while len(plan) > 0:
            t = plan.pop(0)
            while t > time.perf_counter() - t0:
                if t < time.perf_counter() - t0 + .03: # python's sleep is not accurate enough
                    while t > time.perf_counter() - t0: # when delta_t < epsilon, busy wait
                        continue
                    else:
                        executor.submit(workload.pop(0), t) # submit the workload precisely at time t.
                        break
                else: 
                    time.sleep(t - (time.perf_counter() - t0) - .03)

def generate(workload : Iterable[Callable], plan_path: str):
    with open(plan_path, 'rb') as fp:
        dump : Optional[dump_t] = pickle.load(fp)
        global plan, parameters
        plan = dump.plan
        parameters = dump.parameters
        generate_impl(workload)
        
def main():
    import sys, copy
    argv = copy.deepcopy(sys.argv)
    global parameters, console_debug

    print_help = False
    explicit_seed = None
    preview_path = None
    argv.pop(0)

    def get_distribution(name: str):
        nonlocal argv
        d = 'zipf'
        match argv.pop(0).lower(): 
            case a if a.startswith('g') or a.startswith('n'):
                d = 'normal'
            case a if a.startswith('z'):
                d = 'zipf'
            case a if a.startswith('u'):
                d = 'uniform'
            case a if a.startswith('p'):
                d = 'poisson'
            case a if a.startswith('i'):
                d = 'inv_gaussian'
            case a if a.startswith('f'):
                d = 'faasgraph'
            case _:
                return
        if d == 'faasgraph' and name != 'outer':
            raise ValueError('faasgraph is only supported as the outer distribution')
        exec(f'parameters.{name} = distribution_t.{d}')

    while len(argv) > 0:
        arg = argv.pop(0)
        match arg.lower().strip():
            case '-o' | '--outer':
                get_distribution('outer')
            case '-i' | '--inner':
                get_distribution('inner')
            case '-a1' | '--a1' | '-mu' | '--mu':
                try: parameters.a1 = float(argv.pop(0))
                except Exception as e: print(e)
            case '-a2' | '--a2' | '-a' | '--a':
                try: parameters.a2 = float(argv.pop(0))
                except Exception as e: print(e)
            case '-b1' | '--b1':
                try: 
                    b1 = float(argv.pop(0))
                    if b1 >= 0 and b1 <= 1: 
                        parameters.b1 = b1
                except Exception as e: print(e)
            case '-b2' | '--b2':
                try: 
                    b2 = float(argv.pop(0))
                    if b2 >= 0 and b2 <= 1: 
                        parameters.b2 = b2
                except Exception as e: print(e)
            case '-b3' | '--b3':
                try: 
                    b3 = float(argv.pop(0))
                    if b3 > 0 : 
                        parameters.b3 = b3
                except Exception as e: print(e)
            case '-b4' | '--b4':
                try: 
                    b4 = float(argv.pop(0))
                    if b4 > 0 : 
                        parameters.b4 = b4
                except Exception as e: print(e)
            case '-n' | '--n':
                try: parameters.n = int(argv.pop(0))
                except Exception as e: print(e)
            case '-d' | '--duration':
                try: parameters.duration = int(argv.pop(0))
                except Exception as e: print(e)
            case '-o' | '--offset': 
                try: parameters.offset = float(argv.pop(0))
                except Exception as e: print(e)
            case '-g' | '--granularity':
                try: parameters.granularity = int(argv.pop(0))
                except Exception as e: print(e)
            case '-s1' :
                try: 
                    shfl = float(argv.pop(0))
                    if shfl >= 0 : 
                        parameters.s1 = shfl
                except Exception as e: print(e)
            case '-s' | '--shuffle':
                try: 
                    shfl = float(argv.pop(0))
                    if shfl >= 0: 
                        parameters.shfl = shfl
                except Exception as e: print(e)
            case '-m' | '--mode':
                try: 
                    match argv.pop(0).lower().strip()[:3]:
                        case 'get' | '0': 
                            parameters.mode = mode_t.get_distribution
                        case 'exec' | 'issue' | 'gen' | '1':
                            parameters.mode = mode_t.generate
                        case s:
                            raise ValueError(f'Invalid mode {s}')
                except Exception as e: print(e)
            case '-f' | '--f':
                try: parameters.f = argv.pop(0)
                except Exception as e: print(e)
            case '--seed':
                try: explicit_seed = int(argv.pop(0))
                except Exception as e: print(e)
            case '--preview':
                try: preview_path = argv.pop(0)
                except Exception as e: print(e)
            case '-h' | '--help':
                print_help = True
            case '-v' | '--verbose':
                console_debug = print
            case s:
                print(f'Invalid option: {s}')
                print_help = True
    if print_help:
        print(help_str)

    init(explicit_seed)
        
    console_log('Parameters:')
    for k, v in parameters.__dict__.items():
        console_log(f'    {k}: {v}')
    if parameters.mode == mode_t.get_distribution:
        gen_distribution(preview_path=preview_path)
    elif parameters.mode == mode_t.generate:
        with open(parameters.f, 'rb') as fp:
            dump : Optional[dump_t] = pickle.load(fp)
            global plan
            plan = dump.plan
            parameters = dump.parameters
            generate([lambda: console_log(f'executing {i}') for i in range(parameters.n)])
            
if __name__ == "__main__":
    main()
