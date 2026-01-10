from synthetic_sim_zd import ChargedParticlesSim, SpringSim
import time
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='springs',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=0,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--ode', type=int, default=6000,
                    help='Length of trajectory.')
parser.add_argument('--num-test-box', type=int, default=1, # 往后延长多少个窗口
                    help='Length of test set trajectory.')
parser.add_argument('--num-test-extra', type=int, default=24,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_max', type=int, default=25,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_min', type=int, default=23,
                    help='Length of test set trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')

args = parser.parse_args()


def _worker_one_sim(idx_and_args):
    """单条 simulation 的 worker，用于 Pool 里."""
    idx, args = idx_and_args

    # 每个 worker 自己的随机种子，保证不同样本不一样
    np.random.seed(args.seed + idx)

    # 每个 worker 自己建一个 sim（不要用全局的那个）
    if args.simulation == 'springs':
        sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    elif args.simulation == 'charged':
        sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls)
    else:
        raise ValueError('Simulation {} not implemented'.format(args.simulation))

    # 图 + 轨迹
    static_graph = sim.generate_static_graph()
    loc, vel, T_samples, loc_lal = sim.sample_trajectory_static_graph_irregular_difflength_each(
        args, edges=static_graph, isTrain=False
    )

    return loc, vel, T_samples, loc_lal, static_graph


def generate_dataset_parallel(args, num_sims):
    """50 个进程并行生成 springs 数据集."""
    loc_all = []
    vel_all = []
    edges = []
    timestamps = []
    loc_label = []

    # 这里开 50 个 worker
    with Pool(processes=50) as pool:
        # 用 tqdm 包一下，能看到进度
        for loc, vel, T_samples, loc_lal, static_graph in tqdm(
                pool.imap_unordered(_worker_one_sim,
                                    ((i, args) for i in range(num_sims))),
                total=num_sims):
            loc_all.append(loc)
            vel_all.append(vel)
            timestamps.append(T_samples)
            loc_label.append(loc_lal)
            edges.append(static_graph)

    # 转成 numpy（object 类型）
    loc_all = np.asarray(loc_all, dtype=object)
    vel_all = np.asarray(vel_all, dtype=object)
    timestamps = np.asarray(timestamps, dtype=object)
    loc_label = np.asarray(loc_label, dtype=object)
    edges = np.stack(edges)

    return loc_all, vel_all, edges, timestamps, loc_label


if __name__ == "__main__":

    if args.simulation == 'springs':
        suffix = "_springs"
    elif args.simulation == 'charged':
        suffix = "_charged"
    else:
        raise ValueError('Simulation {} not implemented'.format(args.simulation))

    suffix += str(args.n_balls)
    np.random.seed(args.seed)

    print(suffix)

    if args.simulation == "springs":
        print("Generating {} test simulations".format(args.num_test))

        loc_test, vel_test, edges_test, timestamps_test, loc_label = \
            generate_dataset_parallel(args, args.num_test)

        np.save('loc_test' + suffix + '.npy', loc_test)
        np.save('vel_test' + suffix + '.npy', vel_test)
        np.save('edges_test' + suffix + '.npy', edges_test)
        np.save('times_test' + suffix + '.npy', timestamps_test)
        np.save('loc_label_test' + suffix + '.npy', loc_label)

    else:
        # 如果之后也想给 charged 做并行，可以再写一个 worker 版本
        raise NotImplementedError("并行版暂时只给 springs 写了，如果要 charged 我可以再帮你改。")
