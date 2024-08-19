import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, '.')
from isegm.utils.exp import load_config_file

def parse_args():
    parser = argparse.ArgumentParser()

    group_pkl_path = parser.add_mutually_exclusive_group(required=True)
    group_pkl_path.add_argument('--folder', type=str, default=None,
                                help='Path to folder with .pickle files.')
    group_pkl_path.add_argument('--files', nargs='+', default=None,
                                help='List of paths to .pickle files separated by space.')
    group_pkl_path.add_argument('--model-dirs', nargs='+', default=None,
                                help="List of paths to model directories with 'plots' folder "
                                     "containing .pickle files separated by space.")
    group_pkl_path.add_argument('--exp-models', nargs='+', default=None,
                                help='List of experiments paths suffixes (relative to cfg.EXPS_PATH/evaluation_logs). '
                                     'For each experiment, the checkpoint prefix must be specified '
                                     'by using the ":" delimiter at the end.')

    parser.add_argument('--mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                           'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        default=None, nargs='*', help='')
    # parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,COCO_MVal,SBD,LoveDA',
    #                     help='List of datasets for plotting the iou analysis'
    #                          'Datasets are separated by a comma. Possible choices: '
    #                          'GrabCut, Berkeley, DAVIS, COCO_MVal, SBD, LoveDA')
    parser.add_argument('--datasets', type=str, default='DAVIS,SBD,LoveDA',
                        help='List of datasets for plotting the iou analysis'
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, COCO_MVal, SBD, LoveDA')
    parser.add_argument('--config-path', type=str, default='../config.yml',
                        help='The path to the config file.')
    parser.add_argument('--n-clicks', type=int, default=-1,
                        help='Maximum number of clicks to plot.')
    parser.add_argument('--plots-path', type=str, default='',
                        help='The path to the evaluation logs. '
                             'Default path: cfg.EXPS_PATH/evaluation_logs/iou_analysis.')

    args = parser.parse_args()

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    args.datasets = args.datasets.split(',')
    if args.plots_path == '':
        args.plots_path = cfg.EXPS_PATH / 'evaluation_logs/iou_analysis'
    else:
        args.plots_path = Path(args.plots_path)
    print(args.plots_path)
    args.plots_path.mkdir(parents=True, exist_ok=True)

    return args, cfg


model_name_mapper = {'sbd_vitb_epoch_54_NoBRS': 'Ours-ViT-B (SBD)', 
                     'sbd_vitl_epoch_54_NoBRS': 'Ours-ViT-L (SBD)',
                     'sbd_vith_epoch_54_NoBRS': 'Ours-ViT-H (SBD)',
                     'cocolvis_vitb_epoch_54_NoBRS': 'Ours-ViT-B (C+L)',
                     'cocolvis_vitl_epoch_54_NoBRS': 'Ours-ViT-L (C+L)',
                     'cocolvis_vith_epoch_52_NoBRS': 'Ours-ViT-H (C+L)',
                     '052_NoBRS': 'Ours-ViT-H (C+L)',
                     'sbd_h18_itermask_NoBRS': 'RITM-HRNet18 (SBD)',
                     'coco_lvis_h32_itermask_NoBRS': 'RITM-HRNet32 (C+L)',
                     'cocolvis_segformer_b3_s2_FocalClick': 'FocalClick-SegF-B3 (C+L)',
                     'cocolvis_segformer_b0_s2_FocalClick': 'FocalClick-SegF-B0 (C+L)',
                     'sbd_cdnet_resnet34_CDNet': 'CDNet-ResNet-34 (SBD)',
                     'cocolvis_cdnet_resnet34_CDNet': 'CDNet-ResNet-34 (C+L)'
}

color_style_mapper = {'SimpleClick-ViT-B': ('#0000ff',   '-'),
                      '+GAT': ('#ff0000',   '-'),
                      '+GCN': ('#008000',   '-'),
                      'Ours-ViT-B (C+L)': ('#0080ff',   '-'),
                      'Ours-ViT-L (C+L)': ('#8000ff',   '-'),
                      'Ours-ViT-H (C+L)': ('#ff8000',   '-'),
                      'RITM-HRNet18 (SBD)': ('#000000',   ':'),
                      'RITM-HRNet32 (C+L)': ('#444444',   ':'),
                      'FocalClick-SegF-B0 (C+L)': ('#888888',   ':'),
                      'FocalClick-SegF-B3 (C+L)': ('#888888',   ':'),
                      'CDNet-ResNet-34 (SBD)': ('', ':'),
                      'CDNet-ResNet-34 (C+L)': ('', ':')
                     }

range_mapper = {'SBD': (73, 96, 3),
                 'DAVIS': (73, 97, 3),
                 'Pascal VOC': (66, 100, 3),
                 'COCO_MVal': (74, 97, 3),
                 'BraTS': (10, 100, 10),
                 'OAIZIB': (0,85, 10),
                 'ssTEM': (5, 100, 10),
                 'GrabCut': (88, 100, 2),
                 'Berkeley': (84, 100, 2),
                 'LoveDA': (36, 96, 3),
               }

dst_idxs = {
    'SBD': 0,
    'DAVIS': 1,
    'LoveDA': 2,
    'GrabCut': 3,
    'Berkeley': 4,
    'PascalVOC': 5,
    'COCO_MVal': 6,
}

def main():
    args, cfg = parse_args()
    simple_click_base = '/root/workspace/SimpleClick_orig/scripts/experiments/evaluation_logs/others/cocolvis_vit_base/plots'
    simple_click_large = '/root/workspace/SimpleClick_orig/scripts/experiments/evaluation_logs/others/cocolvis_vit_large/plots'
    simple_click_huge = '/root/workspace/SimpleClick_orig/scripts/experiments/evaluation_logs/others/cocolvis_vit_huge/plots'

    files_list_base = Path(simple_click_base).glob('*.pickle')
    files_list_base = [file for file in files_list_base if any(dataset in file.stem for dataset in args.datasets)]

    files_list = get_files_list(args, cfg)
    files_list_gat = [file for file in files_list if '580_' in str(file)]

    gats = set([x.stem[:3] for x in files_list])
    for gat in gats:
        files_list_gcn = [file for file in files_list if gat in str(file)]
        # Dict of dicts with mapping dataset_name -> model_name -> results
        top_algo = ''
        aggregated_plot_data = defaultdict(dict)
        for file_gcn, file_gat, file_base in zip(files_list_gcn, files_list_gat, files_list_base):
            with open(file_gcn, 'rb') as f:
                data_gcn = pickle.load(f)
            with open(file_gat, 'rb') as f:
                data_gat = pickle.load(f)
            with open(file_base, 'rb') as f:
                data_base = pickle.load(f)
            data_gcn['all_ious'] = [x[:] if args.n_clicks == -1 else x[:args.n_clicks] for x in data_gcn['all_ious']]
            data_gat['all_ious'] = [x[:] if args.n_clicks == -1 else x[:args.n_clicks] for x in data_gat['all_ious']]
            data_base['all_ious'] = [x[:] if args.n_clicks == -1 else x[:args.n_clicks] for x in data_base['all_ious']]

            # aggregated_plot_data[data['dataset_name']][data['model_name']] = np.array(data['all_ious']).mean(0)
            aggregated_plot_data[data_gcn['dataset_name']]['+GCN'] = np.array(data_gcn['all_ious']).mean(0)
            aggregated_plot_data[data_gat['dataset_name']]['+GAT'] = np.array(data_gat['all_ious']).mean(0)
            aggregated_plot_data[data_base['dataset_name']]['SimpleClick-ViT-B'] = np.array(data_base['all_ious']).mean(0)

        fig, ax = plt.subplots(1, len(args.datasets), figsize=(11*len(args.datasets), 6))

        for dataset_name, dataset_results in aggregated_plot_data.items():
            # plt.figure(figsize=(8, 7))
            dst_idx = dst_idxs[dataset_name]
            max_clicks = 0
            min_val, max_val = 100, -1
            for model_name, model_results in dataset_results.items():
                if args.n_clicks != -1:
                    model_results = model_results[:args.n_clicks]
                model_results = model_results  * 100

                min_val = min(min_val, min(model_results))
                max_val = max(max_val, max(model_results))

                n_clicks = len(model_results)
                max_clicks = max(max_clicks, n_clicks)

                # miou_str = ' '.join([f'mIoU@{click_id}={model_results[click_id-1]/100:.2%};'
                #                      for click_id in [1, 3, 5, 10, 20] if click_id <= len(model_results)])
                miou_str = ' '.join([f'@{click_id}={model_results[click_id - 1] / 100:.2%},'
                                     for click_id in [1, 5] if click_id <= len(model_results)])
                print(f'{model_name} on {dataset_name}:\n{miou_str}\n')

                label = model_name_mapper[model_name] if model_name in model_name_mapper else model_name

                color, style = None, None
                if label in color_style_mapper:
                    color, style = color_style_mapper[label]
                auc = np.sum([x/100 for x in model_results])
                # plt.plot(1 + np.arange(n_clicks), model_results, linewidth=2, label=f'{label}[{auc:02.2f}]', linestyle=style)
                ax[dst_idx].plot(1 + np.arange(n_clicks), model_results, linewidth=2, label=f'{label} [{miou_str} auc={auc:02.2f}]',
                                 color=color, linestyle=style)

            if dataset_name == 'PascalVOC':
                dataset_name = 'Pascal VOC'

            # plt.title(f'{dataset_name}', fontsize=22)
            # plt.grid()
            ax[dst_idx].set_title(f'SimpleClick on {dataset_name}', fontsize=22)
            ax[dst_idx].grid()
            # 获取图例句柄和标签
            # handles, labels = plt.gca().get_legend_handles_labels()
            handles, labels = ax[dst_idx].get_legend_handles_labels()
            # 根据标签排序
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0][-6:-1]), reverse=True))
            # 创建排序后的图例
            # plt.legend(handles, labels, loc=4, fontsize='x-large')
            ax[dst_idx].legend(handles, labels, loc=4, fontsize='x-large')

            min_val, max_val, step = range_mapper[dataset_name]
            # plt.yticks(np.arange(min_val, max_val, step=6), fontsize='xx-large')
            ax[dst_idx].set_yticks(np.arange(min_val, max_val, step=4))
            # plt.xticks(1 + np.arange(max_clicks, step=3), fontsize='xx-large')
            ax[dst_idx].set_xticks(1 + np.arange(max_clicks, step=4))
            # plt.xlabel('Number of Clicks', fontsize='xx-large')
            ax[dst_idx].set_xlabel('Number of Clicks', fontsize='xx-large')
            # plt.ylabel('mIoU score (%)', fontsize='xx-large')
            ax[dst_idx].set_ylabel('mIoU score (%)', fontsize='xx-large')
            # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax[dst_idx].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            # fig_path = get_target_file_path(args.plots_path, dataset_name)
            # plt.savefig(str(fig_path))
            dst_idx += 1
            top_algo = f'{top_algo}' + labels[0].split('[')[0]

        fig_path = get_target_file_path(args.plots_path, f'{gat}{top_algo}')
        plt.savefig(str(fig_path).replace('.png', '.pdf'))

def get_target_file_path(plots_path, dataset_name):
    previous_plots = sorted(plots_path.glob(f'{dataset_name}_*.png'))
    if len(previous_plots) == 0:
        index = 0
    else:
        index = int(previous_plots[-1].stem.split('_')[-1]) + 1

    return str(plots_path / f'{dataset_name}_{index:03d}.png')


def get_files_list(args, cfg):
    if args.folder is not None:
        files_list = Path(args.folder).glob('*.pickle')
    elif args.files is not None:
        files_list = [Path(file) for file in args.files]
    elif args.model_dirs is not None:
        files_list = []
        for folder in args.model_dirs:
            folder = Path(folder) / 'plots'
            files_list.extend(folder.glob('*.pickle'))
    elif args.exp_models is not None:
        files_list = []
        for rel_exp_path in args.exp_models:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')
            exp_path_prefix = cfg.EXPS_PATH / 'evaluation_logs' / rel_exp_path
            candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
            assert len(candidates) == 1, "Invalid experiment path."
            exp_path = candidates[0]
            files_list.extend(sorted((exp_path / 'plots').glob(checkpoint_prefix + '*.pickle')))

    if args.mode is not None:
        files_list = [file for file in files_list
                      if any(mode in file.stem for mode in args.mode)]
    files_list = [file for file in files_list
                  if any(dataset in file.stem for dataset in args.datasets)]

    return files_list


if __name__ == '__main__':
    main()