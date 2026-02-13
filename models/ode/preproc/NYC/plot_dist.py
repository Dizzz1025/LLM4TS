import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_final_academic_pdf(dataset_base_path, dataset_name, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 字体与学术风格配置
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["axes.labelweight"] = "bold"
    
    # 文件路径
    paths = {
        'train_raw': 'train_raw.csv', 'train_norm': 'train_norm.csv',
        'test_sup_raw': 'test_supervised_raw.csv', 'test_sup_norm': 'test_supervised_norm.csv',
        'test_zs_raw': 'test_zeroshot_raw.csv', 'test_zs_norm': 'test_zeroshot_norm.csv'
    }

    data = {}
    stats_list = []

    for key, filename in paths.items():
        p = os.path.join(dataset_base_path, filename)
        if os.path.exists(p):
            df = pd.read_csv(p)
            data[key] = df
            # 基础统计量计算
            for col in ['Value_0', 'Value_1']:
                v = df[col]
                stats_list.append({
                    'Dataset': dataset_name, 'Set Type': key,
                    'Feature': 'Inflow' if col=='Value_0' else 'Outflow',
                    'Mean': v.mean(), 'Std': v.std()
                })

    # 2. 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 经典高对比度配色
    c_train, c_sup, c_zs = 'blue', 'red', 'magenta'
    c_norm_train, c_norm_sup, c_norm_zs = 'green', 'orange', 'purple'

    # 子图编号
    subtitles = ["(a) inflow_raw", "(b) outflow_raw", "(c) inflow_norm", "(d) outflow_norm"]

    for i, col in enumerate(['Value_0', 'Value_1']):
        # --- Top Row: Raw Data ---
        ax_raw = axes[0, i]
        ax_raw.grid(True, linestyle='--', alpha=0.6) # 添加虚线方格
        if 'train_raw' in data:
            sns.kdeplot(data['train_raw'][col], ax=ax_raw, label='train', fill=True, color=c_train, alpha=0.15, linewidth=2)
        if 'test_sup_raw' in data:
            sns.kdeplot(data['test_sup_raw'][col], ax=ax_raw, label='test supervised', color=c_sup, linewidth=2.5)
        if 'test_zs_raw' in data:
            sns.kdeplot(data['test_zs_raw'][col], ax=ax_raw, label='test zero-shot', color=c_zs, linestyle='--', linewidth=2.5)
        
        ax_raw.set_xlabel("Value", fontsize=12, fontweight='bold')
        ax_raw.set_ylabel("Density", fontsize=12, fontweight='bold')
        # 将子图编号写在下面
        ax_raw.text(0.5, -0.3, subtitles[i], transform=ax_raw.transAxes, fontsize=14, fontweight='bold', ha='center')
        ax_raw.legend(fontsize=10, frameon=True)

        # --- Bottom Row: Normalized Data ---
        ax_norm = axes[1, i]
        ax_norm.grid(True, linestyle='--', alpha=0.6) # 添加虚线方格
        if 'train_norm' in data:
            sns.kdeplot(data['train_norm'][col], ax=ax_norm, label='train', fill=True, color=c_norm_train, alpha=0.15, linewidth=2)
        if 'test_sup_norm' in data:
            sns.kdeplot(data['test_sup_norm'][col], ax=ax_norm, label='test supervised', color=c_norm_sup, linewidth=2.5)
        if 'test_zs_norm' in data:
            sns.kdeplot(data['test_zs_norm'][col], ax=ax_norm, label='test zero-shot', color=c_norm_zs, linestyle='--', linewidth=2.5)
        
        ax_norm.set_xlabel("Value", fontsize=12, fontweight='bold')
        ax_norm.set_ylabel("Density", fontsize=12, fontweight='bold')
        # 将子图编号写在下面
        ax_norm.text(0.5, -0.3, subtitles[i+2], transform=ax_norm.transAxes, fontsize=14, fontweight='bold', ha='center')
        ax_norm.legend(fontsize=10, frameon=True)

    plt.tight_layout(pad=3.0) # 留出底部编号的空间
    
    # 保存为 PDF
    save_path = os.path.join(output_dir, f"{dataset_name}_final_academic.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    return pd.DataFrame(stats_list)

# 执行绘图
all_stats = []
# for ds in ['bike', 'crime', 'taxi']:
for ds in ['crime']:
    base_path = f'data/{ds}/' 
    df = save_final_academic_pdf(base_path, ds)
    all_stats.append(df)

# 打印统计结果
if all_stats:
    summary = pd.concat(all_stats)
    print("\n--- Summary Statistics Table ---")
    print(summary.to_string(index=False))