import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_normalized_distribution_pdf(dataset_base_path, dataset_name, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 字体与学术风格配置
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["axes.labelweight"] = "bold"
    
    # 归一化文件路径
    paths = {
        'train_norm': 'train_norm.csv',
        'test_sup_norm': 'test_supervised_norm.csv',
        'test_zs_norm': 'test_zeroshot_norm.csv'
    }

    data = {}
    for key, filename in paths.items():
        p = os.path.join(dataset_base_path, filename)
        if os.path.exists(p):
            data[key] = pd.read_csv(p)
    
    if not data:
        print(f"Skipping {dataset_name}: No data found in {dataset_base_path}")
        return

    # 2. 创建画布 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # 学术配色 (高对比度)
    c_train, c_sup, c_zs = '#2c7bb6', '#d7191c', '#800080'
    subtitles = ["(a) Inflow Distribution", "(b) Outflow Distribution"]

    for i, col in enumerate(['Value_0', 'Value_1']):
        ax = axes[i]
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        
        # 绘制 KDE 曲线
        if 'train_norm' in data:
            sns.kdeplot(data['train_norm'][col], ax=ax, label='Train', fill=True, color=c_train, alpha=0.1, linewidth=1.5)
        if 'test_sup_norm' in data:
            sns.kdeplot(data['test_sup_norm'][col], ax=ax, label='Test Supervised', color=c_sup, linewidth=2)
        if 'test_zs_norm' in data:
            sns.kdeplot(data['test_zs_norm'][col], ax=ax, label='Test Zero-shot', color=c_zs, linestyle='--', linewidth=2)

        # 3. 绘制 +5 sigma 虚线
        sigma_limit = 5
        ax.axvline(x=sigma_limit, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # 标注文字
        y_limits = ax.get_ylim()
        ax.text(sigma_limit + 0.2, y_limits[1] * 0.7, r'$5\sigma$', color='black', fontsize=12, fontweight='bold')

        # 细节美化
        ax.set_xlabel("Normalized Value ($\sigma$)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{dataset_name.capitalize()} - {subtitles[i]}", fontsize=13, pad=12)
        ax.legend(fontsize=9, frameon=True, loc='upper right')
        
        # 动态调整 X 轴：确保至少能看到 6 sigma
        current_xlim = ax.get_xlim()
        ax.set_xlim(left=-2, right=max(current_xlim[1], 6))

    plt.tight_layout()
    
    # 保存为 PDF
    save_path = os.path.join(output_dir, f"{dataset_name}_norm_dist.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Successfully saved plot for {dataset_name} to {save_path}")
    plt.close()

# 执行：遍历三个数据集
datasets = ['bike', 'crime', 'taxi']
for ds in datasets:
    base_path = f'data/{ds}/' 
    save_normalized_distribution_pdf(base_path, ds)