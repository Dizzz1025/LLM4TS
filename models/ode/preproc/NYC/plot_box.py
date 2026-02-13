import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_boxplot_analysis(dataset_base_path, dataset_name, output_dir='plots_box'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 准备数据文件路径
    files = {
        'Train': 'train_norm.csv',
        'Test_Sup': 'test_supervised_norm.csv',
        'Test_ZS': 'test_zeroshot_norm.csv'
    }

    combined_data = []

    # 2. 读取并整合数据
    for label, filename in files.items():
        path = os.path.join(dataset_base_path, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 只要 Value_0 和 Value_1
            df_melted = df[['Value_0', 'Value_1']].copy()
            df_melted['Source'] = label  # 标记来源：Train, Sup, ZS
            combined_data.append(df_melted)
    
    if not combined_data:
        print(f"Skip {dataset_name}: No files found.")
        return

    full_df = pd.concat(combined_data, axis=0)
    
    # 将数据从宽格式转为长格式，方便 Seaborn 画图
    # 转换后列名为: Source, Variable (Value_0/1), Value
    plot_df = full_df.melt(id_vars=['Source'], var_name='Variable', value_name='Normalized_Value')

    # 3. 绘图
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # 画箱线图
    ax = sns.boxplot(x='Variable', y='Normalized_Value', hue='Source', data=plot_df, palette='Set2')

    plt.title(f"Normalized Data Distribution (Boxplot): {dataset_name}", fontsize=16, fontweight='bold')
    plt.ylabel("Normalized Value (Z-Score Scale)")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3) # 0基准线

    # 4. 保存图片
    save_path = os.path.join(output_dir, f"{dataset_name}_norm_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Success: Boxplot saved to {save_path}")

# --- 运行示例 ---
for ds in ['bike', 'crime', 'taxi']:
    base = f'data/{ds}/' # 请根据实际路径修改
    save_boxplot_analysis(base, ds)