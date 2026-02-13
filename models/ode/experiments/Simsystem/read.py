import numpy as np

# 文件路径
file_path = "test_p_vec_springs.npy"

try:
    # 加载数据，allow_pickle=True 是为了防止保存时使用了 object 格式
    data = np.load(file_path, allow_pickle=True)
    
    print(f"数据加载成功！")
    print(f"数组形状 (Shape): {data.shape}")
    
    # 针对你之前的讨论，验证三个维度的含义
    if len(data.shape) == 3:
        steps, samples, dims = data.shape
        print(f" - 第一维 (时间步): {steps}")
        print(f" - 第二维 (样本数): {samples}")
        print(f" - 第三维 (特征维): {dims}")
        
except FileNotFoundError:
    print(f"错误：未找到文件 '{file_path}'，请检查路径是否正确。")
except Exception as e:
    print(f"读取时发生错误: {e}")