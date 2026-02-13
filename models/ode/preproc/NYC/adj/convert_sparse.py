import numpy as np
import scipy.sparse as sp
import pickle
import os

def convert_npy_to_sparse(file_list):
    for npy_path in file_list:
        if not os.path.exists(npy_path):
            print(f"找不到文件: {npy_path}")
            continue
            
        # 1. 加载 npy 文件
        # 确保数据类型为 float32 以节省空间并匹配 PyTorch 默认精度
        adj = np.load(npy_path).astype(np.float32)
        
        # 2. 转换为 CSR (Compressed Sparse Row) 稀疏矩阵
        # 这种格式在进行矩阵乘法 A * X 时效率最高
        sparse_adj = sp.csr_matrix(adj)
        
        # 3. 构造保存路径 (例如 bike_80x80.npy -> bike_80x80_sparse.pkl)
        save_path = npy_path.replace('.npy', '_sparse.pkl')
        
        # 4. 序列化保存
        with open(save_path, 'wb') as f:
            pickle.dump(sparse_adj, f)
            
        print(f"转换成功: {npy_path} -> {save_path}")
        print(f"  - 原始形状: {adj.shape}")
        print(f"  - 非零元素个数: {sparse_adj.nnz}")

# 定义你的文件名列表
files = [
    'bike_80x80.npy',
    'bike_81x160.npy',  # 既然你说都是80x80，这里脚本也会按80x80处理
    'taxi_80x80.npy',
    'taxi_81x160.npy'
]

if __name__ == "__main__":
    convert_npy_to_sparse(files)