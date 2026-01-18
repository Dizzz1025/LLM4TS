import json
import os

def generate_json(output_path, mode="train", system_type=None, num_samples_per_sys=100000):
    """
    生成匹配 STLlama 训练/测试需求的 JSON 文件。
    
    Args:
        output_path: 输出文件路径
        mode: "train" 或 "test"
        system_type: 仅在 mode="test" 时有效，可选 "springs" 或 "charged"
        num_samples_per_sys: 每个系统类别的样本数
    """
    
    # 定义 Prompt 模板
    prompts = {
        "springs": (
            "Task: Predict future 2D positions (x, y) of particles at the query slots. "
            "System: 2D spring-coupled particle system. "
            "QuerySlots:"
        ),
        "charged": (
            "Task: Predict future 2D positions (x, y) of particles at the query slots. "
            "System: 2D interacting particle system with pairwise forces. "
            "QuerySlots:"
        )
    }

    data_list = []

    if mode == "train":
        print(f"正在生成训练集：前 {num_samples_per_sys} 条为 Springs，后 {num_samples_per_sys} 条为 Charged...")
        # 前 100,000 条: Springs
        for i in range(num_samples_per_sys):
            data_list.append({
                "id": i,
                "conversations": [{"from": "human", "value": prompts["springs"]}]
            })
        # 后 100,000 条: Charged
        for i in range(num_samples_per_sys, 2 * num_samples_per_sys):
            data_list.append({
                "id": i,
                "conversations": [{"from": "human", "value": prompts["charged"]}]
            })

    elif mode == "test":
        if system_type not in prompts:
            raise ValueError("测试模式下必须指定 system_type 为 'springs' 或 'charged'")
        
        print(f"正在生成测试集：系统类型为 {system_type}，样本数 {num_samples_per_sys}...")
        for i in range(num_samples_per_sys):
            data_list.append({
                "id": i,
                "conversations": [{"from": "human", "value": prompts[system_type]}]
            })

    # 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"成功导出至: {output_path} | 总样本数: {len(data_list)}")

if __name__ == "__main__":
    # --- 1. 生成训练集 (混合模式) ---
    generate_json("generated_file/train/Sim_train_prompts.json", mode="train", num_samples_per_sys=100000)

    # --- 2. 生成测试集 (单一模式，按需修改) ---
    # generate_json("generated_file/test/Springs_test_prompts.json", mode="test", system_type="springs", num_samples_per_sys=25000)
    # generate_json("generated_file/test/Charged_test_prompts.json", mode="test", system_type="charged", num_samples_per_sys=25000)