import json
import os

def generate_training_json(output_path, num_samples=100000):
    """
    生成匹配 train_st_new.py 的 JSON 数据文件。
    
    Args:
        output_path: 输出 json 的路径 (例如 'data/train_data.json')
        num_samples: 样本数量，应与你的 .pt 文件中的样本数一致
    """
    data_list = []
    
    # 这里的 Prompt 可以根据你的任务需求进行修改
    # 注意：模型现在会自动处理后 60 步的预测，Prompt 只需要描述任务即可
    prompt_template = (
        "Task: Predict the future state at the query slots. "
        "QuerySlots:"
    )

    for i in range(num_samples):
        sample = {
            "id": i,  # 这个 ID 必须能对应到 st_features.pt 中的 key
            "conversations": [
                {
                    "from": "human",
                    "value": prompt_template
                }
            ]
        }
        data_list.append(sample)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"成功生成 JSON 文件: {output_path}")
    print(f"总计样本数: {len(data_list)}")

if __name__ == "__main__":
    # 修改为你希望保存的路径
    generate_training_json("generated_file/test/Sim_prompts.json", num_samples=25000)