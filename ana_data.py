def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def count_entities(lines):
    entity_count = {}
    current_entity = None

    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) != 2:
                continue  # 跳过格式不正确的行
            word, label = parts

            if label.startswith('B-'):
                if current_entity:
                    entity_count[current_entity] = entity_count.get(current_entity, 0) + 1
                current_entity = label[2:]  # Start a new entity
            elif label.startswith('I-') and current_entity:
                continue  # Continue the current entity
            else:
                if current_entity:
                    entity_count[current_entity] = entity_count.get(current_entity, 0) + 1
                current_entity = None  # No entity or O tag

    # Count the last entity if needed
    if current_entity:
        entity_count[current_entity] = entity_count.get(current_entity, 0) + 1

    return entity_count

def analyze_dataset(data_dir):
    datasets = ['train.txt', 'dev.txt', 'test.txt']
    total_counts = {}
    dataset_counts = {}

    for dataset in datasets:
        lines = read_file(f"{data_dir}/{dataset}")
        counts = count_entities(lines)

        for entity, count in counts.items():
            total_counts[entity] = total_counts.get(entity, 0) + count

        dataset_counts[dataset] = counts

    proportions = {dataset: {entity: count / total_counts[entity] for entity, count in dataset_counts[dataset].items()} for dataset in datasets}

    return dataset_counts, proportions

# 定义数据集目录路径
data_dir = 'data/geo2/ori_data'  # 修改为你的数据集路径

# 分析数据集
dataset_counts, proportions = analyze_dataset(data_dir)

# 打印结果
for dataset in dataset_counts:
    print(f"{dataset} entity counts:")
    for entity, count in dataset_counts[dataset].items():
        print(f"{entity}: {count}")

print("\nEntity proportions across datasets:")
for dataset in proportions:
    print(f"\n{dataset} entity proportions:")
    for entity, proportion in proportions[dataset].items():
        print(f"{entity}: {proportion:.2f}")
