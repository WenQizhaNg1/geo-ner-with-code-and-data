import json

categories = ['沉积相', '储层类型', '地区', '年代地层', '盆地与构造', '岩石地层', '岩石类型']
data = {}

for category in categories:
    with open(f'{category}.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        items = content.split('\n')
        # 去除空字符串，假如文件末尾有多余的换行符
        items = [item for item in items if item]
        data[category] = items  # 注意这里，我直接使用列表存储

json_data = json.dumps(data, ensure_ascii=False, indent=4)

# 将JSON数据保存到文件
with open('combined_data.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

# 打印JSON数据
print(json_data)
