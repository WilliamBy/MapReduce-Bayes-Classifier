# 8:2切分数据集合
import os
import shutil
import random

root_path = 'NBCorpus/Country/'  # 替换为你的文件夹路径
out_path = 'dataset/'
nations = ['BRAZ', 'AUSTR', 'CANA']
src_dirs = [root_path + nation for nation in nations]
train_dirs = [out_path + 'train/' + nation for nation in nations]
test_dirs = [out_path + 'test/' + nation for nation in nations]
train_ratio = 0.8  # 训练集所占比例，这里是80%

for d in train_dirs:
    os.makedirs(d, exist_ok=True)
for d in test_dirs:
    os.makedirs(d, exist_ok=True)

for i in range(len(nations)):
    data_path = src_dirs[i]
    file_list = os.listdir(data_path)
    random.shuffle(file_list)
    split_index = int(len(file_list) * train_ratio)
    train_files = file_list[:split_index]
    test_files = file_list[split_index:]

    for file_name in train_files:
        src_path = os.path.join(src_dirs[i], file_name)
        dst_path = os.path.join(train_dirs[i], file_name)
        shutil.copy(src_path, dst_path)

    for file_name in test_files:
        src_path = os.path.join(src_dirs[i], file_name)
        dst_path = os.path.join(test_dirs[i], file_name)
        shutil.copy(src_path, dst_path)

print("Partition finished!")
