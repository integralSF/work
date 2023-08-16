import os
import shutil
import random

source_folder = 'dataset/test_dataset_processed/train/partial'
target_folder = 'dataset/test_dataset_processed/val/partial'

classes = os.listdir(source_folder)
files_partial = []
for class_ in classes:
    files_path = os.listdir(os.path.join(source_folder, class_))
    
        
    for file_path in files_path:
        each_file_path_source_partial = os.path.join(source_folder, class_, file_path)

        files_partial.append(each_file_path_source_partial)
        
files_to_move = random.sample(files_partial, 8)


for file in files_to_move:
    source_path_partial = file
    target_path_partial = file.replace("train", "val")

    source_path_gt = source_path_partial.replace("partial", "gt")
    target_path_gt = target_path_partial.replace("partial", "gt")

    print(source_path_gt)
    shutil.move(source_path_partial, target_path_partial)
    shutil.move(source_path_gt, target_path_gt)
