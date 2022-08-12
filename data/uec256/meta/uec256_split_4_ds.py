import os
from sklearn.model_selection import train_test_split
import pandas as pd


data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
mdata_dir = os.path.join(data_dir, 'meta')
img_dir = os.path.join(data_dir, 'UECFOOD256')

# read category.txt as tsv
class_info_path = os.path.join(mdata_dir, 'category.txt')
class_info = pd.read_csv(class_info_path, sep = '\t')

num_class = max(class_info['id'])




print('Splitting class for offline and incremental learning...')
full_class_id = class_info['id'].tolist()
off_label, inc_label = train_test_split(range(1, num_class+1), test_size = 0.2, random_state = 812)
num_off_class = len(off_label)
num_inc_class = len(inc_label)
print(f'Number of offline class = {num_off_class}, Number of incremental class = {num_inc_class}')

# create label map
def id_to_class(row, off_label, inc_label):
    id = row['id']
    if id in off_label:
        manual_class = off_label.index(id)
    elif id in inc_label:
        manual_class = inc_label.index(id) + len(off_label)
    else:
        print("Can't find ID in label list!")
        manual_class = False

    return manual_class

class_info['manual_class'] = class_info.apply(lambda row: id_to_class(row, off_label, inc_label), axis =1)
class_info = class_info[['manual_class', 'id', 'name']].sort_values('manual_class')

print('Splitting train test set for offline learning...')
col_names = ['manual_class', 'img', 'x1', 'y1', 'x2', 'y2']
off_train_df = pd.DataFrame(columns = col_names)
off_test_df = pd.DataFrame(columns = col_names)

for i in range(num_off_class):
    class_id = off_label[i]
    class_dir = os.path.join(img_dir, str(class_id))

    # read bb_info.txt
    bb_name = 'bb_info.txt'
    bb_path = os.path.join(class_dir, bb_name)

    # obtain info of all images in current class
    class_img_data = pd.read_csv(bb_path, sep = ' ')
    class_img_data['manual_class'] = i
    class_img_data['actual_id'] = class_id

    #split train test and concat to full
    class_train_df, class_test_df = train_test_split(class_img_data, test_size = 0.2, random_state = i)
    off_train_df = pd.concat([off_train_df, class_train_df], ignore_index=True)
    off_test_df = pd.concat([off_test_df, class_test_df], ignore_index=True)

off_train_len = len(off_train_df)
off_test_len = len(off_test_df)
print(f'Train: {off_train_len} images, Test: {off_test_len} images.')


print('Splitting train test set for incremental learning...')
inc_train_df = pd.DataFrame(columns = col_names)
inc_test_df = pd.DataFrame(columns = col_names)

for i in range(num_inc_class):
    class_id = inc_label[i]
    class_dir = os.path.join(img_dir, str(class_id))

    # read bb_info.txt
    bb_name = 'bb_info.txt'
    bb_path = os.path.join(class_dir, bb_name)

    # obtain info of all images in current class
    class_img_data = pd.read_csv(bb_path, sep = ' ')
    class_img_data['manual_class'] = i + num_off_class
    class_img_data['actual_id'] = class_id

    #split train test and concat to full
    class_train_df, class_test_df = train_test_split(class_img_data, test_size = 0.2, random_state = i)
    inc_train_df = pd.concat([inc_train_df, class_train_df], ignore_index=True)
    inc_test_df = pd.concat([inc_test_df, class_test_df], ignore_index=True)

inc_train_len = len(inc_train_df)
inc_test_len = len(inc_test_df)
print(f'Train: {inc_train_len} images, Test: {inc_test_len} images.')


print('Saving 2*2 dataset list info and class list...')
off_train_path = os.path.join(mdata_dir, 'off_train.csv')
off_train_df.to_csv(off_train_path, header=True, index=False)

off_test_path = os.path.join(mdata_dir, 'off_test.csv')
off_test_df.to_csv(off_test_path, header=True, index=False)

inc_train_path = os.path.join(mdata_dir, 'inc_train.csv')
inc_train_df.to_csv(inc_train_path, header=True, index=False)

inc_test_path = os.path.join(mdata_dir, 'inc_test.csv')
inc_test_df.to_csv(inc_test_path, header=True, index=False)

class_info_path = os.path.join(mdata_dir, 'class_list.csv')
class_info.to_csv(class_info_path, header=True, index=False)

print(f'Saved all list info at {mdata_dir}')