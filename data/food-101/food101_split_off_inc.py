import os
import yaml
from sklearn.model_selection import train_test_split

data_dir = os.path.dirname(os.path.realpath(__file__))

###
num_class = 101
off_label, inc_label = train_test_split(range(num_class), test_size = 0.2, random_state = 723)
to_save = {
    'off': off_label,
    'inc': inc_label
}

# save split
food101_split_path = os.path.join(data_dir, 'food101_test_split.yaml')
outfile_file = open(food101_split_path, 'w')
yaml.dump(to_save, outfile_file, sort_keys=False)
outfile_file.close()
print('Done splitting food101 data.')
print(f'yaml file save at {food101_split_path}')