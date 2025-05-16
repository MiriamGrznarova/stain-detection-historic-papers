import os
def count_lines_in_directory(directory):
    lines_foxing = 0
    lines_feces = 0
    lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] == '0':
                            lines_foxing += 1
                        else:
                            lines_feces += 1
                lines +=1
    return [lines_foxing, lines_feces, lines]

base_dir = fr''
train_labels_dir = os.path.join(base_dir, 'train/labels')
valid_labels_dir = os.path.join(base_dir, 'valid/labels')
test_labels_dir = os.path.join(base_dir, 'test/labels')
#
# foxing = count_lines_in_directory(train_labels_dir)[0] + count_lines_in_directory(valid_labels_dir)[0] + count_lines_in_directory(test_labels_dir)[0]
# feces = count_lines_in_directory(train_labels_dir)[1] + count_lines_in_directory(valid_labels_dir)[1] + count_lines_in_directory(test_labels_dir)[1]
# print(f'Foxing: {foxing}')
# print(f'Feces: {feces}')
# print(foxing + feces)

foxing_train = count_lines_in_directory(test_labels_dir)
print(foxing_train[0])
print(foxing_train[1])
print(foxing_train[2])