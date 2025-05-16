import os

def count_lines_in_directory(directory):
    lines_foxing_real = 0
    lines_feces_real = 0
    lines_foxing_syn = 0
    lines_feces_syn = 0
    real_files = 0
    syn_files = 0
    total_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                total_files += 1
                if 'jpg' in file:
                    is_real_data = True
                    real_files += 1
                else:
                    is_real_data = False
                    syn_files += 1
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] == '0':  # Ak je to riadok s "foxing"
                            if is_real_data:
                                lines_foxing_real += 1
                            else:
                                lines_foxing_syn += 1
                        else:  # Ak je to riadok s "feces"
                            if is_real_data:
                                lines_feces_real += 1
                            else:
                                lines_feces_syn += 1

    return {
        'real_files': real_files,
        'syn_files': syn_files,
        'lines_foxing_real': lines_foxing_real,
        'lines_feces_real': lines_feces_real,
        'lines_foxing_syn': lines_foxing_syn,
        'lines_feces_syn': lines_feces_syn,
        'total_files': total_files
    }


base_dir = fr''
train_labels_dir = os.path.join(base_dir, 'train/labels')
valid_labels_dir = os.path.join(base_dir, 'valid/labels')
test_labels_dir = os.path.join(base_dir, 'test/labels')

stats = count_lines_in_directory(train_labels_dir)

print(f"Real files: {stats['real_files']}")
print(f"Synthetic files: {stats['syn_files']}")
print(f"Foxing in real data: {stats['lines_foxing_real']} / Foxing in synthetic data: {stats['lines_foxing_syn']}")
print(f"Feces in real data: {stats['lines_feces_real']} / Feces in synthetic data: {stats['lines_feces_syn']}")

print(f"Total files processed: {stats['total_files']}")
