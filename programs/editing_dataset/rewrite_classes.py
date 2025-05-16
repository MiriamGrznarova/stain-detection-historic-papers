import os

def replace_1_with_5_in_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            if parts[0] == '0':
                parts[0] = '1'
            else:
                parts[0] = '0'
            file.write(' '.join(parts) + '\n')


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            replace_1_with_5_in_file(file_path)
            print(f"Processed file: {file_path}")
directory_path =fr''
process_directory(directory_path)
