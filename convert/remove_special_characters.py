import os

def sanitize_name(name):
    # Replace colon with underscore
    return name.replace(':', '_')

def rename_special_characters(root_dir):
    for current_root, dirs, files in os.walk(root_dir, topdown=False):
        # Rename files
        for file_name in files:
            new_name = sanitize_name(file_name)
            if new_name != file_name:
                old_path = os.path.join(current_root, file_name)
                new_path = os.path.join(current_root, new_name)
                print(f"Renaming file: {old_path} → {new_path}")
                os.rename(old_path, new_path)

        # Rename directories
        for dir_name in dirs:
            new_name = sanitize_name(dir_name)
            if new_name != dir_name:
                old_path = os.path.join(current_root, dir_name)
                new_path = os.path.join(current_root, new_name)
                print(f"Renaming folder: {old_path} → {new_path}")
                os.rename(old_path, new_path)

if __name__ == "__main__":
    # Change this to the folder you want to clean up
    target_directory = "/home/pi0/multi-modal/droid-multi-modal/data/success/2026-01-06-infer"
    
    if os.path.isdir(target_directory):
        rename_special_characters(target_directory)
    else:
        print(f"Error: Directory '{target_directory}' does not exist.")
