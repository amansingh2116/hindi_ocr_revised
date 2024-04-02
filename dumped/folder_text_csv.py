import os
import csv

def write_image_paths_to_csv(folder_path, csv_file_path):
    # Open CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csvfile:
        # Create CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image
            if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Write file path to CSV
                csv_writer.writerow([os.path.join(folder_path, filename)])

# Example usage
folder_path = r'C:\path\to\your\folder'
csv_file_path = r'C:\path\to\output\file.csv'

write_image_paths_to_csv(folder_path, csv_file_path)
