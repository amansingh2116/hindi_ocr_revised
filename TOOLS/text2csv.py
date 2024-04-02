import os
import csv

def write_image_paths_to_text_file(folder_path, text_file_path):
    # Open text file in write mode
    with open(text_file_path, 'w') as textfile:
        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image
            if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Write file path to text file
                textfile.write(os.path.join(folder_path, filename) + '\n')



def create_csv_from_text(text_file, csv_file):
    # Open the text file
    with open(text_file, 'r') as file:
        lines = file.readlines()  # Read all lines from the text file

    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write each line as a row in the CSV file
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]  # Remove first and last character if they are space and double quotes
            writer.writerow([line])  # Write to CSV



# Example usage
folder_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\images_proj'
text_file_path = r'C:\Users\amans\OneDrive\Documents\GitHub\stat_sem2_project\TOOLS\paths.txt'
csv_file_path = "TOOLS\\path.csv"

write_image_paths_to_text_file(folder_path, text_file_path)
create_csv_from_text(text_file_path, csv_file_path)