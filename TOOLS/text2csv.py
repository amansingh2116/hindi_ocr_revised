import csv

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
text_file_path = "TOOLS\\paths.txt"  # Path to the input text file
csv_file_path = "TOOLS\\path.csv"  # Path to the output CSV file
create_csv_from_text(text_file_path, csv_file_path)
