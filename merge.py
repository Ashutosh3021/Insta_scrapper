
import os
import re

# Folder containing your TXT files
folder_path = "C:/Users/chulu/Downloads/Insta_scrapper/transcripts_2"  # Current folder

# Output file
output_file = os.path.join(folder_path, "complete_reel.txt")

# Match files like reel_0001.txt
pattern = re.compile(r"^reel_(\d+)\.txt$")

# Find and sort files by serial number
files = []

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        serial_number = int(match.group(1))
        files.append((serial_number, filename))

files.sort(key=lambda x: x[0])

# Merge all files
with open(output_file, "w", encoding="utf-8") as outfile:
    for index, (serial_number, filename) in enumerate(files):

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as infile:
            content = infile.read()

        if index > 0:
            outfile.write("\n\n")

        outfile.write(content)

        print(f"Merged: {filename}")

print("\nSuccessfully merged", len(files), "files!")
print("Output:", output_file)
