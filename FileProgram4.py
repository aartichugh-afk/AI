# Read data from Source.txt
with open("Sample.txt", "r") as source_file:
    data = source_file.readlines()  # Read all lines as a list of strings

# Append the read data to Sample.txt
with open("Source.txt", "a") as target_file:
    target_file.writelines(data)  # Write the lines to Sample.txt

# Reading and verifying the file contents
with open("Source.txt", "r") as target_file:
    content = target_file.read()

print(content)

