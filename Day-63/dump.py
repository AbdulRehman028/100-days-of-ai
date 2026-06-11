# this script will create a markdown file named `codebase.md` that contains the contents of all files in the `client/src` directory, organized by file path. Each file's content is wrapped in a code block with the appropriate language syntax highlighting based on the file extension.

import os

src_dir = os.path.join(".", "client", "src")
output_file = "codebase.md"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, ".")
            ext = os.path.splitext(file)[1].replace(".", "")
            
            print(f"Adding: {relative_path}")
            outfile.write(f"\n\n# File: {relative_path}\n```{ext}\n")
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"// Error reading file: {str(e)}")
            outfile.write("\n```")

print("Finished! Your codebase.md is ready.")
