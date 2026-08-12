from pathlib import Path

ignored_dirs = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build"
}




def discover_files(p):
    files = []

    for item in p.iterdir():

        if item.is_dir():
            if item.name not in ignored_dirs:
                files.extend(discover_files(item))

        elif item.is_file():
            files.append(item)

    return files

p = Path("data")


files_name = discover_files(p)

for file in files_name:
    print(file)

text_files = []
for file in files_name:
    with open(str(file), "r", encoding="utf-8") as f:
        content = f.read()
        text_files.append((file, content))
        # print(f"Content of {file}:")
        # print(content)
        # print("-" * 40)

# print(text_files)