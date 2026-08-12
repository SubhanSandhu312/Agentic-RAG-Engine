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

# allowed_extensions = {
#     ".py",
#     ".md",
#     ".txt",
#     ".yml",
#     ".yaml",
#     ".json",
#     ".toml"
# }

# allowed_files = {
#     "Dockerfile",
#     "requirements.txt"
# }


def discover_files(p):
    files = []

    for item in p.iterdir():

        if item.is_dir():
            if item.name not in ignored_dirs:
                files.extend(discover_files(item))

        elif item.is_file():
            # if item.suffix in allowed_extensions or item.name in allowed_files:
            files.append(item)

    return files


p = Path("data")

files = discover_files(p)

for file in files:
    print(file)