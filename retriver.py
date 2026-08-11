from pathlib import Path

p = Path("data")

for items in p.iterdir():
    print(items)
    if items.is_dir():
        for inner in items.iterdir():
            print(inner)
            if inner.is_dir():
                for innest in inner.iterdir():
                    # if innest.exists():
                    print(innest)