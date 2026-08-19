import json
from retriver import text_files

metadata_list = []
chunk_counter = 0

for file, content in text_files:
    for i in range(0, len(content), 400):
        chunk_text = content[i:i+400]
        
        metadata_entry = {
            "chunk_id": chunk_counter,
            "repository": "broken-pipeline-lab",
            "source_file": str(file),  # <--- Convert Path object to string here
            "text": chunk_text
        }
        
        metadata_list.append(metadata_entry)
        chunk_counter += 1

with open("chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2)