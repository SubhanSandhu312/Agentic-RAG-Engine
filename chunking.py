from retriver import text_files

chunks = []

for file, content in text_files:
    for i in range(0, len(content), 400):
        chunk = content[i:i+400]
        chunks.append((file, chunk))


# print(chunks)
# print(len(text_files))
# print("Total number of chunks are ", len(chunks))

# print(type(chunks))
# print(type(chunks[0]))
# print(chunks[0])
# print(chunks[0])