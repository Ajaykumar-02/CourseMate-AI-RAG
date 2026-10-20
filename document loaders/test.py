from pathlib import Path
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap=1
)


# test.py jis folder me hai, usi folder me 'notes.txt' dhoondhega
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "notes.txt"

loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

# print(docs[0].page_content)
# print(docs[0].metadata)

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()