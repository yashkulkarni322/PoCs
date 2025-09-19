import pandas as pd
from io import StringIO

excel_path = "C:\\Users\\prask\\OneDrive\\Desktop\\CODING\\Chunking\\jugdish.xlsx"
xls = pd.ExcelFile(excel_path)
df = xls.parse(xls.sheet_names[0], header=None)

metadata_df = df.iloc[0:19].fillna("")

metadata_lines = metadata_df.astype(str).apply(
    lambda row: " | ".join(cell for cell in row if cell != ''), axis=1
)
metadata_text = "\n".join(metadata_lines)

header_row = df.iloc[19].dropna().tolist()
data_rows = df.iloc[20:].dropna(how='all').reset_index(drop=True)

data_rows.columns = header_row + list(data_rows.columns[len(header_row):])
data_rows = data_rows[header_row]

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)

chunks = []

# Chunk 0: metadata only (no headers)
chunks.append({
    "id": "chunk_0",
    "text": metadata_text,
    "metadata": {
        "type": "metadata_only",
        "source": "jugdish.xlsx",
        "chunk_index": 0
    }
})

# Chunks 1...n: Transactions in groups of 10 with headers (included once)
for i in range(0, len(data_rows), 5):
    chunk = data_rows.iloc[i:i+10]

    # Convert to text with single header
    content = chunk.to_string(index=False, header=True)

    chunks.append({
        "id": f"chunk_{len(chunks)}",
        "text": content,
        "metadata": {
            "type": "transactions",
            "source": "jugdish.xlsx",
            "chunk_index": len(chunks)
        }
    })

chunk_id_to_view = "chunk_3"  # We can change to any chunk like 'chunk_2', 'chunk_0'

for chunk in chunks:
    if chunk["id"] == chunk_id_to_view:
        print(f"\n Chunk ID: {chunk['id']}")
        print(" Metadata")
        for key, value in chunk["metadata"].items():
            print(f"{key}: {value}")
        
        print("\nText Block (Stored Content)")
        print(chunk["text"])
        break
