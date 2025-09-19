import os

def parse_chats(chat_text: str):
    """
    Breaks down the raw chat export into individual messages.

    Each message usually starts with either 'From:' or 'System Message'.
    Everything until the next marker is grouped into one message block.
    """
    lines = chat_text.splitlines()
    messages = []
    current = []

    for line in lines:
        if line.startswith("From:") or line.startswith("System Message"):
            if current:
                messages.append("\n".join(current).strip())
                current = []
        current.append(line)

    if current:
        messages.append("\n".join(current).strip())

    return messages


def chunk_chats(messages, group_size: int = 5, overlap: int = 1):
    """
    Groups chat messages into overlapping chunks.
    """
    chunks = []
    step = group_size - overlap

    for i in range(0, len(messages), step):
        block = messages[i:i + group_size]
        if len(block) < group_size:
            break
        chunks.append("\n\n".join(block))

    return chunks


def process_chat_file(file_path: str):
    """
    Loads a chat file from disk, parses it into messages,
    and returns a list of overlapping chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chat_text = f.read()

    messages = parse_chats(chat_text)
    return chunk_chats(messages)


if __name__ == "__main__":
    chat_file = "C:\\Users\\prask\\OneDrive\\Desktop\\Internship\\Chat Intelligence\\chats\\WhatsApp_917304407812@s.whatsapp.net\\chat-4.txt"

    chunks = process_chat_file(chat_file)

    for idx, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {idx}\n{chunk}\n")
