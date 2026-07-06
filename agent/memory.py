from collections import deque


class ConversationMemory:
    def __init__(self, maxlen: int):
        self._history = deque(maxlen=maxlen)

    def add(self, question: str, answer: str) -> None:
        self._history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    def clear(self) -> None:
        self._history.clear()

    def to_list(self) -> list[dict]:
        return list(self._history)

    def display(self) -> None:
        if not self._history:
            print("目前沒有對話記憶。")
            return

        print("\n=== 最近對話記憶 ===\n")
        for index, item in enumerate(self._history, start=1):
            print(f"{index}. Q: {item.get('question', '')}")
            print(f"   A: {item.get('answer', '')}\n")
