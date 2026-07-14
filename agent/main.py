from config import CONVERSATION_MEMORY_SIZE
from llm.ollama_client import ask_ollama, check_ollama_health
from memory import ConversationMemory
from rag_engine.citation import append_source_list
from rag_engine.formatter import format_context
from rag_engine.indexer import build_index
from rag_engine.retriever import evaluate_retrieval_quality, retrieve_docs


def main() -> None:
    health = check_ollama_health()
    print(f"\nOllama 狀態：{health['message']}")

    while True:
        print("\n=== NKUST Local RAG Agent v0.6 ===")
        print("1. 增量更新索引")
        print("2. 完整重建索引")
        print("3. 進入常駐問答模式")
        print("q. 離開")

        mode = input("請選擇模式：").strip()

        if mode == "1":
            build_index(full_rebuild=False)
            print("\n索引增量更新完畢，已回到主選單。")
            continue

        elif mode == "2":
            build_index(full_rebuild=True)
            print("\n索引完整重建完畢，已回到主選單。")
            continue

        elif mode == "3":
            memory = ConversationMemory(maxlen=CONVERSATION_MEMORY_SIZE)

            print("\n已進入常駐問答模式。")
            print("輸入 back 可回到主選單。")
            print("輸入 rebuild 可增量更新索引。")
            print("輸入 full_rebuild 可完整重建索引。")
            print("輸入 memory 可查看最近對話。")
            print("輸入 clear_memory 可清空對話記憶。")
            print("輸入 q 可離開。\n")

            while True:
                question = input("\n請輸入問題：").strip()

                if question.lower() == "q":
                    print("已離開 Agent。")
                    return

                if question.lower() == "back":
                    print("已回到主選單。")
                    break

                if question.lower() == "rebuild":
                    build_index(full_rebuild=False)
                    print("\n索引增量更新完畢，回到問答模式。")
                    continue

                if question.lower() == "full_rebuild":
                    build_index(full_rebuild=True)
                    print("\n索引完整重建完畢，回到問答模式。")
                    continue

                if question.lower() == "memory":
                    memory.display()
                    continue

                if question.lower() == "clear_memory":
                    memory.clear()
                    print("已清空對話記憶。")
                    continue

                if not question:
                    continue

                try:
                    results = retrieve_docs(question)
                    quality = evaluate_retrieval_quality(results)
                    context = format_context(results)

                    if not quality["accepted"]:
                        print("\n=== 檢索內容 ===\n")
                        print(context)
                        print("\n=== Agent 回答 ===\n")
                        print(quality["reason"])
                        continue

                    health = check_ollama_health()
                    if not health["available"] or not health["model_available"]:
                        print(f"\n無法產生回答：{health['message']}")
                        continue

                    answer = ask_ollama(question, context, memory.to_list())
                    answer = append_source_list(answer, results)
                except Exception as exc:
                    print(f"問答時發生錯誤：{exc}")
                    continue

                memory.add(question, answer)

                print("\n=== 檢索內容 ===\n")
                print(context)

                print("\n=== Agent 回答 ===\n")
                print(answer)

        elif mode.lower() == "q":
            print("已離開 Agent。")
            break

        else:
            print("無效模式，請重新選擇。")


if __name__ == "__main__":
    main()
