from config import CONVERSATION_MEMORY_SIZE, FILTERABLE_METADATA_FIELDS
from llm.ollama_client import ask_ollama, check_ollama_health, rewrite_query
from memory import ConversationMemory
from rag_engine.citation import append_source_list
from rag_engine.formatter import format_context
from rag_engine.indexer import build_index
from rag_engine.knowledge_manager import run_knowledge_manager
from rag_engine.retriever import evaluate_retrieval_quality, retrieve_docs


def format_main_help() -> str:
    return """\
\n=== 主選單使用指南 ===
1：只處理新增、修改或刪除的文件，適合平常更新知識庫。
2：清除舊索引後重新處理全部文件；更換模型或切塊規則後使用。
3：進入常駐問答模式，根據知識庫內容持續提問。
4：查看知識庫統計、列出文件或依檔名／路徑搜尋已索引文件。
help 或 ?：再次顯示本指南。
q：離開程式。"""


def format_qa_help() -> str:
    fields = ", ".join(sorted(FILTERABLE_METADATA_FIELDS))
    return f"""\
\n=== 問答模式使用指南 ===
直接輸入自然語言問題即可查詢知識庫。
help 或 ?：顯示本指南。
back：回到主選單。
rebuild：增量更新索引。
full_rebuild：完整重建索引。
memory：查看最近對話。
clear_memory：清空對話記憶。
filter show：查看目前篩選條件。
filter <欄位> <值>：新增或更新 Metadata Filter。
filter clear：清除全部篩選條件。
q：離開程式。

可篩選欄位：{fields}
範例：
  filter week Week03
  filter file_type .py
  filter root_source knowledge_base"""


def main() -> None:
    health = check_ollama_health()
    print(f"\nOllama 狀態：{health['message']}")

    while True:
        print("\n=== NKUST Local RAG Agent v0.9 ===")
        print("1. 增量更新索引")
        print("2. 完整重建索引")
        print("3. 進入常駐問答模式")
        print("4. Knowledge Manager")
        print("help. 使用指南")
        print("q. 離開")

        mode = input("請選擇模式：").strip()

        if mode.lower() in {"help", "?"}:
            print(format_main_help())
            continue

        if mode == "1":
            if build_index(full_rebuild=False):
                print("\n索引增量更新完畢，已回到主選單。")
            continue

        elif mode == "2":
            if build_index(full_rebuild=True):
                print("\n索引完整重建完畢，已回到主選單。")
            continue

        elif mode == "3":
            memory = ConversationMemory(maxlen=CONVERSATION_MEMORY_SIZE)
            active_filter: dict[str, str] = {}

            print("\n已進入常駐問答模式。")
            print("直接輸入問題，或輸入 help 查看完整使用指南。")

            while True:
                question = input("\n請輸入問題：").strip()

                if question.lower() == "q":
                    print("已離開 Agent。")
                    return

                if question.lower() == "back":
                    print("已回到主選單。")
                    break

                if question.lower() in {"help", "?"}:
                    print(format_qa_help())
                    continue

                if question.lower() == "rebuild":
                    if build_index(full_rebuild=False):
                        print("\n索引增量更新完畢，回到問答模式。")
                    continue

                if question.lower() == "full_rebuild":
                    if build_index(full_rebuild=True):
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

                if question.lower().startswith("filter"):
                    parts = question.split(maxsplit=2)
                    if len(parts) == 2 and parts[1].lower() == "show":
                        print(f"目前 Metadata Filter：{active_filter or '無'}")
                    elif len(parts) == 2 and parts[1].lower() == "clear":
                        active_filter.clear()
                        print("已清除 Metadata Filter。")
                    elif len(parts) == 3 and parts[1] in FILTERABLE_METADATA_FIELDS:
                        active_filter[parts[1]] = parts[2]
                        print(f"已設定篩選：{parts[1]} = {parts[2]}")
                    else:
                        print("可用欄位：" + ", ".join(sorted(FILTERABLE_METADATA_FIELDS)))
                    continue

                try:
                    rewritten_question = rewrite_query(question, memory.to_list())
                    if rewritten_question != question:
                        print(f"查詢改寫：{rewritten_question}")
                    results = retrieve_docs(
                        rewritten_question,
                        metadata_filter=active_filter,
                    )
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

        elif mode == "4":
            run_knowledge_manager()

        elif mode.lower() == "q":
            print("已離開 Agent。")
            break

        else:
            print("無效模式，請重新選擇。")


if __name__ == "__main__":
    main()
