# NKUST Local AI

> A local AI platform built on Ollama, ChromaDB and BGE-M3, designed for personal knowledge management, code understanding and future AI Agent development.

---

## 📖 專案介紹

NKUST Local AI 是一套完全本地部署（Local AI）的 AI 平台。

本專案以 **RAG（Retrieval-Augmented Generation，檢索增強生成）** 為核心，整合：

- Ollama
- ChromaDB
- BAAI/bge-m3 Embedding
- BAAI/bge-reranker-v2-m3
- Hybrid Search
- Incremental Index

建立一套可完全離線運作的知識管理系統。

目前主要目標為：

- 本地知識庫問答
- 程式碼理解
- 文件檢索
- NotebookLM 類型應用

未來將逐步擴充：

- GPT Mode
- AI Agent
- Web UI
- Tool Calling
- Long-term Memory

---

# ✨ 目前功能

目前已完成：

- ✅ 本地 RAG 問答
- ✅ Incremental Index（增量索引）
- ✅ Full Rebuild（完整重建索引）
- ✅ ChromaDB 向量資料庫
- ✅ BGE-M3 Embedding
- ✅ BGE-Reranker-v2-M3
- ✅ Hybrid Search
- ✅ Conversation Memory
- ✅ Structure-aware Chunking
- ✅ Metadata
- ✅ 多格式文件讀取
- ✅ Ollama 本地模型推論

---

# 📂 專案架構

```
AI_Server
│
├── agent
│   ├── main.py
│   ├── rag_engine
│   ├── llm
│   ├── config.py
│   ├── memory.py
│   └── requirements.txt
│
├── knowledge_base
│
└── docs
```

---

# ⚙️ 系統架構

```
Knowledge Base
        │
        ▼
File Reader
        │
        ▼
Chunking
        │
        ▼
Embedding
(BGE-M3)
        │
        ▼
ChromaDB
        │
        ▼
Hybrid Retrieval
        │
        ▼
Reranker
        │
        ▼
Context Builder
        │
        ▼
Ollama
(Qwen2.5)
        │
        ▼
Answer
```

---

# 📚 支援文件格式

目前支援：

- Markdown (.md)
- Text (.txt)
- PDF (.pdf)
- DOCX (.docx)
- Python (.py)
- C (.c)
- C++ (.cpp)
- Header (.h/.hpp)
- JSON (.json)
- YAML (.yaml/.yml)
- Jupyter Notebook (.ipynb)

---

# 🚀 安裝方式

## 1. Clone Repository

```bash
git clone https://github.com/yourname/NKUST_Local_AI.git
```

---

## 2. 建立虛擬環境

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

---

## 3. 安裝套件

```bash
pip install -r requirements.txt
```

---

## 4. 安裝 Ollama

https://ollama.com/

下載後確認：

```bash
ollama list
```

---

## 5. 下載模型

例如：

```bash
ollama pull qwen2.5:14b
```

---

## 6. 執行

```bash
python main.py
```

---

# 📁 Knowledge Base

將欲建立索引的文件放入：

```
knowledge_base/
```

完成後執行：

```
1. 增量更新索引
```

或

```
2. 完整重建索引
```

---

# 💬 CLI 使用方式

主選單：

```
1. 增量更新索引
2. 完整重建索引
3. 進入常駐問答模式
q. 離開
```

---

問答模式：

```
back
```

回主選單

```
rebuild
```

增量更新索引

```
full_rebuild
```

完整重建索引

```
memory
```

查看最近對話

```
clear_memory
```

清除最近對話

```
q
```

離開

---

# 🔍 RAG 流程

```
Documents
      │
      ▼
Reader
      │
      ▼
Chunking
      │
      ▼
Embedding
      │
      ▼
ChromaDB
      │
      ▼
Hybrid Search
      │
      ▼
Reranker
      │
      ▼
Prompt
      │
      ▼
Ollama
```

---

# 🗺️ Development Roadmap

## Phase 1

- [x] RAG Engine
- [x] Hybrid Search
- [x] Incremental Index
- [x] Conversation Memory

## Phase 2

- [ ] BM25 Hybrid Search
- [ ] Query Expansion
- [ ] Citation System
- [ ] Confidence Score
- [ ] Knowledge Manager

## Phase 3

- [ ] Notebook Mode
- [ ] GPT Mode
- [ ] Agent Mode

## Phase 4

- [ ] Web UI
- [ ] Multi-user
- [ ] Tool Calling
- [ ] Long-term Memory

---

# 📌 專案特色

- 完全本地執行
- 不依賴雲端 API
- 支援 GPU 推論
- 可自由擴充
- 適合作為 AI Agent 基礎平台

---

# 📄 License

MIT License