# NKUST Local AI

以 Ollama、ChromaDB、BGE-M3 與 BM25 建立的本地 RAG
（Retrieval-Augmented Generation，檢索增強生成）系統。

本專案目前專注於打造可追溯、可增量更新且完全在本機執行的
知識庫問答核心，適合用於研究筆記、技術文件、程式碼與論文檢索。

## 專案定位

目前版本是一套 CLI 型本地 RAG Engine，而不是通用聊天機器人或完整
AI Agent。回答會先檢索本機文件，再交由 Ollama 中的模型生成，並附上
可追溯的來源編號。

主要使用情境：

- 本地知識庫問答
- 論文與課程筆記檢索
- Python、C++ 專案程式碼理解
- ROS 2、VLM、Isaac Sim 等技術文件查詢
- NotebookLM 類型的本地文件助理

## 目前功能

- [x] 本地 RAG 問答
- [x] Ollama 本地模型推論
- [x] Ollama 服務與模型健康檢查
- [x] ChromaDB 持久化向量資料庫
- [x] BAAI/bge-m3 Embedding
- [x] BAAI/bge-reranker-v2-m3 Reranker
- [x] Vector Search + BM25 Hybrid Retrieval
- [x] 中英文混合查詢與中文 bigram
- [x] Structure-aware Chunking
- [x] 多格式文件讀取
- [x] Metadata 與來源資訊
- [x] 增量索引與完整重建
- [x] 索引掃描排除規則
- [x] 短期 Conversation Memory
- [x] `[來源 N]` 引用系統
- [x] 低品質檢索拒答機制
- [x] CPU／CUDA 自動選擇與執行環境顯示

## 系統流程

```text
本機文件
   │
   ▼
File Reader
   │
   ▼
Structure-aware Chunking
   │
   ▼
BGE-M3 Embedding
   │
   ▼
ChromaDB
   │
   ├── Vector Search
   └── BM25 Search
          │
          ▼
     候選合併與去重
          │
          ▼
 BGE Reranker + Keyword Bonus
          │
          ▼
   檢索品質判定
      │         │
      │通過     │不足
      ▼         ▼
 Citation    直接拒答
 Context
      │
      ▼
Ollama / Qwen2.5
      │
      ▼
含 [來源 N] 的回答
```

## 專案結構

```text
AI_Server/
├── agent/
│   ├── main.py                 # CLI 入口
│   ├── config.py               # 模型、路徑與檢索參數
│   ├── memory.py               # 工作階段短期記憶
│   ├── llm/
│   │   └── ollama_client.py    # Ollama 健康檢查與回答生成
│   ├── rag_engine/
│   │   ├── citation.py         # 引用編號與來源清單
│   │   ├── chunker.py          # 結構化切塊
│   │   ├── file_reader.py      # 文件讀取
│   │   ├── formatter.py        # Context 格式化
│   │   ├── indexer.py          # 增量與完整索引
│   │   ├── manifest.py         # 檔案雜湊與索引紀錄
│   │   ├── models.py           # Embedding、Reranker、ChromaDB
│   │   ├── path_filter.py      # 掃描排除規則
│   │   └── retriever.py        # Vector、BM25、Rerank 與拒答判定
│   └── test_rag_agent.py       # 基礎回歸測試
├── knowledge_base/             # 本機知識文件，不建議提交私人資料
├── requirements.txt
├── README.md
└── .gitignore
```

`agent/rag_agent.py` 與 `agent/old_vesion/` 為早期版本；目前正式入口為
`agent/main.py`。

## 支援文件格式

- Markdown：`.md`
- 純文字：`.txt`
- PDF：`.pdf`
- Word：`.docx`
- Python：`.py`
- C++／Header：`.cpp`、`.h`、`.hpp`
- JSON：`.json`
- YAML：`.yaml`、`.yml`
- Jupyter Notebook：`.ipynb`

目前 PDF 使用文字層擷取，不包含 OCR。掃描型 PDF 可能無法取得內容。

## 環境需求

- Windows 10／11
- Python 3.11（建議）
- Ollama
- 至少一個 Ollama 模型，例如 `qwen2.5:14b`
- NVIDIA GPU 為選配；沒有 CUDA 時會退回 CPU

目前開發環境使用 RTX 5070 Ti 16 GB，但程式並未綁定特定 GPU。

## 安裝

### 1. Clone 專案

```powershell
git clone https://github.com/rueijiunlin-crypto/my-ollama.git
cd my-ollama
```

### 2. 建立虛擬環境

目前專案慣例將虛擬環境放在 `agent/.venv`：

```powershell
python -m venv agent\.venv
```

啟用 PowerShell 虛擬環境：

```powershell
.\agent\.venv\Scripts\Activate.ps1
```

若 PowerShell 暫時阻擋腳本，可只對目前視窗放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\agent\.venv\Scripts\Activate.ps1
```

### 3. 安裝 Python 套件

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

主要套件：

- `requests`
- `chromadb`
- `sentence-transformers`
- `pypdf`
- `python-docx`
- `rank-bm25`

若要使用 NVIDIA GPU，請依照 PyTorch 官方安裝頁選擇與系統相容的
CUDA wheel，再確認：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### 4. 安裝並準備 Ollama

從 [Ollama 官方網站](https://ollama.com/) 安裝後下載模型：

```powershell
ollama pull qwen2.5:14b
ollama list
```

若 Ollama 沒有自動啟動：

```powershell
ollama serve
```

## 設定知識來源

編輯 `agent/config.py` 的 `DATA_DIRS`：

```python
DATA_DIRS = [
    Path(r"G:\NKUST_VLM\learning"),
    Path(r"G:\AI_Server\knowledge_base"),
]
```

請將路徑改成自己電腦上實際存在的資料夾。程式會遞迴掃描這些
資料夾內支援的文件。

同一檔案中的設定還包含：

```python
DB_DIR = r"G:\AI_Server\agent\chroma_db"
MANIFEST_PATH = Path(r"G:\AI_Server\agent\index_manifest.json")
MODEL = "qwen2.5:14b"
```

- `DB_DIR`：ChromaDB 索引儲存位置
- `MANIFEST_PATH`：增量索引追蹤檔
- `MODEL`：回答時使用的 Ollama 模型

### 排除規則

索引掃描會略過下列類型的目錄：

```text
.venv
venv
env
__pycache__
.git
.vscode
.idea
site-packages
chroma_db
node_modules
old_version
old vesion
```

`.gitignore` 只影響 Git；真正控制 RAG 掃描範圍的是 `DATA_DIRS` 與
`EXCLUDED_DIR_NAMES`。

## 執行

從專案根目錄執行：

```powershell
python agent\main.py
```

啟動時會顯示：

- PyTorch 與 CUDA 狀態
- Embedding 使用裝置
- Reranker 使用裝置
- Ollama 連線狀態
- 設定的 Ollama 模型是否已安裝

## CLI 操作

主選單：

```text
=== NKUST Local RAG Agent v0.6 ===
1. 增量更新索引
2. 完整重建索引
3. 進入常駐問答模式
q. 離開
```

### 第一次使用

第一次使用、切換 Embedding 模型或修改 Chunking 規則時，選擇：

```text
2. 完整重建索引
```

一般新增、修改或刪除文件後，選擇：

```text
1. 增量更新索引
```

### 問答模式指令

| 指令 | 功能 |
|---|---|
| `back` | 回到主選單 |
| `rebuild` | 增量更新索引 |
| `full_rebuild` | 完整重建索引 |
| `memory` | 顯示最近對話 |
| `clear_memory` | 清空目前工作階段記憶 |
| `q` | 離開程式 |

## 索引機制

### 增量更新

`index_manifest.json` 會記錄檔案的 SHA-256、修改時間、大小與 Chunk ID。

```text
新增檔案   → 建立新索引
修改檔案   → 更新該檔案的索引
刪除檔案   → 移除對應索引
未變更檔案 → 跳過
```

### 完整重建

完整重建會清除 ChromaDB Collection 內的舊 Chunk、Embedding 與
Metadata，再重新掃描所有 `DATA_DIRS`。

它不會刪除知識來源中的原始文件。

## 檢索與回答

### Hybrid Retrieval

每次提問會同時執行：

1. BGE-M3 Vector Search
2. BM25 關鍵字搜尋
3. 合併並去除重複候選 Chunk
4. BGE Reranker 重新排序
5. Keyword Score 作為小幅精確詞彙加分

BM25 分數為 `0` 不一定代表文件不相關，也可能表示該 Chunk 只由
Vector Search 找到，沒有進入 BM25 Top K。

### 引用來源

檢索結果會被編號：

```text
[來源 1]
[來源 2]
[來源 3]
```

模型被要求在重要事實後標註來源，例如：

```text
Token 是模型處理文字的基本單位。[來源 1]
```

程式也會在回答後附上固定來源清單，包含可取得的路徑、章節、頁碼或
函式／類別名稱。

### 拒答機制

系統會在呼叫 Ollama 前評估檢索結果。若 Reranker、BM25、Keyword
與 fallback 排序均未提供足夠證據，會直接回覆：

```text
目前知識庫中沒有找到足夠可靠的資料，請補充資料或重新描述問題。
```

門檻集中於 `agent/config.py`：

```python
MIN_RERANK_SCORE = 0.15
MIN_FALLBACK_FINAL_SCORE = 0.20
MIN_BM25_SCORE = 0.50
MIN_KEYWORD_SCORE = 0.15
```

這些是可調整的檢索門檻，不是回答正確率或機率。正式使用前應以自己
的真實問題集校準。

## 測試

在虛擬環境啟用後執行：

```powershell
python agent\test_rag_agent.py
```

測試涵蓋：

- Markdown 與 Python Chunking
- 中英文混合 Tokenizer
- BM25 搜尋與分數正規化
- 索引排除路徑
- Hybrid Retrieval 回傳欄位
- Reranker 與 Keyword 最終排序
- Citation 來源格式
- 檢索品質接受／拒絕判定
- Ollama 健康檢查

成功時會顯示：

```text
All tests passed.
```

## 知識庫與 Git

知識庫通常包含私人筆記、教材或受版權保護的文件，不建議直接提交到
公開 GitHub Repository。

建議 `.gitignore` 保留：

```gitignore
knowledge_base/*
!knowledge_base/.gitkeep
!knowledge_base/README.md
```

另外也不應提交：

- `.venv/`
- `chroma_db/`
- `index_manifest.json`
- Hugging Face 模型快取
- Ollama 模型
- `.env`

## 已知限制

- PDF 尚未支援 OCR 與複雜表格重建
- 對話記憶只存在目前執行期間，關閉程式後會消失
- BM25 Corpus 目前可能在每次查詢時重建，資料量大時會變慢
- 檢索結果尚未進行相似 Chunk 去重與來源多樣性控制
- 增量更新尚未提供完整交易式／原子式保護
- 尚未支援 Query Rewrite、Metadata Filter 與 Parent-Child Retrieval
- 尚無 Web UI、多使用者、權限管理與長期記憶

## Roadmap

### Phase 1：RAG 核心

- [x] 多格式文件讀取
- [x] Structure-aware Chunking
- [x] BGE-M3 Embedding
- [x] ChromaDB
- [x] 增量索引
- [x] Vector + BM25 Hybrid Retrieval
- [x] BGE Reranker
- [x] 索引排除規則
- [x] Conversation Memory

### Phase 2：可信度與可追溯性

- [x] Citation System
- [x] 檢索品質拒答機制
- [x] Ollama Health Check
- [ ] 以真實問題集校準拒答門檻
- [ ] 回答引用完整性驗證

### Phase 3：檢索品質

- [ ] Query Rewrite
- [ ] Metadata Filter
- [ ] Context 去重與來源多樣性
- [ ] Parent-Child Retrieval
- [ ] 中英文術語別名／查詢擴展

### Phase 4：效能與安全

- [ ] BM25 Cache
- [ ] Batch Embedding
- [ ] 原子式增量索引更新
- [ ] 完整重建清除失敗時中止
- [ ] PDF 擷取品質報告
- [ ] Knowledge Manager

### Phase 5：應用介面

- [ ] Notebook Mode
- [ ] GPT Mode
- [ ] Agent Mode
- [ ] Web UI
- [ ] Tool Calling
- [ ] Long-term Memory
- [ ] Multi-user 與權限管理

## License

目前 README 宣告以 MIT License 為預期授權方式；正式公開前請在專案
根目錄加入完整的 `LICENSE` 檔案。
