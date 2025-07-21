<div align="center">
<img alt="COVER" src="./cover.png" width="300" height="300" />
  
  # AI 英語學習夥伴
  
  [**English**](./README.md) | [**繁體中文**](./README.zh-TW.md)
  
  這是一個整合多種深度學習模型的多模態「 AI 英語學習夥伴」聊天機器人 (LINE Bot)
  
  透過易於上手的 LINE 介面，以及文字、語音與圖像多種互動方式，幫助使用者輕鬆地進行情境式英語學習

</div>

-----

# 目錄

  - [主要功能](https://www.google.com/search?q=%23%E4%B8%BB%E8%A6%81%E5%8A%9F%E8%83%BD)
  - [如何使用](https://www.google.com/search?q=%23%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8)
  - [專案結構](https://www.google.com/search?q=%23%E5%B0%88%E6%A1%88%E7%B5%90%E6%A7%8B)
  - [版權宣告](https://www.google.com/search?q=%23%E7%89%88%E6%AC%8A%E5%AE%A3%E5%91%8A)

-----

# 主要功能

  - ✅ **多模態聊天機器人**：結合文字、語音和圖像輸入/輸出，創造沉浸式學習情境。
  - 🧠 **整合深度學習模型**：
      - **LLM (大型語言模型)**：處理對話生成。
      - **STT (語音轉文字)**：將使用者語音轉換為文字。
      - **TTS (文字轉語音)**：朗讀 AI 回應。
      - **VLM (視覺語言模型)**：處理基於圖像的問題或輸入。
  - 📚 **RAG (檢索增強生成)**：使用向量資料庫提高生成回應的準確性和資訊量。
  - 🔄 **排程推播通知**：保持學習者參與度並養成每日習慣。
  - 💬 **LINE Bot 介面**：無需下載應用程式，直接在 LINE 上聊天即可。
  - ⚙️ **一鍵環境設定**：執行 `install.py` 即可安裝所有所需依賴項。

-----

# 如何使用  

### 🌐 1. 設定 ngrok (將你的本地伺服器暴露到網際網路)  

LINE Bot 的 webhook 需要一個公開的 HTTPS URL。Ngrok 可以幫助你從本地機器建立一個安全的隧道：

請記下生成的 HTTPS URL，你將會把這個 URL 用作你的 webhook URL。

### 🔔 2. 註冊並設定你的 LINE Bot

前往 LINE Developers Console，建立一個 Provider，然後建立一個 Messaging API Channel。

找到並複製你的 `LINE_CHANNEL_ACCESS_TOKEN` 和 `LINE_CHANNEL_SECRET`。

啟用「使用 webhook」。

將你的 webhook URL 設定為你的 ngrok HTTPS URL。

透過 LINE Developers Console 中的 QR code 將你的 LINE Bot 加為好友。

### 🐍 3. 準備 Python 環境並安裝依賴項

> **Python 版本**：3.12 (推薦)
>  
> **初始設定時間**：約 10 分鐘 (透過 `install.py`)

```bash
git clone https://github.com/your-username/AIEnglishStudyPartner.git
cd AIEnglishStudyPartner

python -m venv venv
# 啟用虛擬環境：
# 在 macOS/Linux 上：
source venv/bin/activate
# 在 Windows 上：
venv\Scripts\activate

python install.py
```

安裝腳本將會：

  - 安裝基本依賴項

  - 安裝啟用 CUDA 的 PyTorch

  - 從 GitHub 安裝 MeloTTS

  - 下載 UniDic 日語詞典

### 🔑 4. 設定你的環境變數

複製 `.env.example` 並填入你的 tokens：

```
cp .env.example .env
```

確保 `.env` 中的以下變數設定正確：

  - `NGROK_TOKEN`

  - `LINE_CHANNEL_ACCESS_TOKEN`

  - `LINE_CHANNEL_SECRET`

  - `HF_TOKEN`
      

-----

# 專案結構

```
AIEnglishStudyPartner/
├── install.py               # 一鍵安裝腳本
├── .env.example             # 環境變數範本
├── main.py                  # 入口點 (LINE bot 伺服器)
├── requirements.txt         # Python 依賴項
├── models/                  # LLM、STT、TTS、VLM 整合
│   ├── llm.py
│   ├── stt.py
│   ├── tts.py
│   ├── vlm.py
│   ├── rag.py
│   ├── relationalDB.py
│   ├── schemas.py
│   └── __init__.py
├── db
│   ├── vector_db/           # RAG 的向量儲存庫
│   ├── relational_db.db
├── static/                  # 靜態檔案 (圖片、音訊)
└── templates/               # HTML 模板 
```

-----

# 版權宣告

Apache-2.0 授權

[(返回頂部)](https://www.google.com/search?q=%23%E7%9B%AE%E9%8C%84)
