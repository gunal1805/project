# Document Q&A Chatbot

A Streamlit app that lets you ask questions about a PDF. It retrieves relevant
document passages with Gemini embeddings and generates answers grounded in those
passages.

## Requirements

- Python 3.12
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set your Gemini API key in the environment:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

## Run

```powershell
streamlit run app.py
```

Alternatively, enter the API key in the app's sidebar. Upload a text-based PDF,
select **Process document**, and then ask questions in the chat.

For a terminal-based interface, run:

```powershell
python rag_core.py your_document.pdf
```

Do not commit API keys or other secrets.
