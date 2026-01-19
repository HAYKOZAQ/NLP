from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import os

# Orin AI Pro - Legal Assistant (Restart Trigger)
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.ingestion import DocumentProcessor
from src.core.indexing import SearchIndex
from src.core.retrieval import HybridRetriever
from src.core.generator import get_generator, create_rag_chain
from src.evaluation.research_benchmark import HallucinationChecker

app = Flask(__name__)
CORS(app)

# Fix for Windows Console Encoding
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Global State
SYSTEM_STATE = {
    "status": "Initializing",
    "docs_count": 0,
    "index": None,
    "retriever": None,
    "llm": None,
    "chain": None,
    "hallucination_checker": None,
}


def initialize_system():
    try:
        data_dir = os.path.join(project_root, "data")
        index_path = os.path.join(project_root, "index")
        index = SearchIndex()

        if os.path.exists(index_path) and os.path.exists(
            os.path.join(index_path, "chunks.pkl")
        ):
            print("[FAST] Loading Persistent Index...")
            index.load_indexes(index_path)
        else:
            print("[SLOW] Building New Index...")
            proc = DocumentProcessor()
            chunks = proc.process_directory(data_dir, use_semantic=True)
            if not chunks:
                print("No chunks found!")
                SYSTEM_STATE["status"] = "No Docs"
                return
            index.build_indexes(chunks)
            index.save_indexes(index_path)

        SYSTEM_STATE["docs_count"] = len(index.chunks)
        SYSTEM_STATE["index"] = index
        SYSTEM_STATE["llm"] = get_generator()
        SYSTEM_STATE["retriever"] = HybridRetriever(index, llm=SYSTEM_STATE["llm"])
        SYSTEM_STATE["chain"] = create_rag_chain(
            SYSTEM_STATE["retriever"], SYSTEM_STATE["llm"]
        )
        SYSTEM_STATE["hallucination_checker"] = HallucinationChecker(
            SYSTEM_STATE["llm"]
        )
        SYSTEM_STATE["status"] = "Active"
        print("[OK] Legal Engine Ready.")
    except Exception as e:
        SYSTEM_STATE["status"] = f"Error: {str(e)}"
        print(f"[ERR] Init Failed: {e}")


@app.route("/")
def home():
    # Use raw string to avoid escape sequence issues
    return render_template_string(
        r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Orin AI Pro | Superior Legal Intelligence</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
  :root {
    --primary: #4f46e5;

    /* layout */
    --bg: #0f172a;          /* app background behind panels */
    --sidebar: #020617;

    /* chatbot (light) */
    --surface: #ffffff;     /* main white surface */
    --surface-2: #f8fafc;   /* subtle light background */
    --border: #e2e8f0;

    /* text (light area) */
    --text-dark: #0f172a;
    --text-muted-dark: #475569;

    /* text (sidebar) */
    --text: #f1f5f9;
    --text-muted: #94a3b8;
  }

  * { box-sizing: border-box; }

  body {
    background: var(--bg);
    margin: 0;
    display: flex;
    height: 100vh;
    overflow: hidden;
    font-family: "Inter", sans-serif;
    font-size: 15px; /* slightly smaller overall */
  }

  .sidebar {
    width: 320px;
    background: var(--sidebar);
    padding: 2.5rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.75rem;
    border-right: 1px solid #1e293b;
    color: var(--text);
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--surface-2); /* light background for chatbot side */
  }

  .chat-area {
    flex: 1;
    padding: 1.75rem 3rem;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    background: var(--surface);          /* white chatbot part */
    color: var(--text-dark);
    border-bottom: 1px solid var(--border);
  }

  .msg {
    max-width: 78%;
    line-height: 1.55;
    padding: 0.9rem 1rem;
    border-radius: 0.9rem;
    animation: slideUp 0.2s ease;
    font-size: 0.95rem; /* smaller message text */
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .user-msg {
    align-self: flex-end;
    background: var(--primary);
    color: #fff;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.18);
  }

  .ai-msg {
    align-self: flex-start;
    background: #f8fafc;
    color: var(--text-dark);
    border: 1px solid var(--border);
    overflow-wrap: break-word;
    word-wrap: break-word;
  }
  
  .ai-msg div, .ai-msg p, .ai-msg span {
    max-width: 100%;
  }

  .input-section {
    padding: 1.25rem 3rem;
    display: flex;
    gap: 1rem;
    background: var(--surface); /* keep input area white too */
  }

  .input-wrapper {
    flex: 1;
    background: #ffffff;
    border-radius: 0.9rem;
    padding: 0.35rem;
    display: flex;
    gap: 0.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  }

  input {
    flex: 1;
    background: transparent;
    border: none;
    padding: 0.75rem 0.9rem;
    color: var(--text-dark);
    font-size: 0.95rem;
    outline: none;
  }

  input::placeholder { color: #94a3b8; }

  button {
    background: var(--primary);
    color: white;
    border: none;
    padding: 0.85rem 1.2rem;
    border-radius: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    transition: 0.15s ease;
  }

  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.18);
  }

  h1 {
    font-family: "Outfit", sans-serif;
    font-size: 1.6rem;
    color: #a5b4fc;
    margin: 0;
  }

  .status {
    font-size: 0.8rem;
    color: #10b981;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.04em;
  }

  .status::before {
    content: "";
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.55);
  }

  .verification-panel {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 1rem;
    padding: 1rem;
    margin-top: auto; /* Push to bottom of sidebar */
  }

  .v-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
  }

  .v-bar-bg {
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    position: relative;
    overflow: hidden;
  }

  .v-bar-fill {
    height: 100%;
    background: var(--primary);
    width: 0%;
    transition: width 0.6s ease;
  }
</style>

    </head>
    <body>
        <div class="sidebar">
            <h1>Orin AI Pro</h1>
            <div class="status">ENGINE ONLINE</div>
            <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">Gemini 3.0 Flash</p>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">Chunks Indexed: {{ count }}</p>

            <div class="verification-panel">
                <div style="font-weight:600; color:#cbd5e1; font-size:0.85rem; margin-bottom:0.8rem; display:flex; align-items:center; gap:6px;">
                    🛡️ Hallucination Guard
                </div>
                <div class="v-stat">
                    <span style="color:#94a3b8">Groundedness Score</span>
                    <span id="verify-score" style="color:#818cf8; font-weight:bold">0%</span>
                </div>
                <div class="v-bar-bg">
                    <div id="verify-bar" class="v-bar-fill"></div>
                </div>
                <div id="verify-status" style="font-size: 0.75rem; margin-top: 0.6rem; color: #94a3b8; line-height: 1.3;">
                    Waiting for analysis...
                </div>
            </div>
        </div>
        
        <div class="main">
            <div class="chat-area" id="chat">
                <div class="msg ai-msg">
                    Ողջույն, ես Orin AI Pro-ն եմ։ Պարտաստ եմ պատասխանել ձեր իրավական հարցերին։ 
                </div>
            </div>
            
            <div class="input-section">
                <div class="input-wrapper">
                    <input type="text" id="userInput" placeholder="Այստեղ գրեք ձեր իրավական հարցը..." onkeypress="if(event.key==='Enter') ask()">
                    <button id="sendBtn" onclick="ask()">Ուղարկել</button>
                </div>
            </div>
        </div>

        <script>
            function appendMessage(type, text, isHtml = false) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = 'msg ' + (type === 'user' ? 'user-msg' : 'ai-msg');
                if (isHtml) {
                    div.innerHTML = text;
                } else {
                    div.textContent = text;
                }
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
                return div;
            }

            async function ask() {
                const input = document.getElementById('userInput');
                const query = input.value.trim();
                if (!query) return;

                appendMessage('user', query);
                input.value = '';

                // Create loading message - use fixed-width span for dots to prevent resize
                const loadingMsg = appendMessage('ai', 'Մտածում եմ');
                
                // Animate dots: . -> .. -> ... -> . (loop every 400ms)
                // Use fixed-width span so bubble size doesn't change
                const baseText = loadingMsg.textContent.replace(/\.+$/, '');
                let dotCount = 0;
                const animateDots = () => {
                    dotCount = (dotCount % 3) + 1;
                    loadingMsg.innerHTML = baseText + '<span style="display:inline-block;width:1.2em;text-align:left;">' + '.'.repeat(dotCount) + '</span>';
                };
                animateDots(); // Show first dot immediately
                const dotInterval = setInterval(animateDots, 400);

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: query})
                    });
                    const data = await response.json();
                    clearInterval(dotInterval);
                    
                    if (data.answer) {
                        loadingMsg.innerHTML = format(data.answer);
                        
                        // Update Verification Panel
                        const score = (data.groundedness * 100).toFixed(0);
                        document.getElementById('verify-score').textContent = score + '%';
                        document.getElementById('verify-bar').style.width = score + '%';
                        
                        const statusEl = document.getElementById('verify-status');
                        if (data.groundedness > 0.8) {
                            statusEl.textContent = 'Highly Retrievable & Grounded';
                            statusEl.style.color = '#10b981';
                        } else if (data.groundedness > 0.5) {
                            statusEl.textContent = 'Likely Accurate (Partial Support)';
                            statusEl.style.color = '#f59e0b';
                        } else {
                            statusEl.textContent = 'Caution: Low Context Support';
                            statusEl.style.color = '#ef4444';
                        }
                        
                    } else {
                        loadingMsg.textContent = 'Error: ' + (data.error || 'Unknown error');
                    }
                } catch (e) {
                    clearInterval(dotInterval);
                    loadingMsg.textContent = 'Failed to connect to Legal engine.';
                }
            }

            function format(text) {
                if (!text) return '';
                
                // Simple formatting - preserve structure without breaking layout
                let html = text
                    // Escape any HTML first
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    // Bold text
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    // Headers
                    .replace(/^###\s+(.*)$/gm, '<div style="font-weight:600; margin:10px 0 5px 0;">$1</div>')
                    // Bullet points - convert to proper bullets with indentation
                    .replace(/^[\s]*[•*-]\s+(.*)$/gm, '<div style="margin-left:16px; text-indent:-10px;">• $1</div>')
                    // Numbered lists
                    .replace(/^[\s]*(\d+\.)\s+(.*)$/gm, '<div style="margin-left:16px;">$1 $2</div>')
                    // Double newlines = paragraph break
                    .replace(/\n\n/g, '<div style="margin-bottom:8px;"></div>')
                    // Single newlines = line break
                    .replace(/\n/g, '<br>');
                
                // Citation styling - bold and noticeable
                html = html.replace(/\[([^\[\]:]+):\s*([^\[\]]+)\]/g, '<span style="color:#4f46e5; font-weight:700; font-size:0.95em;">[📄 $2]</span>');
                
                return html;
            }
        </script>
    </body>
    </html>
    """,
        count=SYSTEM_STATE["docs_count"],
    )


@app.route("/ask", methods=["POST"])
def ask_api():
    if not SYSTEM_STATE["chain"]:
        return jsonify({"error": "Engine warming up..."}), 503
    try:
        question = request.json.get("question")

        # 1. Retrieve first to get context for hallucination check
        retriever = SYSTEM_STATE["retriever"]
        chunks = retriever.retrieve(question, k=4)

        context_text = ""
        for c in chunks:
            source = c["metadata"].get("source", "Unknown")
            context_text += f"\n[Source: {source}]\n{c['text']}\n"

        # 2. Generate answer
        ans = SYSTEM_STATE["chain"].invoke(question)

        # 3. Live Hallucination Guard (The Innovation)
        score_data = {"groundedness": 0.0, "hallucination": 0.0}
        if SYSTEM_STATE["hallucination_checker"]:
            h_res = SYSTEM_STATE["hallucination_checker"].get_hallucination_score(
                ans, context_text
            )
            score_data["groundedness"] = h_res["groundedness_score"]
            score_data["hallucination"] = h_res["hallucination_score"]

        return jsonify(
            {
                "answer": ans,
                "groundedness": score_data["groundedness"],
                "hallucination": score_data["hallucination"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    initialize_system()
    app.run(host="0.0.0.0", port=5151, debug=True)
