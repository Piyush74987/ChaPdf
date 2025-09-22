import React, { useState } from "react";
import "./index.css";

const API_BASE = import.meta.env.VITE_API_BASE;

export default function App() {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loadingAnswer, setLoadingAnswer] = useState(false);

  const upload = async () => {
    if (!file) return alert("Choose a PDF first");
    setUploading(true);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload_pdf/`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setFileId(data.file_id);
      setMessages(prev => [...prev, { role: "system", text: `Uploaded ${data.filename} (${data.num_chunks} chunks)` }]);
    } catch (e) {
      alert("Upload error: " + e.message);
    } finally {
      setUploading(false);
    }
  };

  const sendQuestion = async () => {
    if (!question.trim() || !fileId) return;
    const userMsg = { role: "user", text: question };
    setMessages(prev => [...prev, userMsg]);
    setQuestion(""); setLoadingAnswer(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file_id: fileId, question }) });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages(prev => [...prev, { role: "assistant", text: data.answer, sources: data.sources }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", text: "Error: " + e.message }]);
    } finally { setLoadingAnswer(false); }
  };

  return (
    <div className="app">
      <h2>PDF Q&A</h2>
      <input type="file" accept="application/pdf" onChange={e => setFile(e.target.files[0])}/>
      <button onClick={upload} disabled={uploading}>{uploading ? "Uploading..." : "Upload PDF"}</button>
      <div className="chat">
        {messages.map((m,i)=><div key={i} className={m.role}><p>{m.text}</p></div>)}
      </div>
      <input value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a question" onKeyDown={e=>e.key==="Enter" && sendQuestion()}/>
      <button onClick={sendQuestion} disabled={loadingAnswer}>{loadingAnswer?"Thinking...":"Send"}</button>
    </div>
  );
}
