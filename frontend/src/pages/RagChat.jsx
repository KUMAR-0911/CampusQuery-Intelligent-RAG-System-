import { useState, useRef, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Send, Trash2, Paperclip, FileText,
  Loader2, Plus, X, UploadCloud, Briefcase, GraduationCap, Award
} from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';
import { useAuth } from '../context/AuthContext';

export default function RagChat() {
  const { user } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const toast = useToast();

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const displayName = user?.name || user?.email?.split('@')[0] || 'User';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Start New Chat (clears backend memory & local session)
  const handleNewChat = useCallback(async () => {
    try {
      await api.post('/clear-chat');
      setMessages([]);
      setUploadedFiles([]);
      toast.success('Started a new chat session! Attach your document to begin.');
    } catch (err) {
      toast.error(`Could not reset session: ${getErrorMessage(err)}`);
    }
  }, [toast]);

  // Listen for resetSession from location state (Sidebar "New Chat" button)
  useEffect(() => {
    if (location.state?.resetSession) {
      handleNewChat();
      navigate('/chat', { replace: true, state: {} });
    }
  }, [location.state, handleNewChat, navigate]);

  // Load existing chat history on mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await api.get('/history');
        if (res.data?.messages && res.data.messages.length > 0) {
          setMessages(res.data.messages.map(m => ({
            role: m.role,
            content: m.content,
            time: new Date()
          })));
        }
      } catch (err) {
        console.warn('Could not fetch previous history:', err);
      }
    };
    fetchHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle document upload - Store in backend with simple loading indicator
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return;
    const formData = new FormData();
    const newFileNames = [];

    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
      newFileNames.push(files[i].name);
    }

    setUploading(true);
    try {
      const res = await api.post('/upload', formData);

      setUploadedFiles((prev) => Array.from(new Set([...prev, ...newFileNames])));
      toast.success(res.data.message || `Uploaded ${files.length} document(s) successfully!`);
    } catch (err) {
      toast.error(`Upload failed: ${getErrorMessage(err)}`);
    } finally {
      setUploading(false);
    }
  };

  // Remove uploaded file from active list
  const handleRemoveFile = (fileName) => {
    setUploadedFiles((prev) => prev.filter((f) => f !== fileName));
    toast.info(`Detached ${fileName}`);
  };

  // User sends a question manually over WebSockets with REST API fallback
  const handleSend = async (customQuestion = null) => {
    const question = (customQuestion || input).trim();
    if (!question) return;

    const userMessage = { role: 'user', content: question, time: new Date() };
    const assistantMessage = {
      role: 'assistant',
      content: '',
      statusText: 'Thinking / Answering...',
      time: new Date(),
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    if (!customQuestion) setInput('');
    setLoading(true);

    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = window.location.hostname || 'localhost';
      const wsUrl = `${wsProtocol}//${wsHost}:8000/ws/chat`;

      const ws = new WebSocket(wsUrl);
      let streamEnded = false;

      ws.onopen = () => {
        ws.send(JSON.stringify({ question }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status' && data.message) {
            setMessages((prev) => {
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              if (lastMsg && lastMsg.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...lastMsg,
                  statusText: data.message,
                };
              }
              return updated;
            });
          } else if (data.type === 'chunk' && data.text) {
            setMessages((prev) => {
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              if (lastMsg && lastMsg.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...lastMsg,
                  content: lastMsg.content + data.text,
                };
              }
              return updated;
            });
          } else if (data.type === 'end') {
            streamEnded = true;
            if (data.answer) {
              setMessages((prev) => {
                const updated = [...prev];
                const lastMsg = updated[updated.length - 1];
                if (lastMsg && lastMsg.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...lastMsg,
                    content: data.answer,
                    retrievedChunks: data.retrieved_chunks || [],
                  };
                }
                return updated;
              });
            }
            ws.close();
            setLoading(false);
          } else if (data.type === 'error') {
            toast.error(data.message || 'WebSocket Error');
            ws.close();
            setLoading(false);
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onerror = async (err) => {
        if (!streamEnded) {
          console.warn('WebSocket stream notice, falling back to HTTP API:', err);
          try {
            const res = await api.post('/chat', { question });
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: 'assistant',
                content: res.data.answer,
                retrievedChunks: res.data.retrieved_chunks || [],
                time: new Date(),
              };
              return updated;
            });
          } catch (restErr) {
            const errorMsg = getErrorMessage(restErr);
            toast.error(errorMsg);
          } finally {
            setLoading(false);
          }
        }
      };

    } catch (err) {
      const errorMsg = getErrorMessage(err);
      toast.error(errorMsg);
      setLoading(false);
    } finally {
      inputRef.current?.focus();
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div
      className="full-page-chat-container"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.pptx,.xlsx,.txt,.md"
        style={{ display: 'none' }}
        onChange={(e) => handleFileUpload(e.target.files)}
      />

      {/* Drag & Drop Glass Overlay */}
      {isDragging && (
        <div className="chat-drag-overlay">
          <UploadCloud size={52} className="text-accent" />
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Drop documents to attach</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            Supports PDF, DOCX, PPTX, XLSX, TXT, MD
          </p>
        </div>
      )}

      {/* ChatGPT Top Bar */}
      <div className="chat-gpt-header">
        <div className="chat-header-left">
          <div className="model-selector-badge">
            <span style={{ fontWeight: 600 }}>Resume & Career Intelligence Assistant</span>
          </div>
        </div>

        <div className="chat-header-right" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button className="btn btn-ghost" onClick={handleNewChat} style={{ fontSize: '0.8125rem', padding: '0.35rem 0.75rem' }}>
            <Plus size={14} /> New Chat
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: 'var(--accent)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>
              {displayName[0]?.toUpperCase()}
            </div>
            <span style={{ fontWeight: 500, fontSize: '0.9rem' }}>{displayName}</span>
          </div>
        </div>
      </div>

      {/* Main Full Page Scrollable Chat Area */}
      <div className="chat-gpt-messages-scroll">
        <div className="chat-gpt-messages-inner">
          
          {messages.length === 0 && (
            <div className="chat-gpt-empty-state">
              <div className="chat-gpt-logo">
                <GraduationCap size={36} color="white" />
              </div>
              <h1 className="chat-gpt-title">Welcome, {displayName}!</h1>
              <p className="chat-gpt-subtitle">
                Upload your resume, job description, course syllabus, or campus documents to get instant, accurate answers tailored for students and job seekers.
              </p>

              {/* Document upload drop zone button */}
              {uploadedFiles.length === 0 && (
                <div style={{ margin: '1rem 0' }}>
                  <button
                    className="chat-upload-cta-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    <Paperclip size={18} />
                    <span>{uploading ? 'Storing & Indexing Document...' : 'Attach Resume or Document'}</span>
                  </button>
                </div>
              )}

              {/* Student & Job Prompt Suggestion Cards */}
              <div className="student-prompt-grid">
                <div
                  className="student-prompt-card"
                  onClick={() => handleSend("Analyze my uploaded document and provide key strengths, skills, and ATS match score.")}
                >
                  <Award size={20} className="prompt-icon text-accent" />
                  <div className="prompt-text">
                    <div className="prompt-heading">ATS Resume & Skill Score</div>
                    <div className="prompt-desc">Check ATS readiness and identify missing technical skills</div>
                  </div>
                </div>

                <div
                  className="student-prompt-card"
                  onClick={() => handleSend("Based on my document, what top technical and behavioral interview questions should I practice?")}
                >
                  <Briefcase size={20} className="prompt-icon text-accent" />
                  <div className="prompt-text">
                    <div className="prompt-heading">Placement Interview Questions</div>
                    <div className="prompt-desc">Get tailored technical & behavioral interview questions</div>
                  </div>
                </div>

                <div
                  className="student-prompt-card"
                  onClick={() => handleSend("Summarize the key information and actionable points in my document.")}
                >
                  <FileText size={20} className="prompt-icon text-accent" />
                  <div className="prompt-text">
                    <div className="prompt-heading">Executive Document Summary</div>
                    <div className="prompt-desc">Extract key points, prerequisites, and main takeaways</div>
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* Render Messages */}
          {messages.map((msg, idx) => {
            const cleanedText = (msg.content || '')
              .replace(/\[\d+\]/g, '')
              .replace(/\[Source\s*.*?:?.*?\]/gi, '')
              .replace(/\[Document Chunk\s*\d*.*?\]/gi, '')
              .replace(/\[\s*filename.*?\s*\]/gi, '')
              .replace(/(^|\n)(Citations?|Sources?):[\s\S]*$/gi, '')
              .replace(/(^|\n)[ \t]*#{1,6}\s*/g, '$1')
              .replace(/\*{1,3}(.*?)\*{1,3}/g, '$1')
              .replace(/[*#]/g, '')
              .replace(/-{2,}/g, '')
              .replace(/[—–]/g, ' ')
              .replace(/(^|\n)[ \t]*-[ \t]+/g, '$1')
              .split('\n')
              .map((line) => line.replace(/[ \t]+/g, ' ').trim())
              .join('\n')
              .replace(/(\d+\..*?)\n\n+(?=\d+\.)/g, '$1\n')
              .replace(/\n{3,}/g, '\n\n')
              .trim();
            const isAssistantLoading = msg.role === 'assistant' && !cleanedText.trim();

            return (
              <div key={idx} className={`chat-message-row ${msg.role}`}>
                <div className="chat-message-avatar">
                  {msg.role === 'user' ? displayName[0]?.toUpperCase() || 'U' : ''}
                </div>
                <div className="chat-message-bubble">
                  {isAssistantLoading ? (
                    <div className="chat-thinking-bar" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Loader2 size={18} className="text-accent" style={{ animation: 'spin 1s linear infinite' }} />
                      <span>{msg.statusText || 'Thinking / Answering...'}</span>
                    </div>
                  ) : (
                    <div dangerouslySetInnerHTML={{ __html: cleanedText.replace(/\n/g, '<br/>') }} />
                  )}
                  <div className="message-timestamp">
                    {formatTime(msg.time)}
                  </div>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ChatGPT Fixed Bottom Input Bar */}
      <div className="chat-gpt-input-container">
        <div className="chat-gpt-input-inner">
          
          {/* Active Indexing Indicator (Only shown while actively indexing) */}
          {uploading && (
            <div className="chat-attached-files-bar" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.25rem 0.5rem' }}>
              <span className="attachment-pill indexing" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', color: 'var(--accent-1)' }}>
                <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                <span>Storing & Indexing Document in vector database...</span>
              </span>
            </div>
          )}

          <form
            className="chat-gpt-input-box"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <button
              type="button"
              className="chat-attach-btn"
              onClick={() => fileInputRef.current?.click()}
              title="Attach document (PDF, DOCX, TXT, PPTX)"
              disabled={uploading}
            >
              <Paperclip size={19} />
            </button>

            <input
              ref={inputRef}
              type="text"
              className="chat-gpt-text-input"
              placeholder="Ask anything about your resume, career, interview questions, or job fit..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              autoComplete="off"
            />

            <button
              type="submit"
              className="chat-gpt-send-btn"
              disabled={loading || !input.trim()}
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          </form>
          
          <div className="chat-disclaimer">
            CampusQuery RAG AI provides trustworthy, document-grounded answers for students & placement candidates.
          </div>
        </div>
      </div>
    </div>
  );
}
