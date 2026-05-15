import React, { useState, useRef, useEffect } from 'react';
import { Send, Plus, MessageSquare, Trash2, ChevronDown, ChevronRight, FileText, Scale, ThumbsUp, ThumbsDown } from 'lucide-react';
import './Chatbot.css';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [showSessions, setShowSessions] = useState(true);
  const [expandedRefs, setExpandedRefs] = useState({});
  const [messagesRemaining, setMessagesRemaining] = useState(10);
  const [limitReached, setLimitReached] = useState(false);
  const [feedbackStates, setFeedbackStates] = useState({});
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const API_BASE = '';

  useEffect(() => {
    loadSessions();
    checkChatLimit();
  }, []);

  const checkChatLimit = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/chat-limit`);
      const data = await response.json();
      setMessagesRemaining(data.remaining);
      setLimitReached(data.exhausted);
    } catch (error) {
      console.error('Failed to check chat limit:', error);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/sessions`);
      const data = await response.json();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const createNewSession = () => {
    setSessionId(null);
    setMessages([]);
  };

  const loadSessionHistory = async (sid) => {
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sid}`);
      if (response.ok) {
        const data = await response.json();
        setSessionId(sid);
        setMessages(data.messages || []);
      }
    } catch (error) {
      console.error('Failed to load session:', error);
    }
  };

  const deleteSession = async (e, sid) => {
    e.stopPropagation();
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${sid}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        if (sid === sessionId) {
          setSessionId(null);
          setMessages([]);
        }
        loadSessions();
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  const toggleRefs = (idx) => {
    setExpandedRefs(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const getQuestion = (idx) => {
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].content;
    }
    return '';
  };

  const handleThumbsUp = (idx) => {
    setFeedbackStates(prev => ({ ...prev, [idx]: { status: 'liked', editText: '' } }));
  };

  const handleThumbsDown = (idx) => {
    setFeedbackStates(prev => ({
      ...prev,
      [idx]: { status: 'editing', editText: messages[idx].content },
    }));
  };

  const handleFeedbackCancel = (idx) => {
    setFeedbackStates(prev => ({ ...prev, [idx]: { status: 'idle', editText: '' } }));
  };

  const handleEditChange = (idx, text) => {
    setFeedbackStates(prev => ({ ...prev, [idx]: { ...prev[idx], editText: text } }));
  };

  const submitFeedback = async (idx) => {
    const fb = feedbackStates[idx];
    if (!fb || fb.status !== 'editing') return;
    const edited_draft = fb.editText.trim();
    if (edited_draft) {
      try {
        await fetch(`${API_BASE}/api/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            question: getQuestion(idx),
            original_draft: messages[idx].content,
            edited_draft,
            sources: messages[idx].sources || [],
          }),
        });
      } catch (error) {
        console.error('Failed to submit feedback:', error);
      }
    }
    setFeedbackStates(prev => ({ ...prev, [idx]: { status: 'submitted', editText: '' } }));
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    let currentSessionId = sessionId;
    if (!currentSessionId) {
      try {
        const response = await fetch(`${API_BASE}/api/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: input.substring(0, 50) }),
        });
        if (response.ok) {
          const newSession = await response.json();
          currentSessionId = newSession.session_id;
          setSessionId(currentSessionId);
        }
      } catch (error) {
        console.error('Failed to create session:', error);
        return;
      }
    }

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: input,
          session_id: currentSessionId,
          num_results: 5,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        if (response.status === 429) {
          setLimitReached(true);
          setMessagesRemaining(0);
        }
        throw new Error(errorData.detail || 'Failed to get response');
      }

      const data = await response.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
      }]);

      // Update remaining messages
      if (data.messages_remaining !== undefined) {
        setMessagesRemaining(data.messages_remaining);
        if (data.messages_remaining <= 0) {
          setLimitReached(true);
        }
      }
      loadSessions();
    } catch (error) {
      const errorMsg = error.message || 'Something went wrong. Please try again.';
      // Only show error message if not a limit error (limit error shown in UI)
      if (!limitReached) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: errorMsg,
        }]);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(e);
    }
  };

  const exampleQuestions = [
    "What are the key legal procedures in this document?",
    "Summarize the main judgments",
    "What cases are referenced here?"
  ];

  return (
    <div className="chat-container">
      {/* Sidebar */}
      <aside className="chat-sidebar">
        <button className="new-chat-btn" onClick={createNewSession}>
          <Plus size={18} />
          <span>New chat</span>
        </button>

        <div className="sidebar-section">
          <button className="section-toggle" onClick={() => setShowSessions(!showSessions)}>
            {showSessions ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            <span>Recent</span>
          </button>

          {showSessions && (
            <div className="sessions-list">
              {sessions.length === 0 ? (
                <p className="empty-state">No conversations yet</p>
              ) : (
                sessions.map((session) => (
                  <div
                    key={session.session_id}
                    className={`session-item ${session.session_id === sessionId ? 'active' : ''}`}
                    onClick={() => loadSessionHistory(session.session_id)}
                  >
                    <MessageSquare size={16} />
                    <span className="session-title">{session.title}</span>
                    <button
                      className="delete-btn"
                      onClick={(e) => deleteSession(e, session.session_id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">
                <Scale size={48} />
              </div>
              <h1>Pearson Specter Litt Legal Assistant</h1>
              <p>Ask questions about your legal documents</p>

              <div className="quick-prompts">
                {exampleQuestions.map((q, i) => (
                  <button
                    key={i}
                    className="prompt-btn"
                    onClick={() => setInput(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-inner">
                    <div className="avatar">
                      {msg.role === 'user' ? 'Mike Ross' : 'PSL'}
                    </div>
                    <div className="message-body">
                      <div className="message-text">{msg.content}</div>

                      {msg.sources && msg.sources.length > 0 && (
                        <div className="sources-section">
                          <button
                            className="sources-toggle"
                            onClick={() => toggleRefs(idx)}
                          >
                            <FileText size={14} />
                            <span>{msg.sources.length} sources</span>
                            {expandedRefs[idx] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                          </button>

                          {expandedRefs[idx] && (
                            <div className="sources-list">
                              {msg.sources.map((source) => (
                                <div key={source.id} className="source-item">
                                  <span className="source-num">{source.id}</span>
                                  <div className="source-info">
                                    <span className="source-file">{source.filename}</span>
                                    {source.case_name && (
                                      <span className="source-case">{source.case_name}</span>
                                    )}
                                  </div>
                                  <span className="source-score">
                                    {Math.round((source.relevance_score || 0) * 100)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {msg.role === 'assistant' && (() => {
                        const fb = feedbackStates[idx] || { status: 'idle' };
                        if (fb.status === 'submitted') {
                          return (
                            <p className="feedback-submitted">Thanks for your feedback!</p>
                          );
                        }
                        if (fb.status === 'editing') {
                          return (
                            <div className="feedback-edit-panel">
                              <textarea
                                value={fb.editText}
                                onChange={(e) => handleEditChange(idx, e.target.value)}
                                rows={5}
                                placeholder="Edit the response to improve it..."
                              />
                              <div className="feedback-edit-actions">
                                <button className="feedback-cancel-btn" onClick={() => handleFeedbackCancel(idx)}>Cancel</button>
                                <button className="feedback-submit-btn" onClick={() => submitFeedback(idx)}>Submit correction</button>
                              </div>
                            </div>
                          );
                        }
                        return (
                          <div className="feedback-actions">
                            <button
                              className={`feedback-btn${fb.status === 'liked' ? ' liked' : ''}`}
                              onClick={() => handleThumbsUp(idx)}
                              title="Good response"
                              disabled={fb.status === 'liked'}
                            >
                              <ThumbsUp size={16} />
                            </button>
                            <button
                              className="feedback-btn"
                              onClick={() => handleThumbsDown(idx)}
                              title="Improve this response"
                            >
                              <ThumbsDown size={16} />
                            </button>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message assistant">
                  <div className="message-inner">
                    <div className="avatar">S</div>
                    <div className="message-body">
                      <div className="thinking">
                        <span></span><span></span><span></span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="input-area">
          {limitReached ? (
            <div className="limit-reached">
              <p>You've used all 10 messages. Thank you for trying Pearson Specter Litt Legal Assistant!</p>
            </div>
          ) : (
            <form className="input-form" onSubmit={sendMessage}>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about legal documents..."
                disabled={loading}
                rows={1}
              />
              <button type="submit" disabled={loading || !input.trim()}>
                <Send size={20} />
              </button>
            </form>
          )}
          <div className="input-footer">
            <div className="battery-container">
              <div className="battery">
                <div
                  className={`battery-level ${messagesRemaining <= 3 ? 'low' : messagesRemaining <= 6 ? 'medium' : 'full'}`}
                  style={{ width: `${(messagesRemaining / 10) * 100}%` }}
                ></div>
              </div>
              <span className="battery-text">Message remaining: {messagesRemaining}</span>
            </div>
            <span className="ai-disclaimer">AI can make mistakes</span>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Chatbot;
