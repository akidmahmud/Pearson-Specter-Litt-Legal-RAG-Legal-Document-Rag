import React, { useState, useEffect } from 'react';
import { FileText, Trash2, Loader, AlertCircle } from 'lucide-react';
import { listDocuments, deleteDocument } from '../api';
import './Documents.css';

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await listDocuments();
      setDocuments(data.documents || []);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to load documents';
      setError(`Failed to load documents: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    try {
      setDeleting(filename);
      await deleteDocument(filename);
      setDocuments(documents.filter(doc => doc.filename !== filename));
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Delete failed';
      setError(`Failed to delete: ${errorMessage}`);
    } finally {
      setDeleting(null);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="documents-page">
      <div className="documents-container">
        <div className="documents-container-header">
          <h2>Uploaded Documents</h2>
          <p className="subtitle">Manage your legal documents in the RAG system</p>
        </div>
        <div className="documents-container-body">
        {error && (
          <div className="error-message">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <div className="loading-state">
            <Loader size={40} className="spinner" />
            <p>Loading documents...</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="empty-state">
            <FileText size={48} />
            <h3>No Documents Uploaded</h3>
            <p>Start by uploading PDF documents to build your legal knowledge base.</p>
            <a href="/upload" className="upload-link">
              Upload Your First Document
            </a>
          </div>
        ) : (
          <div className="documents-list">
            <div className="list-header">
              <div className="col-name">Document Name</div>
              <div className="col-size">File Size</div>
              <div className="col-date">Uploaded</div>
              <div className="col-actions">Actions</div>
            </div>

            {documents.map((doc) => (
              <div key={doc.filename} className="document-row">
                <div className="col-name">
                  <FileText size={20} className="doc-icon" />
                  <span className="doc-name" title={doc.filename}>
                    {doc.filename}
                  </span>
                </div>
                <div className="col-size">
                  {formatFileSize(doc.size_bytes)}
                </div>
                <div className="col-date">
                  {formatDate(doc.uploaded_at)}
                </div>
                <div className="col-actions">
                  <button
                    className="action-button delete"
                    onClick={() => handleDelete(doc.filename)}
                    disabled={deleting === doc.filename}
                    title="Delete document"
                  >
                    {deleting === doc.filename ? (
                      <Loader size={18} className="spinner" />
                    ) : (
                      <Trash2 size={18} />
                    )}
                  </button>
                </div>
              </div>
            ))}

            <div className="list-footer">
              <p>Total: {documents.length} document(s)</p>
            </div>
          </div>
        )}

        <div className="info-section">
          <h3>Document Management</h3>
          <ul>
            <li>Each uploaded document is automatically processed and indexed</li>
            <li>Documents are split into chunks for better search accuracy</li>
            <li>Legal metadata is extracted from documents</li>
            <li>You can delete documents at any time</li>
            <li>Search automatically includes all indexed documents</li>
          </ul>
        </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentsPage;
