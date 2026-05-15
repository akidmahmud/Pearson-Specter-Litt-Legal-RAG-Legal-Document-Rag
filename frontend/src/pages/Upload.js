import React, { useState, useEffect } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader, FileText, Zap, Database, Brain, ExternalLink } from 'lucide-react';
import { uploadPDF } from '../api';
import './Upload.css';

function UploadPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [messageType, setMessageType] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(null);

  const ALLOWED_TYPES = new Set([
    'application/pdf',
    'image/png',
    'image/jpeg',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  ]);

  const isAllowedFile = (f) => {
    if (ALLOWED_TYPES.has(f.type)) return true;
    const ext = f.name.split('.').pop().toLowerCase();
    return ['pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx'].includes(ext);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (isAllowedFile(selectedFile)) {
        setFile(selectedFile);
        setMessage(null);
      } else {
        setMessage('Please select a PDF, image (PNG/JPG), or Word document (DOC/DOCX)');
        setMessageType('error');
        setFile(null);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isAllowedFile(droppedFile)) {
      setFile(droppedFile);
      setMessage(null);
    } else {
      setMessage('Please drop a PDF, image (PNG/JPG), or Word document (DOC/DOCX)');
      setMessageType('error');
    }
  };

  // Simulate progress through upload steps (stops before Index - waits for actual completion)
  useEffect(() => {
    if (!loading) return;

    const steps = [
      { name: 'Uploading file', progress: 15 },
      { name: 'Extracting text from PDF', progress: 30 },
      { name: 'Analyzing document structure', progress: 45 },
      { name: 'Splitting into chunks', progress: 60 },
      { name: 'Generating embeddings', progress: 75 },
      { name: 'Indexing in vector database', progress: 85 }  // Stops at 85%, waits for actual completion
    ];

    let currentStepIndex = 0;
    setCurrentStep(steps[0].name);
    setUploadProgress(10);

    const interval = setInterval(() => {
      currentStepIndex++;
      if (currentStepIndex < steps.length) {
        const step = steps[currentStepIndex];
        setCurrentStep(step.name);
        setUploadProgress(step.progress);
      } else {
        // Stay at indexing step until actual completion
        clearInterval(interval);
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [loading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setMessage('Please select a file first');
      setMessageType('error');
      return;
    }

    try {
      setLoading(true);
      setMessage(null);
      setUploadProgress(5);
      setCurrentStep('Uploading file');

      const response = await uploadPDF(file);

      setUploadProgress(100);
      setCurrentStep('Complete!');
      setMessage(`✓ Successfully uploaded: ${response.filename} (${response.chunks_added} chunks added)`);
      setMessageType('success');
      setFile(null);

      // Clear form after 2 seconds
      setTimeout(() => {
        setLoading(false);
        setUploadProgress(0);
        setCurrentStep(null);
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
          fileInput.value = '';
        }
      }, 2000);
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      setMessage(`✗ ${errorMessage}`);
      setMessageType('error');
      setLoading(false);
      setUploadProgress(0);
      setCurrentStep(null);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-container-header">
          <h2>Upload Legal Documents</h2>
          <p className="subtitle">Upload documents to add them to the RAG system</p>
        </div>
        <div className="upload-container-body">
        <form onSubmit={handleSubmit} className="upload-form">
          <div
            className={`file-drop-zone ${file ? 'has-file' : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="file-info">
                <CheckCircle size={48} className="success-icon" />
                <p className="file-name">{file.name}</p>
                <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            ) : (
              <>
                <Upload size={48} className="upload-icon" />
                <p className="drop-text">Drag and drop your document here</p>
                <p className="file-types-hint">PDF · PNG · JPG · JPEG · DOC · DOCX</p>
                <p className="or-text">or</p>
                <label htmlFor="file-input" className="file-label">
                  Browse Files
                </label>
              </>
            )}

            <input
              id="file-input"
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.doc,.docx"
              onChange={handleFileChange}
              className="file-input"
              disabled={loading}
            />
          </div>

          {loading && (
            <div className="upload-progress-section">
              <div className="progress-header">
                <h4>Processing Your Document</h4>
                <span className="progress-percent">{Math.round(uploadProgress)}%</span>
              </div>

              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${uploadProgress}%` }}></div>
              </div>

              {currentStep && (
                <div className="current-step">
                  <div className="step-icon">
                    <Loader size={20} className="spinner-icon" />
                  </div>
                  <div className="step-info">
                    <p className="step-name">{currentStep}</p>
                    <p className="step-description">
                      {currentStep === 'Uploading file' && 'Transferring your document to the server...'}
                      {currentStep === 'Extracting text from PDF' && 'Extracting readable text from the document...'}
                      {currentStep === 'Analyzing document structure' && 'Understanding document layout and sections...'}
                      {currentStep === 'Splitting into chunks' && 'Breaking text into manageable pieces for search...'}
                      {currentStep === 'Generating embeddings' && 'Creating AI embeddings for semantic search...'}
                      {currentStep === 'Indexing in vector database' && 'Storing processed content in database...'}
                      {currentStep === 'Complete!' && 'Your document has been successfully processed!'}
                    </p>
                  </div>
                </div>
              )}

              <div className="process-timeline">
                <div className={`timeline-item ${uploadProgress >= 10 ? 'completed' : ''}`}>
                  <FileText size={16} />
                  <span>Upload</span>
                </div>
                <div className={`timeline-item ${uploadProgress >= 30 ? 'completed' : ''}`}>
                  <Zap size={16} />
                  <span>Extract</span>
                </div>
                <div className={`timeline-item ${uploadProgress >= 45 ? 'completed' : ''}`}>
                  <FileText size={16} />
                  <span>Analyze</span>
                </div>
                <div className={`timeline-item ${uploadProgress >= 60 ? 'completed' : ''}`}>
                  <Database size={16} />
                  <span>Chunk</span>
                </div>
                <div className={`timeline-item ${uploadProgress >= 80 ? 'completed' : ''}`}>
                  <Brain size={16} />
                  <span>Embed</span>
                </div>
                <div className={`timeline-item ${uploadProgress >= 95 ? 'completed' : ''}`}>
                  <Database size={16} />
                  <span>Index</span>
                </div>
              </div>
            </div>
          )}

          {message && !loading && (
            <div className={`message ${messageType}`}>
              {messageType === 'success' ? (
                <CheckCircle size={20} />
              ) : (
                <AlertCircle size={20} />
              )}
              <span>{message}</span>
            </div>
          )}

          <button
            type="submit"
            className="submit-button"
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <Loader size={18} className="spinner" />
                Uploading...
              </>
            ) : (
              <>
                <Upload size={18} />
                Upload Document
              </>
            )}
          </button>
        </form>

        <div className="intended-docs-section">
          <h3>Intended Documents</h3>
          <p className="intended-description">
            This RAG pipeline was curated specifically for <strong>Legal Documents</strong>.
            For optimal results, it is highly recommended to use official legal PDFs.
          </p>
          <div className="example-source">
            <FileText size={18} />
            <span>Example documents available at:</span>
            <a
              href="https://www.supremecourt.gov.bd/web/?page=bulletin.php&menu=10&lang="
              target="_blank"
              rel="noopener noreferrer"
              className="source-link"
            >
              Supreme Court of Bangladesh - Bulletins
              <ExternalLink size={14} />
            </a>
          </div>
        </div>

        <div className="info-section">
          <h3>Upload Guidelines</h3>
          <ul>
            <li>Supported formats: PDF, PNG, JPG/JPEG, DOC, DOCX</li>
            <li>Images are processed with OCR to extract text</li>
            <li>The file will be processed and indexed automatically</li>
            <li>Larger files may take a few minutes to process</li>
            <li>The document will be split into chunks for better search</li>
            <li>Legal metadata will be extracted automatically</li>
          </ul>
        </div>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
