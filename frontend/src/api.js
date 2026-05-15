import axios from 'axios';

// Use relative URL for production (HF Spaces), localhost for development
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = async () => {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

// Get system status
export const getSystemStatus = async () => {
  try {
    const response = await api.get('/api/status');
    return response.data;
  } catch (error) {
    console.error('Failed to get system status:', error);
    throw error;
  }
};

// Upload PDF
export const uploadPDF = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Failed to upload PDF:', error);
    throw error;
  }
};

// Query documents
export const queryDocuments = async (question, numResults = 5, useAIAnswer = false) => {
  try {
    const response = await api.post('/api/query', {
      question,
      num_results: numResults,
      use_ai_answer: useAIAnswer,
    });

    return response.data;
  } catch (error) {
    console.error('Failed to query documents:', error);
    throw error;
  }
};

// Chat with documents (multi-turn conversation)
export const chatWithDocuments = async (messages, numResults = 5) => {
  try {
    const response = await api.post('/api/chat', {
      messages,
      num_results: numResults,
    });

    return response.data;
  } catch (error) {
    console.error('Failed to chat:', error);
    throw error;
  }
};

// List documents
export const listDocuments = async () => {
  try {
    const response = await api.get('/api/documents');
    return response.data;
  } catch (error) {
    console.error('Failed to list documents:', error);
    throw error;
  }
};

// Delete document
export const deleteDocument = async (filename) => {
  try {
    const response = await api.delete(`/api/documents/${filename}`);
    return response.data;
  } catch (error) {
    console.error('Failed to delete document:', error);
    throw error;
  }
};

// Get rate limit status
export const getRateLimitStatus = async () => {
  try {
    const response = await api.get('/api/rate-limit');
    return response.data;
  } catch (error) {
    console.error('Failed to get rate limit status:', error);
    throw error;
  }
};

export default api;
