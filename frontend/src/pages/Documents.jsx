import { useState, useRef } from 'react';
import { UploadCloud, FileText, X, File } from 'lucide-react';
import api, { getErrorMessage } from '../api/api';
import { useToast } from '../context/ToastContext';

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

export default function Documents() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragover, setDragover] = useState(false);
  const inputRef = useRef(null);
  const toast = useToast();

  const addFiles = (newFiles) => {
    const fileArray = Array.from(newFiles);
    setFiles((prev) => [...prev, ...fileArray]);
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragover(false);
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setProgress(10);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file));

      setProgress(30);
      const res = await api.post('/upload', formData, {
        onUploadProgress: (e) => {
          if (e.total) {
            const pct = Math.round((e.loaded / e.total) * 60) + 30;
            setProgress(Math.min(pct, 90));
          }
        },
      });

      setProgress(100);
      toast.success(res.data.message || `Successfully indexed ${files.length} document(s).`);
      setFiles([]);
      setTimeout(() => setProgress(0), 500);
    } catch (err) {
      toast.error(getErrorMessage(err));
      setProgress(0);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Documents</h1>
        <p className="page-subtitle">Upload and index documents for the RAG assistant</p>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Upload Documents</h3>
            <p className="card-subtitle">
              Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT
            </p>
          </div>
        </div>

        {/* Upload Zone */}
        <div
          className={`upload-zone ${dragover ? 'dragover' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
          onDragLeave={() => setDragover(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            onChange={(e) => addFiles(e.target.files)}
            style={{ display: 'none' }}
            accept=".pdf,.docx,.pptx,.xlsx,.html,.md,.txt"
          />
          <div className="upload-zone-icon">
            <UploadCloud size={28} />
          </div>
          <h3>Drop files here or click to browse</h3>
          <p>Select one or more files to upload and index</p>
        </div>

        {/* File list */}
        {files.length > 0 && (
          <div className="file-list">
            {files.map((file, idx) => (
              <div className="file-item" key={idx}>
                <div className="file-item-info">
                  <File size={16} color="var(--accent-3)" />
                  <span className="file-item-name">{file.name}</span>
                  <span className="file-item-size">{formatFileSize(file.size)}</span>
                </div>
                <button className="file-remove" onClick={() => removeFile(idx)} aria-label="Remove file">
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Progress bar */}
        {progress > 0 && (
          <div className="progress-bar" style={{ marginTop: '1rem' }}>
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        )}

        {/* Upload button */}
        <div style={{ marginTop: '1.25rem' }}>
          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={uploading || files.length === 0}
            style={{ width: 'auto' }}
          >
            {uploading ? (
              <><span className="spinner" /> Indexing Documents...</>
            ) : (
              <><UploadCloud size={16} /> Upload &amp; Index ({files.length})</>
            )}
          </button>
        </div>
      </div>

      {/* Empty state when no files selected */}
      {files.length === 0 && !uploading && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">
              <FileText size={28} />
            </div>
            <h3>No files selected</h3>
            <p>Upload your resumes, job descriptions, or academic documents to get started</p>
          </div>
        </div>
      )}
    </div>
  );
}
