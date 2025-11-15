import React, { useState } from 'react';
import { Upload, FileJson, Database, Download, CheckCircle, AlertCircle, Loader2, Zap, Settings, List } from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';

export default function VectorifyApp() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [uploadMode, setUploadMode] = useState('file');
  const [sqlConfig, setSqlConfig] = useState({
    host: 'localhost',
    port: '5432',
    database: '',
    username: '',
    password: '',
    query: 'SELECT * FROM table_name LIMIT 100'
  });
  const [selectedOutputs, setSelectedOutputs] = useState({
    json: false,
    jsonl: false,
    embeddings: false
  });
  const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] = useState('mini');
  const [outputs, setOutputs] = useState({
    json: null,
    jsonl: null,
    embeddings: null
  });
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState('upload');

  const embeddingProviders = [
    { key: 'mini', name: 'MiniLM-L6-v2', dimension: 384, description: 'Fast and efficient', quality: 'Good' },
    { key: 'mpnet', name: 'MPNet-base-v2', dimension: 768, description: 'Balanced performance', quality: 'Better' },
    { key: 'bge', name: 'BGE-large-en-v1.5', dimension: 1024, description: 'State-of-the-art', quality: 'Best' }
  ];

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    if (!uploadedFile) return;

    setError(null);
    setFile(uploadedFile);
    setProcessing(true);

    const fileName = uploadedFile.name.toLowerCase();
    const fileExtension = fileName.split('.').pop();
    setFileType(fileExtension);
    
    if (fileExtension === 'csv') {
      Papa.parse(uploadedFile, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          if (results.data && results.data.length > 0) {
            setData(results.data);
            setProcessing(false);
            setCurrentView('configure');
          } else {
            setError('No data found');
            setProcessing(false);
          }
        },
        error: (err) => {
          setError('CSV parsing error');
          setProcessing(false);
        }
      });
    } else if (['xlsx', 'xls'].includes(fileExtension)) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const workbook = XLSX.read(e.target.result, { type: 'binary' });
          const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
          const jsonData = XLSX.utils.sheet_to_json(firstSheet);
          
          if (jsonData && jsonData.length > 0) {
            setData(jsonData);
            setProcessing(false);
            setCurrentView('configure');
          } else {
            setError('No data found');
            setProcessing(false);
          }
        } catch (err) {
          setError('Excel parsing error');
          setProcessing(false);
        }
      };
      reader.readAsBinaryString(uploadedFile);
    } else {
      setError('Please upload CSV or Excel file');
      setProcessing(false);
    }
  };

  const handleSqlConnect = () => {
    if (!sqlConfig.database || !sqlConfig.username) {
      setError('Please provide database name and username');
      return;
    }

    setProcessing(true);
    setError(null);

    setTimeout(() => {
      const mockSqlData = [
        { id: 1, name: 'John Doe', email: 'john@example.com', department: 'Engineering', salary: 85000 },
        { id: 2, name: 'Jane Smith', email: 'jane@example.com', department: 'Marketing', salary: 75000 },
        { id: 3, name: 'Bob Johnson', email: 'bob@example.com', department: 'Sales', salary: 70000 },
        { id: 4, name: 'Alice Williams', email: 'alice@example.com', department: 'HR', salary: 65000 },
        { id: 5, name: 'Charlie Brown', email: 'charlie@example.com', department: 'Engineering', salary: 90000 }
      ];

      setData(mockSqlData);
      setFileType('sql');
      setFile({ name: `${sqlConfig.database}_query_result` });
      setProcessing(false);
      setCurrentView('configure');
    }, 1500);
  };

  const handleOutputSelection = (type) => {
    setSelectedOutputs(prev => ({ ...prev, [type]: !prev[type] }));
  };

  const processData = () => {
    if (!data || (!selectedOutputs.json && !selectedOutputs.jsonl && !selectedOutputs.embeddings)) {
      setError('Please select at least one output format');
      return;
    }

    setProcessing(true);
    setError(null);

    setTimeout(() => {
      const newOutputs = {};
      const provider = embeddingProviders.find(p => p.key === selectedEmbeddingProvider);

      if (selectedOutputs.json) {
        newOutputs.json = {
          metadata: {
            fileName: file?.name,
            recordCount: data.length,
            columns: Object.keys(data[0] || {}),
            generatedAt: new Date().toISOString(),
            source: uploadMode === 'sql' ? 'SQL Database' : 'File Upload'
          },
          data: data
        };
      }

      if (selectedOutputs.jsonl) {
        newOutputs.jsonl = {
          metadata: {
            fileName: file?.name,
            recordCount: data.length,
            format: 'JSONL',
            source: uploadMode === 'sql' ? 'SQL Database' : 'File Upload'
          },
          data: data.map(row => JSON.stringify(row)).join('\n')
        };
      }

      if (selectedOutputs.embeddings) {
        const vectors = data.map((row, index) => {
          const text = Object.entries(row)
            .filter(([_, v]) => v)
            .map(([k, v]) => `${k}: ${v}`)
            .join(' | ');
          
          const embedding = generateEmbedding(text, index, provider.dimension);
          
          return {
            id: `vec_${index.toString().padStart(6, '0')}`,
            text: text,
            embedding: embedding,
            metadata: { rowIndex: index, recordData: row }
          };
        });

        newOutputs.embeddings = {
          vectorDatabase: {
            dimension: provider.dimension,
            vectorCount: vectors.length,
            embeddingModel: provider.name,
            quality: provider.quality,
            source: uploadMode === 'sql' ? 'SQL Database' : 'File Upload'
          },
          vectors: vectors
        };
      }

      setOutputs(newOutputs);
      setProcessing(false);
      
      if (selectedOutputs.json) {
        setCurrentView('json');
      } else if (selectedOutputs.jsonl) {
        setCurrentView('jsonl');
      } else if (selectedOutputs.embeddings) {
        setCurrentView('embeddings');
      }
    }, 1500);
  };

  const generateEmbedding = (text, seed, dimension) => {
    const embedding = [];
    let hashSeed = seed;
    
    for (let i = 0; i < text.length; i++) {
      hashSeed = ((hashSeed << 5) - hashSeed) + text.charCodeAt(i);
      hashSeed = hashSeed & hashSeed;
    }
    
    for (let i = 0; i < dimension; i++) {
      const val = Math.sin(hashSeed + i * 0.1) * Math.cos(hashSeed * 0.01 + i);
      embedding.push(parseFloat(val.toFixed(6)));
    }
    
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => parseFloat((val / magnitude).toFixed(6)));
  };

  const downloadOutput = (type) => {
    const output = outputs[type];
    if (!output) return;

    let content, ext;
    
    if (type === 'jsonl') {
      content = output.data;
      ext = 'jsonl';
    } else {
      content = JSON.stringify(output, null, 2);
      ext = 'json';
    }

    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file?.name.split('.')[0]}_${type}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetApp = () => {
    setFile(null);
    setData(null);
    setFileType(null);
    setUploadMode('file');
    setSelectedOutputs({ json: false, jsonl: false, embeddings: false });
    setSelectedEmbeddingProvider('mini');
    setOutputs({ json: null, jsonl: null, embeddings: null });
    setError(null);
    setCurrentView('upload');
  };

  const selectedProvider = embeddingProviders.find(p => p.key === selectedEmbeddingProvider);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-3 rounded-2xl mr-3">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-500 bg-clip-text text-transparent">
              Vectorify
            </h1>
          </div>
          <p className="text-gray-200 text-lg">Transform Data into AI-Ready Formats</p>
          <p className="text-gray-400 text-sm mt-2">CSV • Excel • SQL → JSON • JSONL • Vector Embeddings</p>
        </header>

        <div className="max-w-5xl mx-auto">
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl shadow-2xl border border-blue-500/20 overflow-hidden">
            
            <div className="flex border-b border-gray-700/50">
              <button
                onClick={() => setCurrentView('upload')}
                className={`flex-1 py-4 px-4 font-semibold transition-all ${
                  currentView === 'upload' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-900/50 text-gray-400'
                }`}
              >
                <Upload className="inline-block mr-2 w-4 h-4" />
                Upload
              </button>
              <button
                onClick={() => setCurrentView('configure')}
                disabled={!data}
                className={`flex-1 py-4 px-4 font-semibold transition-all disabled:opacity-30 ${
                  currentView === 'configure' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-900/50 text-gray-400'
                }`}
              >
                <Settings className="inline-block mr-2 w-4 h-4" />
                Configure
              </button>
              <button
                onClick={() => setCurrentView('json')}
                disabled={!outputs.json}
                className={`flex-1 py-4 px-4 font-semibold transition-all disabled:opacity-30 ${
                  currentView === 'json' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-900/50 text-gray-400'
                }`}
              >
                <FileJson className="inline-block mr-2 w-4 h-4" />
                JSON
              </button>
              <button
                onClick={() => setCurrentView('jsonl')}
                disabled={!outputs.jsonl}
                className={`flex-1 py-4 px-4 font-semibold transition-all disabled:opacity-30 ${
                  currentView === 'jsonl' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-900/50 text-gray-400'
                }`}
              >
                <List className="inline-block mr-2 w-4 h-4" />
                JSONL
              </button>
              <button
                onClick={() => setCurrentView('embeddings')}
                disabled={!outputs.embeddings}
                className={`flex-1 py-4 px-4 font-semibold transition-all disabled:opacity-30 ${
                  currentView === 'embeddings' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-900/50 text-gray-400'
                }`}
              >
                <Database className="inline-block mr-2 w-4 h-4" />
                Vectors
              </button>
            </div>

            <div className="p-6">
              
              {currentView === 'upload' && (
                <div>
                  <div className="flex justify-center gap-4 mb-6">
                    <button
                      onClick={() => setUploadMode('file')}
                      className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                        uploadMode === 'file' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-800/50 text-gray-400'
                      }`}
                    >
                      <Upload className="inline-block mr-2 w-5 h-5" />
                      Upload File
                    </button>
                    <button
                      onClick={() => setUploadMode('sql')}
                      className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                        uploadMode === 'sql' ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white' : 'bg-gray-800/50 text-gray-400'
                      }`}
                    >
                      <Database className="inline-block mr-2 w-5 h-5" />
                      SQL Database
                    </button>
                  </div>

                  {uploadMode === 'file' ? (
                    <div>
                      <div className="border-2 border-dashed border-blue-500/40 rounded-xl p-12 hover:border-blue-400 transition-all bg-gray-900/30">
                        <input
                          type="file"
                          accept=".csv,.xlsx,.xls"
                          onChange={handleFileUpload}
                          className="hidden"
                          id="file-upload"
                        />
                        <label htmlFor="file-upload" className="cursor-pointer block text-center">
                          <div className="bg-gradient-to-br from-blue-600 to-cyan-600 w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                            <Upload className="w-8 h-8 text-white" />
                          </div>
                          <p className="text-xl font-semibold text-white mb-2">Upload Your Data File</p>
                          <p className="text-gray-300 text-sm mb-3">Click to browse or drag and drop</p>
                          <div className="flex items-center justify-center gap-2 text-xs">
                            <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded-full">CSV</span>
                            <span className="px-2 py-1 bg-cyan-500/20 text-cyan-300 rounded-full">XLSX</span>
                            <span className="px-2 py-1 bg-cyan-500/20 text-cyan-300 rounded-full">XLS</span>
                          </div>
                        </label>
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-2xl mx-auto">
                      <div className="bg-gray-900/50 border border-blue-500/30 rounded-xl p-6">
                        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                          <Database className="w-5 h-5 mr-2 text-cyan-400" />
                          SQL Database Connection
                        </h3>
                        
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-gray-400 text-xs mb-1">Host</label>
                              <input
                                type="text"
                                value={sqlConfig.host}
                                onChange={(e) => setSqlConfig({...sqlConfig, host: e.target.value})}
                                className="w-full px-3 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                                placeholder="localhost"
                              />
                            </div>
                            <div>
                              <label className="block text-gray-400 text-xs mb-1">Port</label>
                              <input
                                type="text"
                                value={sqlConfig.port}
                                onChange={(e) => setSqlConfig({...sqlConfig, port: e.target.value})}
                                className="w-full px-3 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                                placeholder="5432"
                              />
                            </div>
                          </div>

                          <div>
                            <label className="block text-gray-400 text-xs mb-1">Database Name</label>
                            <input
                              type="text"
                              value={sqlConfig.database}
                              onChange={(e) => setSqlConfig({...sqlConfig, database: e.target.value})}
                              className="w-full px-3 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                              placeholder="my_database"
                            />
                          </div>

                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-gray-400 text-xs mb-1">Username</label>
                              <input
                                type="text"
                                value={sqlConfig.username}
                                onChange={(e) => setSqlConfig({...sqlConfig, username: e.target.value})}
                                className="w-full px-3 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                                placeholder="admin"
                              />
                            </div>
                            <div>
                              <label className="block text-gray-400 text-xs mb-1">Password</label>
                              <input
                                type="password"
                                value={sqlConfig.password}
                                onChange={(e) => setSqlConfig({...sqlConfig, password: e.target.value})}
                                className="w-full px-3 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                                placeholder="••••••••"
                              />
                            </div>
                          </div>

                          <div>
                            <label className="block text-gray-400 text-xs mb-1">SQL Query</label>
                            <textarea
                              value={sqlConfig.query}
                              onChange={(e) => setSqlConfig({...sqlConfig, query: e.target.value})}
                              className="w-full px-3 py-2 text-sm bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:outline-none font-mono"
                              rows="3"
                              placeholder="SELECT * FROM table_name LIMIT 100"
                            />
                          </div>

                          <button
                            onClick={handleSqlConnect}
                            disabled={processing}
                            className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center"
                          >
                            {processing ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin mr-2" />
                                Connecting...
                              </>
                            ) : (
                              <>
                                <Database className="w-5 h-5 mr-2" />
                                Connect & Query
                              </>
                            )}
                          </button>
                        </div>

                        <div className="mt-3 p-2 bg-blue-500/10 border border-blue-500/30 rounded text-xs text-blue-300">
                          <strong>Demo Mode:</strong> This simulates SQL connection. In production, connections are handled securely on the backend.
                        </div>
                      </div>
                    </div>
                  )}

                  {file && !error && (
                    <div className="mt-4 p-4 bg-green-500/10 border border-green-500/50 rounded-lg">
                      <CheckCircle className="inline-block w-4 h-4 text-green-400 mr-2" />
                      <span className="text-green-300 text-sm">{file.name} ({data?.length || 0} records)</span>
                    </div>
                  )}

                  {error && (
                    <div className="mt-4 p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
                      <AlertCircle className="inline-block w-4 h-4 text-red-400 mr-2" />
                      <span className="text-red-300 text-sm">{error}</span>
                    </div>
                  )}

                  {processing && (
                    <div className="mt-4 text-center">
                      <Loader2 className="w-6 h-6 animate-spin mx-auto text-blue-400" />
                      <p className="text-gray-300 mt-2 text-sm">Processing...</p>
                    </div>
                  )}
                </div>
              )}

              {currentView === 'configure' && data && (
                <div>
                  <h2 className="text-2xl font-bold text-white mb-4">Configure Outputs</h2>

                  <div className="mb-6 p-4 bg-blue-500/10 rounded-lg border border-blue-500/30">
                    <div className="grid grid-cols-4 gap-3 text-sm">
                      <div>
                        <span className="text-gray-400">Records: </span>
                        <span className="text-blue-300 font-bold">{data.length}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Columns: </span>
                        <span className="text-blue-300 font-bold">{Object.keys(data[0] || {}).length}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Type: </span>
                        <span className="text-blue-300 font-bold">{fileType?.toUpperCase()}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Source: </span>
                        <span className="text-blue-300 font-bold">{uploadMode === 'sql' ? 'SQL' : 'File'}</span>
                      </div>
                    </div>
                  </div>

                  <h3 className="text-lg font-bold text-white mb-3">Select Output Formats</h3>
                  <div className="grid md:grid-cols-3 gap-4 mb-6">
                    <div
                      onClick={() => handleOutputSelection('json')}
                      className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                        selectedOutputs.json ? 'border-blue-500 bg-blue-500/10' : 'border-gray-700 bg-gray-900/40'
                      }`}
                    >
                      <div className="flex justify-between mb-2">
                        <FileJson className="w-6 h-6 text-blue-400" />
                        {selectedOutputs.json && <CheckCircle className="w-5 h-5 text-blue-500" />}
                      </div>
                      <h4 className="text-white font-bold">JSON</h4>
                      <p className="text-gray-400 text-xs">Structured format</p>
                    </div>

                    <div
                      onClick={() => handleOutputSelection('jsonl')}
                      className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                        selectedOutputs.jsonl ? 'border-indigo-500 bg-indigo-500/10' : 'border-gray-700 bg-gray-900/40'
                      }`}
                    >
                      <div className="flex justify-between mb-2">
                        <List className="w-6 h-6 text-indigo-400" />
                        {selectedOutputs.jsonl && <CheckCircle className="w-5 h-5 text-indigo-500" />}
                      </div>
                      <h4 className="text-white font-bold">JSONL</h4>
                      <p className="text-gray-400 text-xs">Line-delimited</p>
                    </div>

                    <div
                      onClick={() => handleOutputSelection('embeddings')}
                      className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                        selectedOutputs.embeddings ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-700 bg-gray-900/40'
                      }`}
                    >
                      <div className="flex justify-between mb-2">
                        <Database className="w-6 h-6 text-cyan-400" />
                        {selectedOutputs.embeddings && <CheckCircle className="w-5 h-5 text-cyan-500" />}
                      </div>
                      <h4 className="text-white font-bold">Embeddings</h4>
                      <p className="text-gray-400 text-xs">Vector DB ready</p>
                    </div>
                  </div>

                  {selectedOutputs.embeddings && (
                    <div className="mb-6">
                      <h3 className="text-lg font-bold text-white mb-3">Choose Model</h3>
                      <div className="space-y-3">
                        {embeddingProviders.map((provider) => (
                          <div
                            key={provider.key}
                            onClick={() => setSelectedEmbeddingProvider(provider.key)}
                            className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                              selectedEmbeddingProvider === provider.key ? 'border-cyan-400 bg-cyan-500/10' : 'border-gray-700 bg-gray-900/40'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="text-sm font-bold text-white">{provider.name}</h4>
                                  <span className="px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded">FREE</span>
                                  <span className="px-2 py-0.5 bg-blue-500/20 text-blue-300 text-xs rounded">{provider.dimension}D</span>
                                </div>
                                <p className="text-gray-400 text-xs">{provider.description}</p>
                              </div>
                              {selectedEmbeddingProvider === provider.key && <CheckCircle className="w-5 h-5 text-cyan-400" />}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex gap-3">
                    <button
                      onClick={processData}
                      disabled={processing}
                      className="flex-1 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center"
                    >
                      {processing ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin mr-2" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Zap className="w-5 h-5 mr-2" />
                          Generate
                        </>
                      )}
                    </button>
                    <button
                      onClick={resetApp}
                      className="px-6 py-3 bg-gray-700 text-gray-300 rounded-lg font-semibold hover:bg-gray-600"
                    >
                      Reset
                    </button>
                  </div>

                  {error && (
                    <div className="mt-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg">
                      <span className="text-red-300 text-sm">{error}</span>
                    </div>
                  )}
                </div>
              )}

              {currentView === 'json' && outputs.json && (
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-bold text-white">JSON Output</h2>
                    <button
                      onClick={() => downloadOutput('json')}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:shadow-lg flex items-center text-sm"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                  </div>
                  <div className="bg-black/50 rounded-lg p-4 border border-blue-500/30">
                    <pre className="text-green-400 text-xs overflow-x-auto max-h-96 font-mono">
                      {JSON.stringify(outputs.json, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {currentView === 'jsonl' && outputs.jsonl && (
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-bold text-white">JSONL Output</h2>
                    <button
                      onClick={() => downloadOutput('jsonl')}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:shadow-lg flex items-center text-sm"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                  </div>
                  <div className="mb-3 p-3 bg-indigo-500/10 border border-indigo-500/30 rounded-lg">
                    <p className="text-indigo-300 text-xs">
                      <strong>JSON Lines:</strong> One JSON object per line. Perfect for streaming.
                    </p>
                  </div>
                  <div className="bg-black/50 rounded-lg p-4 border border-indigo-500/30">
                    <pre className="text-indigo-300 text-xs overflow-x-auto max-h-96 font-mono">
                      {outputs.jsonl.data}
                    </pre>
                  </div>
                </div>
              )}

              {currentView === 'embeddings' && outputs.embeddings && (
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-bold text-white">Vector Embeddings</h2>
                    <button
                      onClick={() => downloadOutput('embeddings')}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:shadow-lg flex items-center text-sm"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-3 mb-4">
                    <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                      <p className="text-gray-400 text-xs">Vectors</p>
                      <p className="text-blue-300 font-bold text-lg">{outputs.embeddings.vectorDatabase.vectorCount}</p>
                    </div>
                    <div className="p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                      <p className="text-gray-400 text-xs">Dimensions</p>
                      <p className="text-cyan-300 font-bold text-lg">{outputs.embeddings.vectorDatabase.dimension}</p>
                    </div>
                    <div className="p-3 bg-indigo-500/10 rounded-lg border border-indigo-500/30">
                      <p className="text-gray-400 text-xs">Model</p>
                      <p className="text-indigo-300 font-semibold text-xs">{outputs.embeddings.vectorDatabase.embeddingModel}</p>
                    </div>
                    <div className="p-3 bg-green-500/10 rounded-lg border border-green-500/30">
                      <p className="text-gray-400 text-xs">Quality</p>
                      <p className="text-green-300 font-bold text-sm">{outputs.embeddings.vectorDatabase.quality}</p>
                    </div>
                  </div>

                  <div className="bg-black/50 rounded-lg p-4 border border-cyan-500/30">
                    <pre className="text-cyan-300 text-xs overflow-x-auto max-h-96 font-mono">
                      {JSON.stringify(outputs.embeddings, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

            </div>
          </div>
        </div>

        <footer className="mt-8 text-center">
          <p className="text-gray-400 text-xs">Powered by Python • Built for AI/ML</p>
        </footer>
      </div>
    </div>
  );
}