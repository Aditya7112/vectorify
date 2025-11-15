# Backend app.py — Render-ready, ONNX-based, full SQL support
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import io
import os
from typing import List, Dict, Any
from docx import Document
import pymysql
import sqlite3
from sqlalchemy import create_engine, text

# Optional API providers (import if available)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except Exception:
    COHERE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Folders
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Embedding providers config (ONNX / API-aware)
EMBEDDING_PROVIDERS = {
    'sentence-transformers-mini': {
        'name': 'all-MiniLM-L6-v2',
        'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimension': 384,
        'type': 'local',
        'model': None
    },
    'sentence-transformers-mpnet': {
        'name': 'all-mpnet-base-v2',
        'model_id': 'sentence-transformers/all-mpnet-base-v2',
        'dimension': 768,
        'type': 'local',
        'model': None
    },
    'sentence-transformers-bge': {
        'name': 'bge-large-en-v1.5',
        'model_id': 'BAAI/bge-large-en-v1.5',
        'dimension': 1024,
        'type': 'local',
        'model': None
    },
    'openai-small': {
        'name': 'text-embedding-3-small',
        'dimension': 1536,
        'type': 'api',
        'api_key_env': 'OPENAI_API_KEY'
    },
    'openai-large': {
        'name': 'text-embedding-3-large',
        'dimension': 3072,
        'type': 'api',
        'api_key_env': 'OPENAI_API_KEY'
    },
    'cohere-english': {
        'name': 'embed-english-v3.0',
        'dimension': 1024,
        'type': 'api',
        'api_key_env': 'COHERE_API_KEY'
    }
}

# Global model state (lazy loaded)
current_model = None
current_provider = None


def load_local_model(provider_key: str):
    """Load an ONNX-compatible SentenceTransformer model in CPU mode (lazy)."""
    global current_model, current_provider

    if provider_key not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_key}")

    provider = EMBEDDING_PROVIDERS[provider_key]
    if provider['type'] != 'local':
        raise ValueError(f"Provider {provider_key} is not local")

    # If same model is already loaded, return it
    if current_provider == provider_key and current_model is not None:
        return current_model

    model_id = provider['model_id']
    print(f"Loading model {model_id} in CPU/ONNX-friendly mode...")
    # SentenceTransformer will select available backend; specifying device="cpu" avoids GPU/PyTorch fallback
    # If SentenceTransformers downloads a PyTorch model and no torch present, sentence-transformers can still run ONNX if available.
    current_model = SentenceTransformer(model_id, device="cpu")
    current_provider = provider_key
    print("Model loaded:", model_id)
    return current_model


def generate_embeddings_local(texts: List[str], provider_key: str) -> List[List[float]]:
    """Generate embeddings via SentenceTransformers in CPU mode (ONNX-friendly)."""
    model = load_local_model(provider_key)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
        device="cpu"
    )
    # Ensure Python plain lists (JSON serializable)
    return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]


def generate_embeddings_openai(texts: List[str], model_name: str) -> List[List[float]]:
    """Generate embeddings using OpenAI (if API key set and openai package available)."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed in environment.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)
    # chunking for request size
    chunk_size = 2048
    out = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        resp = client.embeddings.create(input=chunk, model=model_name)
        out.extend([item.embedding for item in resp.data])
    return out


def generate_embeddings_cohere(texts: List[str], model_name: str) -> List[List[float]]:
    """Generate embeddings using Cohere (if available)."""
    if not COHERE_AVAILABLE:
        raise RuntimeError("cohere package not installed in environment.")
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set")
    client = cohere.Client(api_key)
    out = []
    chunk_size = 96
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        resp = client.embed(texts=chunk, model=model_name, input_type='search_document')
        out.extend(resp.embeddings)
    return out


def generate_embeddings(texts: List[str], provider_key: str) -> List[List[float]]:
    """Unified embedding entrypoint — local or API."""
    provider = EMBEDDING_PROVIDERS.get(provider_key)
    if provider is None:
        raise ValueError("Unknown embedding provider: " + provider_key)

    if provider['type'] == 'local':
        return generate_embeddings_local(texts, provider_key)
    elif provider['type'] == 'api':
        # route to OpenAI or Cohere as configured
        if provider_key.startswith("openai"):
            return generate_embeddings_openai(texts, provider['name'])
        if provider_key.startswith("cohere"):
            return generate_embeddings_cohere(texts, provider['name'])
        raise ValueError("API provider not supported")
    else:
        raise ValueError("Unsupported provider type")


def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract paragraphs from docx file."""
    doc = Document(file_path)
    paragraphs = []
    for idx, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if text:
            paragraphs.append({'section': idx + 1, 'content': text, 'wordCount': len(text.split())})
    return paragraphs


@app.route("/api/providers", methods=["GET"])
def api_providers():
    """Return provider list and availability (API keys)"""
    providers_list = []
    for key, cfg in EMBEDDING_PROVIDERS.items():
        info = {
            'key': key,
            'name': cfg.get('name'),
            'dimension': cfg.get('dimension'),
            'type': cfg.get('type'),
            'available': True,
            'requiresApiKey': cfg.get('type') == 'api'
        }
        if cfg.get('type') == 'api':
            info['available'] = bool(os.getenv(cfg.get('api_key_env', '')))
        providers_list.append(info)
    return jsonify({'providers': providers_list, 'default': 'sentence-transformers-mini'})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a file (CSV, XLSX/XLS, DOCX) and return a preview"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename
    filename_lower = filename.lower()
    ext = filename_lower.split('.')[-1]
    if ext not in ['csv', 'xlsx', 'xls', 'docx', 'doc']:
        return jsonify({'error': 'Supported: CSV, XLSX, XLS, DOCX, DOC'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        if ext == 'csv':
            try:
                df = pd.read_csv(save_path, encoding='utf-8')
            except Exception:
                df = pd.read_csv(save_path, encoding='latin-1')
            df = df.dropna(how='all')
            preview = df.fillna('').head(10).to_dict('records')
            return jsonify({'success': True, 'filename': filename, 'fileType': 'CSV',
                            'recordCount': len(df), 'columns': list(df.columns), 'preview': preview,
                            'filepath': save_path})
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(save_path, engine='openpyxl' if ext == 'xlsx' else None)
            df = df.dropna(how='all')
            preview = df.fillna('').head(10).to_dict('records')
            return jsonify({'success': True, 'filename': filename, 'fileType': 'XLSX',
                            'recordCount': len(df), 'columns': list(df.columns), 'preview': preview,
                            'filepath': save_path})
        else:  # docx
            paragraphs = extract_text_from_docx(save_path)
            return jsonify({'success': True, 'filename': filename, 'fileType': 'DOCX',
                            'recordCount': len(paragraphs), 'columns': ['section', 'content', 'wordCount'],
                            'preview': paragraphs[:10], 'filepath': save_path})
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


@app.route("/api/sql/connect", methods=["POST"])
def api_sql_connect():
    """
    Connect to SQL database (postgresql, mysql, sqlite) and execute query.
    Request JSON:
    {
      "dbType": "postgresql"|"mysql"|"sqlite",
      "host": "...",
      "port": 5432,
      "database": "...",
      "username": "...",
      "password": "...",
      "query": "SELECT ..."
    }
    """
    try:
        config = request.json or {}
        db_type = config.get('dbType', 'postgresql')
        host = config.get('host', 'localhost')
        port = config.get('port')
        database = config.get('database')
        username = config.get('username')
        password = config.get('password')
        query = config.get('query', 'SELECT * FROM table_name LIMIT 100')

        if db_type == 'postgresql':
            port = port or 5432
            # use psycopg2 via SQLAlchemy if psycopg2-binary is available
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'mysql':
            port = port or 3306
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'sqlite':
            # database may be a path to file
            connection_string = f"sqlite:///{database}"
        else:
            return jsonify({'error': f'Unsupported database type: {db_type}'}), 400

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(query))
            cols = result.keys()
            rows = result.fetchall()
            data = [dict(zip(cols, row)) for row in rows]

        # Save temporary JSON file for the frontend to reference
        temp_name = f"sql_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_name)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({'success': True, 'filename': temp_name, 'fileType': 'SQL',
                        'recordCount': len(data), 'columns': list(cols), 'preview': data[:10],
                        'filepath': temp_path})
    except Exception as e:
        return jsonify({'error': f'SQL connection error: {str(e)}'}), 500


@app.route("/api/convert", methods=["POST"])
def api_convert():
    """Convert file to JSON/JSONL and optionally generate embeddings.
    Request JSON:
    {
      "filepath": "...",
      "outputTypes": ["json","jsonl","embeddings"],
      "embeddingProvider": "sentence-transformers-mini"
    }
    """
    try:
        payload = request.json or {}
        filepath = payload.get('filepath')
        output_types = payload.get('outputTypes', [])
        embedding_provider = payload.get('embeddingProvider', 'sentence-transformers-mini')

        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        ext = filepath.lower().split('.')[-1]

        # Load into DataFrame
        if ext == 'csv':
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except Exception:
                df = pd.read_csv(filepath, encoding='latin-1')
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(filepath, engine='openpyxl' if ext == 'xlsx' else None)
        elif ext in ['docx']:
            df = pd.DataFrame(extract_text_from_docx(filepath))
        elif ext == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        df = df.replace({np.nan: None})

        results = {}

        # JSON output
        if 'json' in output_types:
            json_output = {
                'metadata': {
                    'fileName': os.path.basename(filepath),
                    'fileType': ext.upper(),
                    'recordCount': len(df),
                    'columns': list(df.columns),
                    'generatedAt': datetime.now().isoformat()
                },
                'data': df.to_dict('records')
            }
            out_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_output.json"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, ensure_ascii=False, indent=2)
            results['json'] = {'filepath': out_path, 'filename': out_name, 'data': json_output}

        # JSONL output
        if 'jsonl' in output_types:
            lines = [json.dumps(row, ensure_ascii=False) for row in df.to_dict('records')]
            out_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_output.jsonl"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            results['jsonl'] = {'filepath': out_path, 'filename': out_name, 'preview': "\n".join(lines[:5])}

        # Embeddings output
        if 'embeddings' in output_types:
            provider_cfg = EMBEDDING_PROVIDERS.get(embedding_provider)
            if not provider_cfg:
                return jsonify({'error': 'Unknown embedding provider'}), 400

            # Prepare texts
            texts = []
            indices = []
            for idx, row in df.iterrows():
                parts = []
                for col in df.columns:
                    val = row[col]
                    if val is not None and str(val).strip() != '':
                        parts.append(f"{col}: {val}")
                text = " | ".join(parts)
                if text.strip():
                    texts.append(text)
                    indices.append(idx)
            if not texts:
                return jsonify({'error': 'No valid text for embeddings'}), 400

            # Generate embeddings (local ONNX)
            if provider_cfg['type'] == 'local':
                vectors = generate_embeddings_local(texts, embedding_provider)
            elif provider_cfg['type'] == 'api':
                vectors = generate_embeddings(texts, embedding_provider)
            else:
                return jsonify({'error': 'Unsupported provider type'}), 400

            # Assemble vectors
            assembled = []
            for i, (idx, txt, emb) in enumerate(zip(indices, texts, vectors)):
                row = df.iloc[idx].replace({np.nan: None}).to_dict()
                assembled.append({
                    'id': f"vec_{str(i).zfill(6)}",
                    'text': txt,
                    'embedding': emb,
                    'metadata': {'rowIndex': int(idx), 'fields': list(df.columns), 'recordData': row}
                })

            emb_output = {
                'vectorDatabase': {
                    'version': '1.0',
                    'dimension': provider_cfg['dimension'],
                    'vectorCount': len(assembled),
                    'embeddingModel': provider_cfg.get('name'),
                    'embeddingProvider': embedding_provider,
                    'distance': 'cosine',
                    'createdAt': datetime.now().isoformat(),
                    'sourceFile': os.path.basename(filepath),
                    'sourceType': ext.upper()
                },
                'vectors': assembled
            }
            out_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_embeddings_{embedding_provider}.json"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(emb_output, f, ensure_ascii=False, indent=2)
            results['embeddings'] = {'filepath': out_path, 'filename': out_name, 'data': emb_output}

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/api/download/<output_type>/<path:filename>", methods=["GET"])
def api_download(output_type: str, filename: str):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, mimetype='application/json', as_attachment=True, download_name=filename)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'providers': list(EMBEDDING_PROVIDERS.keys())
    })


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Vectorify Backend (ONNX-ready) - Starting")
    print("=" * 80)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
