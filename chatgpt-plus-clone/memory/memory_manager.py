"""
Memory Manager - Persistent Memory and Context Management
Provides conversation history, vector storage, and context management for the AI assistant
"""

import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import time
from datetime import datetime

# Vector storage imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available - using basic memory storage")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available - using basic similarity")

class MemoryManager:
    """Comprehensive memory management system"""
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self.conversation_history = []
        self.user_preferences = {}
        self.context_memory = {}
        
        # Initialize databases
        self.setup_databases()
        
        # Initialize vector storage
        self.setup_vector_storage()
        
        # Load existing data
        self.load_memory()
    
    def setup_databases(self):
        """Setup SQLite databases for persistent storage"""
        # Main memory database
        self.db_path = self.memory_dir / "memory.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                assistant_response TEXT,
                session_id TEXT,
                metadata TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_memory (
                key TEXT PRIMARY KEY,
                value TEXT,
                importance REAL,
                last_accessed TEXT,
                created_at TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_memory (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT,
                summary TEXT,
                metadata TEXT,
                last_accessed TEXT
            )
        ''')
        
        self.conn.commit()
    
    def setup_vector_storage(self):
        """Setup vector storage for semantic search"""
        if CHROMA_AVAILABLE:
            try:
                # Initialize ChromaDB
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.memory_dir / "chroma"),
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Create collections
                self.conversation_collection = self.chroma_client.get_or_create_collection(
                    name="conversations",
                    metadata={"description": "Conversation history embeddings"}
                )
                
                self.context_collection = self.chroma_client.get_or_create_collection(
                    name="context",
                    metadata={"description": "Context and preferences embeddings"}
                )
                
                self.logger.info("Vector storage initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Error initializing vector storage: {e}")
                self.chroma_client = None
        else:
            self.chroma_client = None
        
        # Initialize sentence transformer for embeddings
        if TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence transformer loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading sentence transformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    def add_interaction(self, user_message: str, assistant_response: str, 
                       session_id: str = None, metadata: Dict[str, Any] = None):
        """Add a new interaction to memory"""
        try:
            timestamp = datetime.now().isoformat()
            session_id = session_id or self.get_current_session_id()
            metadata_json = json.dumps(metadata or {})
            
            # Store in SQLite
            self.cursor.execute('''
                INSERT INTO conversations (timestamp, user_message, assistant_response, session_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, user_message, assistant_response, session_id, metadata_json))
            
            # Store in conversation history
            interaction = {
                'timestamp': timestamp,
                'user_message': user_message,
                'assistant_response': assistant_response,
                'session_id': session_id,
                'metadata': metadata or {}
            }
            self.conversation_history.append(interaction)
            
            # Store in vector storage
            if self.chroma_client and self.embedding_model:
                self.store_in_vector_storage(interaction)
            
            self.conn.commit()
            self.logger.info(f"Added interaction to memory: {len(user_message)} chars")
            
        except Exception as e:
            self.logger.error(f"Error adding interaction to memory: {e}")
    
    def store_in_vector_storage(self, interaction: Dict[str, Any]):
        """Store interaction in vector storage for semantic search"""
        try:
            # Create embedding for the interaction
            text = f"{interaction['user_message']} {interaction['assistant_response']}"
            embedding = self.embedding_model.encode([text])[0].tolist()
            
            # Store in conversation collection
            self.conversation_collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[interaction],
                ids=[f"conv_{int(time.time() * 1000)}"]
            )
            
        except Exception as e:
            self.logger.error(f"Error storing in vector storage: {e}")
    
    def search_similar_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar conversations using vector storage"""
        try:
            if not self.chroma_client or not self.embedding_model:
                return self.search_similar_basic(query, limit)
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in vector storage
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Convert results to standard format
            similar_conversations = []
            for i in range(len(results['metadatas'][0])):
                similar_conversations.append({
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0.0
                })
            
            return similar_conversations
            
        except Exception as e:
            self.logger.error(f"Error searching similar conversations: {e}")
            return []
    
    def search_similar_basic(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Basic similarity search using keyword matching"""
        query_lower = query.lower()
        similar = []
        
        for interaction in self.conversation_history[-100:]:  # Search last 100 interactions
            text = f"{interaction['user_message']} {interaction['assistant_response']}".lower()
            
            # Simple keyword matching
            common_words = set(query_lower.split()) & set(text.split())
            if len(common_words) > 0:
                similarity = len(common_words) / max(len(query_lower.split()), len(text.split()))
                similar.append({
                    'metadata': interaction,
                    'document': text,
                    'distance': 1 - similarity
                })
        
        # Sort by similarity and return top results
        similar.sort(key=lambda x: x['distance'])
        return similar[:limit]
    
    def get_conversation_history(self, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        try:
            if session_id:
                self.cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (session_id, limit))
            else:
                self.cursor.execute('''
                    SELECT * FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = self.cursor.fetchall()
            conversations = []
            
            for row in rows:
                conversations.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'user_message': row[2],
                    'assistant_response': row[3],
                    'session_id': row[4],
                    'metadata': json.loads(row[5]) if row[5] else {}
                })
            
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []
    
    def add_user_preference(self, key: str, value: Any):
        """Add or update a user preference"""
        try:
            timestamp = datetime.now().isoformat()
            value_json = json.dumps(value)
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value_json, timestamp))
            
            self.user_preferences[key] = value
            self.conn.commit()
            
            # Store in vector storage for semantic search
            if self.chroma_client and self.embedding_model:
                self.store_preference_in_vector_storage(key, value)
            
        except Exception as e:
            self.logger.error(f"Error adding user preference: {e}")
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        try:
            self.cursor.execute('SELECT value FROM user_preferences WHERE key = ?', (key,))
            row = self.cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            else:
                return default
                
        except Exception as e:
            self.logger.error(f"Error getting user preference: {e}")
            return default
    
    def add_context_memory(self, key: str, value: Any, importance: float = 1.0):
        """Add context memory with importance weighting"""
        try:
            timestamp = datetime.now().isoformat()
            value_json = json.dumps(value)
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO context_memory (key, value, importance, last_accessed, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (key, value_json, importance, timestamp, timestamp))
            
            self.context_memory[key] = {
                'value': value,
                'importance': importance,
                'last_accessed': timestamp,
                'created_at': timestamp
            }
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error adding context memory: {e}")
    
    def get_context_memory(self, key: str, default: Any = None) -> Any:
        """Get context memory and update last accessed time"""
        try:
            self.cursor.execute('SELECT value, importance FROM context_memory WHERE key = ?', (key,))
            row = self.cursor.fetchone()
            
            if row:
                # Update last accessed time
                timestamp = datetime.now().isoformat()
                self.cursor.execute('''
                    UPDATE context_memory SET last_accessed = ? WHERE key = ?
                ''', (timestamp, key))
                self.conn.commit()
                
                return json.loads(row[0])
            else:
                return default
                
        except Exception as e:
            self.logger.error(f"Error getting context memory: {e}")
            return default
    
    def get_important_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most important context memories"""
        try:
            self.cursor.execute('''
                SELECT key, value, importance, last_accessed 
                FROM context_memory 
                ORDER BY importance DESC, last_accessed DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = self.cursor.fetchall()
            context_list = []
            
            for row in rows:
                context_list.append({
                    'key': row[0],
                    'value': json.loads(row[1]),
                    'importance': row[2],
                    'last_accessed': row[3]
                })
            
            return context_list
            
        except Exception as e:
            self.logger.error(f"Error getting important context: {e}")
            return []
    
    def add_file_memory(self, file_path: str, content: str, summary: str = None):
        """Add file content to memory"""
        try:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            timestamp = datetime.now().isoformat()
            
            # Check if file has changed
            self.cursor.execute('SELECT content_hash FROM file_memory WHERE file_path = ?', (file_path,))
            row = self.cursor.fetchone()
            
            if row and row[0] == content_hash:
                # File hasn't changed, just update last accessed
                self.cursor.execute('''
                    UPDATE file_memory SET last_accessed = ? WHERE file_path = ?
                ''', (timestamp, file_path))
            else:
                # File has changed or is new
                metadata = {
                    'file_size': len(content),
                    'content_hash': content_hash,
                    'added_at': timestamp
                }
                metadata_json = json.dumps(metadata)
                
                self.cursor.execute('''
                    INSERT OR REPLACE INTO file_memory (file_path, content_hash, summary, metadata, last_accessed)
                    VALUES (?, ?, ?, ?, ?)
                ''', (file_path, content_hash, summary, metadata_json, timestamp))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error adding file memory: {e}")
    
    def get_file_memory(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file memory"""
        try:
            self.cursor.execute('''
                SELECT content_hash, summary, metadata, last_accessed 
                FROM file_memory 
                WHERE file_path = ?
            ''', (file_path,))
            
            row = self.cursor.fetchone()
            if row:
                return {
                    'content_hash': row[0],
                    'summary': row[1],
                    'metadata': json.loads(row[2]),
                    'last_accessed': row[3]
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting file memory: {e}")
            return None
    
    def clear_session(self, session_id: str = None):
        """Clear conversation history for a session"""
        try:
            if session_id:
                self.cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
            else:
                self.cursor.execute('DELETE FROM conversations')
            
            self.conversation_history = []
            self.conn.commit()
            self.logger.info(f"Cleared session: {session_id or 'all'}")
            
        except Exception as e:
            self.logger.error(f"Error clearing session: {e}")
    
    def get_current_session_id(self) -> str:
        """Generate a session ID for the current session"""
        return f"session_{int(time.time())}"
    
    def load_memory(self):
        """Load existing memory from storage"""
        try:
            # Load user preferences
            self.cursor.execute('SELECT key, value FROM user_preferences')
            for row in self.cursor.fetchall():
                self.user_preferences[row[0]] = json.loads(row[1])
            
            # Load recent conversation history
            self.conversation_history = self.get_conversation_history(limit=100)
            
            # Load context memory
            self.cursor.execute('SELECT key, value, importance, last_accessed FROM context_memory')
            for row in self.cursor.fetchall():
                self.context_memory[row[0]] = {
                    'value': json.loads(row[1]),
                    'importance': row[2],
                    'last_accessed': row[3]
                }
            
            self.logger.info("Memory loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save memory to storage"""
        try:
            self.conn.commit()
            self.logger.info("Memory saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            # Conversation stats
            self.cursor.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = self.cursor.fetchone()[0]
            
            # User preferences stats
            self.cursor.execute('SELECT COUNT(*) FROM user_preferences')
            total_preferences = self.cursor.fetchone()[0]
            
            # Context memory stats
            self.cursor.execute('SELECT COUNT(*) FROM context_memory')
            total_context = self.cursor.fetchone()[0]
            
            # File memory stats
            self.cursor.execute('SELECT COUNT(*) FROM file_memory')
            total_files = self.cursor.fetchone()[0]
            
            return {
                'total_conversations': total_conversations,
                'total_preferences': total_preferences,
                'total_context': total_context,
                'total_files': total_files,
                'vector_storage_available': self.chroma_client is not None,
                'embedding_model_available': self.embedding_model is not None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.conn.close()
            self.logger.info("Memory manager cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up memory manager: {e}")

# Global memory manager instance
memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    return memory_manager