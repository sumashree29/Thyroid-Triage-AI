"""
Agent 2: Retriever / RAG Agent
Handles vector database queries to retrieve clinical guidelines and evidence.
Implements semantic search and citation management.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class RetrievedDocument:
    """Retrieved guideline document with metadata."""
    id: str
    content: str
    source: str
    category: str
    severity: str
    relevance_score: float


class RetrieverAgent:
    """
    Retrieves relevant clinical guidelines and medical evidence from knowledge base.
    Uses TF-IDF for semantic similarity (production would use embeddings).
    """
    
    def __init__(self, kb_path: str = "docs/guidelines/knowledge_base.json"):
        """
        Initialize the retriever with medical knowledge base.
        
        Args:
            kb_path: Path to knowledge base JSON
        """
        self.kb_path = kb_path
        self.guidelines = {}
        self.vectorizer = None
        self.document_vectors = None
        self.document_list = []
        
        self._load_kb()
        self._build_index()
    
    def _load_kb(self):
        """Load knowledge base from file."""
        if Path(self.kb_path).exists():
            with open(self.kb_path, 'r') as f:
                self.guidelines = json.load(f)
            print(f"✓ Loaded {len(self.guidelines)} guidelines from knowledge base")
        else:
            print(f"⚠️ Knowledge base not found at {self.kb_path}")
            self._initialize_default_kb()
    
    def _initialize_default_kb(self):
        """Initialize with default guidelines if file doesn't exist."""
        # Import from knowledge_base module
        from docs.guidelines.knowledge_base import MEDICAL_GUIDELINES
        self.guidelines = MEDICAL_GUIDELINES
    
    def _build_index(self):
        """Build TF-IDF index for semantic search."""
        if not self.guidelines:
            print("⚠️ No guidelines loaded - skipping index build")
            return
        
        # Prepare documents
        self.document_list = []
        contents = []
        
        for key, data in self.guidelines.items():
            self.document_list.append({
                'id': key,
                'content': data['content'],
                'source': data['source'],
                'category': data['category'],
                'severity': data['severity']
            })
            contents.append(data['content'])
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = self.vectorizer.fit_transform(contents)
        print(f"✓ Built semantic index for {len(contents)} documents")
    
    def retrieve_by_query(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents by semantic similarity.
        
        Args:
            query: Clinical question or symptom
            top_k: Number of results to return
            
        Returns:
            List of RetrievedDocument objects
        """
        if not self.vectorizer or self.document_vectors is None:
            return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top K
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.0:  # Only include if some similarity
                    doc = self.document_list[idx]
                    results.append(RetrievedDocument(
                        id=doc['id'],
                        content=doc['content'],
                        source=doc['source'],
                        category=doc['category'],
                        severity=doc['severity'],
                        relevance_score=float(similarities[idx])
                    ))
            
            return results
            
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []
    
    def retrieve_by_risk_level(self, risk_level: str, top_k: int = 5) -> List[RetrievedDocument]:
        """
        Retrieve guidelines specific to risk level.
        
        Args:
            risk_level: 'high' or 'low'
            
        Returns:
            Relevant medical guidelines
        """
        # Map risk level to keywords
        keywords_map = {
            'high': 'urgent critical treatment hypothyroidism hyperthyroidism monitoring',
            'low': 'subclinical monitoring normal preventive lifestyle',
            'moderate': 'follow-up education TSH testing'
        }
        
        query = keywords_map.get(risk_level, 'thyroid TSH management')
        results = self.retrieve_by_query(query, top_k=top_k)
        
        # Also prioritize high-severity documents for high-risk cases
        if risk_level == 'high':
            results = sorted(results, key=lambda x: (x.severity != 'critical', -x.relevance_score))
        
        return results
    
    def retrieve_by_category(self, category: str) -> List[RetrievedDocument]:
        """
        Retrieve all documents in a specific category.
        
        Args:
            category: Category type (e.g., 'diagnostic', 'treatment', 'monitoring')
            
        Returns:
            Documents in category
        """
        results = []
        for doc in self.document_list:
            if doc['category'].lower() == category.lower():
                results.append(RetrievedDocument(
                    id=doc['id'],
                    content=doc['content'],
                    source=doc['source'],
                    category=doc['category'],
                    severity=doc['severity'],
                    relevance_score=1.0
                ))
        
        return results
    
    def retrieve_by_symptoms(self, symptoms: List[str]) -> List[RetrievedDocument]:
        """
        Retrieve documents matching patient symptoms.
        
        Args:
            symptoms: List of reported symptoms
            
        Returns:
            Relevant guidelines
        """
        query = ' '.join(symptoms)
        return self.retrieve_by_query(query, top_k=10)
    
    def format_citations(self, documents: List[RetrievedDocument]) -> str:
        """
        Format retrieved documents as citations.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted citation string
        """
        if not documents:
            return "No sources available"
        
        citations = []
        for doc in documents:
            citation = f"• {doc.source} ({doc.category}): {doc.content[:100]}..."
            citations.append(citation)
        
        return "\n".join(citations)
