#!/usr/bin/env python3
"""
SpeechBrainAgent - Wrapper for SpeechBrain speaker embeddings.

This agent provides real speaker embedding extraction using SpeechBrain's
ECAPA-TDNN models trained on VoxCeleb.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Any, Union
import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    log.warning("SpeechBrain not available. Install with: pip install speechbrain")


class SpeechBrainAgent:
    """
    Real speaker embedding extraction using SpeechBrain.
    
    This agent:
    - Extracts speaker embeddings using ECAPA-TDNN
    - Performs speaker verification
    - Calculates similarity scores
    """
    
    def __init__(self, 
                 model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 device: Optional[str] = None,
                 embedding_size: int = 192):
        """
        Initialize SpeechBrainAgent.
        
        Args:
            model_source: Hugging Face model source
            device: Device to run on ('cuda' or 'cpu')
            embedding_size: Size of embedding vectors
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError("SpeechBrain not installed")
        
        self.model_source = model_source
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_size = embedding_size
        self.classifier = None
        
        log.info(f"SpeechBrainAgent initialized on {self.device}")
    
    def load_model(self):
        """Load the speaker embedding model."""
        if self.classifier is None:
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_source,
                    run_opts={"device": self.device}
                )
                log.info(f"Loaded SpeechBrain model: {self.model_source}")
            except Exception as e:
                log.error(f"Failed to load model: {e}")
                raise
    
    def extract_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Embedding vector as numpy array
        """
        self.load_model()
        
        # Load audio
        signal, fs = torchaudio.load(str(audio_path))
        
        # Ensure single channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # Extract embedding
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(signal)
        
        # Convert to numpy
        embedding = embeddings.squeeze().cpu().numpy()
        
        return embedding
    
    def extract_embedding_from_array(self, audio_array: np.ndarray, 
                                    sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from audio array.
        
        Args:
            audio_array: Audio as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Embedding vector as numpy array
        """
        self.load_model()
        
        # Convert to tensor
        if len(audio_array.shape) == 1:
            audio_array = audio_array[np.newaxis, :]
        
        signal = torch.from_numpy(audio_array).float()
        
        # Extract embedding
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(signal)
        
        # Convert to numpy
        embedding = embeddings.squeeze().cpu().numpy()
        
        return embedding
    
    def verify_speakers(self, audio_path1: str, audio_path2: str,
                       threshold: float = 0.25) -> Tuple[bool, float]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio_path1: Path to first audio file
            audio_path2: Path to second audio file
            threshold: Similarity threshold
            
        Returns:
            Tuple of (same_speaker, similarity_score)
        """
        self.load_model()
        
        # Load audio files
        signal1, fs1 = torchaudio.load(audio_path1)
        signal2, fs2 = torchaudio.load(audio_path2)
        
        # Verify using the model
        score = self.classifier.verify_batch(signal1, signal2)
        
        # Convert score to numpy
        score_value = score.squeeze().item()
        
        # Check threshold
        same_speaker = score_value > threshold
        
        return same_speaker, score_value
    
    def calculate_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize embeddings
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(norm1, norm2)
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def batch_extract_embeddings(self, audio_paths: List[str]) -> List[np.ndarray]:
        """
        Extract embeddings for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of embedding vectors
        """
        self.load_model()
        
        embeddings = []
        for audio_path in audio_paths:
            try:
                embedding = self.extract_embedding(audio_path)
                embeddings.append(embedding)
            except Exception as e:
                log.error(f"Failed to extract embedding from {audio_path}: {e}")
                # Return zero embedding on failure
                embeddings.append(np.zeros(self.embedding_size))
        
        return embeddings
    
    def find_most_similar(self, query_embedding: np.ndarray,
                         reference_embeddings: List[np.ndarray],
                         threshold: float = 0.7) -> Tuple[int, float]:
        """
        Find most similar speaker from reference embeddings.
        
        Args:
            query_embedding: Query embedding
            reference_embeddings: List of reference embeddings
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (best_index, best_similarity)
            Returns (-1, 0.0) if no match above threshold
        """
        best_index = -1
        best_similarity = 0.0
        
        for i, ref_embedding in enumerate(reference_embeddings):
            similarity = self.calculate_similarity(query_embedding, ref_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
        
        # Check threshold
        if best_similarity < threshold:
            return -1, 0.0
        
        return best_index, best_similarity