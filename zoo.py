"""
FiftyOne integration for Qwen3-VL-Embedding model.

This module provides a FiftyOne zoo model wrapper for the Qwen3-VL-Embedding
model, enabling multimodal embeddings for videos, images, and text in a shared
vector space.

Capabilities:
    - Video/image embedding via compute_embeddings()
    - Text embedding for natural language queries
    - Text-video/image similarity search via sort_by_similarity()
    - Zero-shot classification via apply_model() with classes

Example - Compute embeddings:
    model = load_model("Qwen/Qwen3-VL-Embedding-8B")
    dataset.compute_embeddings(model, embeddings_field="embeddings")

Example - Similarity search:
    query = model.embed_prompt("A person cooking in a kitchen")
    view = dataset.sort_by_similarity(query, k=10, brain_key="embeddings")

Example - Zero-shot classification:
    model.classes = ["sports", "cooking", "travel"]
    dataset.apply_model(model, label_field="predictions")
"""

import logging

import numpy as np
import torch
from transformers.utils.import_utils import is_flash_attn_2_available

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem, ClassifierOutputProcessor
import os
import tempfile
from io import BytesIO
from pathlib import Path

from .qwen3_vl_embedding import Qwen3VLEmbedder

logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device for inference.
    
    Checks for available devices in priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _mkstemp_path(self, suffix: str) -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p


def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
    a = arr
    if a.dtype == np.uint8:
        return a

    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).round().astype(np.uint8)
        return a

    return np.clip(a, 0, 255).astype(np.uint8)


def _write_png(self, arr: np.ndarray, path: str) -> None:
    a = self._normalize_to_uint8(arr)

    try:
        from PIL import Image

        if a.ndim == 2:
            Image.fromarray(a, mode="L").save(path, format="PNG")
            return

        if a.ndim == 3 and a.shape[2] == 1:
            Image.fromarray(a[:, :, 0], mode="L").save(path, format="PNG")
            return

        if a.ndim == 3 and a.shape[2] in (3, 4):
            Image.fromarray(a[:, :, :3], mode="RGB").save(path, format="PNG")
            return

        raise ValueError(f"Unsupported image array shape: {a.shape}")

    except Exception:
        import cv2

        if a.ndim == 2:
            ok = cv2.imwrite(path, a)
            if not ok:
                raise ValueError("Failed to write image to disk")
            return

        if a.ndim == 3 and a.shape[2] == 1:
            ok = cv2.imwrite(path, a[:, :, 0])
            if not ok:
                raise ValueError("Failed to write image to disk")
            return

        if a.ndim == 3 and a.shape[2] in (3, 4):
            rgb = a[:, :, :3]
            bgr = rgb[:, :, ::-1]
            ok = cv2.imwrite(path, bgr)
            if not ok:
                raise ValueError("Failed to write image to disk")
            return

        raise ValueError(f"Unsupported image array shape: {a.shape}")


def _extract_media_path_and_type(self, media):
    cleanup = None

    if isinstance(media, dict):
        p = media.get("filepath")
        if not p:
            raise ValueError("Expected dict with 'filepath'")
        return p, cleanup, self.config.media_type

    if isinstance(media, (str, os.PathLike)):
        return str(media), cleanup, self.config.media_type

    if hasattr(media, "inpath"):
        return media.inpath, cleanup, self.config.media_type

    if isinstance(media, np.ndarray):
        p = self._mkstemp_path(".png")
        self._write_png(media, p)
        return p, p, "image"

    if isinstance(media, (bytes, bytearray, memoryview)):
        b = bytes(media)
        try:
            from PIL import Image
            img = Image.open(BytesIO(b)).convert("RGB")
            p = self._mkstemp_path(".png")
            img.save(p, format="PNG")
            return p, p, "image"
        except Exception:
            raise TypeError("Byte input is not a valid image; video bytes not supported yet")

    try:
        from PIL.Image import Image as PILImageType
        if isinstance(media, PILImageType):
            p = self._mkstemp_path(".png")
            media.convert("RGB").save(p, format="PNG")
            return p, p, "image"
    except Exception:
        pass

    raise TypeError(f"Unsupported media input type: {type(media)}")

class Qwen3VLEmbeddingGetItem(GetItem):
    """GetItem transform for Qwen3-VL embedding model.
    
    Lightweight transform that extracts filepath from samples for batch
    processing. Media loading happens in the main process via Qwen3VLEmbedder.
    """
    
    @property
    def required_keys(self):
        """Fields needed from each sample (STATIC).
        
        Returns:
            list: ["filepath"] - always the same, no runtime dependencies
        """
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Extract filepath for embedding.
        
        Args:
            sample_dict: Dict with "filepath" key from FiftyOne sample
        
        Returns:
            dict: {"filepath": str} ready for embed_all() processing
        """
        return {"filepath": sample_dict["filepath"]}


class Qwen3VLEmbeddingModelConfig(fout.TorchImageModelConfig):
    """Configuration for Qwen3-VL-Embedding model.
    
    Attributes:
        model_path: HuggingFace model identifier
        media_type: "video" or "image" - controls FiftyOne pipeline
        fps: Frame sampling rate for videos
        max_frames: Maximum frames to sample from video
        num_frames: Target number of frames to extract
        min_pixels: Minimum pixels for processing
        max_pixels: Maximum pixels for processing
        total_pixels: Total pixel budget for video frames
        text_prompt: Prefix for class label prompts in zero-shot classification
    """
    
    def __init__(self, d):
        """Initialize configuration from dictionary.
        
        Args:
            d: Dictionary containing configuration parameters
        """
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        
        if "classes" in d and d["classes"] is not None and len(d["classes"]) > 0:
            if "output_processor_cls" not in d:
                d["output_processor_cls"] = "fiftyone.utils.torch.ClassifierOutputProcessor"
        
        super().__init__(d)
        
        # Model path
        self.model_path = self.parse_string(
            d, "model_path", default="Qwen/Qwen3-VL-Embedding-8B"
        )
        
        # Media type for FiftyOne pipeline
        self.media_type = self.parse_string(d, "media_type", default="video")
        if self.media_type not in ("video", "image"):
            raise ValueError(f"media_type must be 'video' or 'image', got '{self.media_type}'")
        
        # Video processing parameters
        self.fps = self.parse_number(d, "fps", default=1.0)
        self.max_frames = self.parse_number(d, "max_frames", default=64)
        self.num_frames = self.parse_number(d, "num_frames", default=64)
        
        # Pixel settings
        self.min_pixels = self.parse_number(d, "min_pixels", default=4 * 32 * 32)
        self.max_pixels = self.parse_number(d, "max_pixels", default=1800 * 32 * 32)
        self.total_pixels = self.parse_number(d, "total_pixels", default=10 * 768 * 32 * 32)
        
        # Tokenizer settings
        self.max_length = self.parse_number(d, "max_length", default=8192)
        
        # Classification settings
        self.text_prompt = self.parse_string(d, "text_prompt", default="")


class Qwen3VLEmbeddingModel(fom.Model, fom.PromptMixin, SupportsGetItem, TorchModelMixin):
    """FiftyOne wrapper for Qwen3-VL-Embedding model.
    
    Provides multimodal embeddings for videos, images, and text in a shared
    vector space. Supports similarity search, retrieval, and zero-shot
    classification.
    """
    
    def __init__(self, config):
        """Initialize the embedding model.
        
        Args:
            config: Qwen3VLEmbeddingModelConfig instance
        """
        SupportsGetItem.__init__(self)
        
        self.config = config
        self._preprocess = False
        self.device = get_device()
        
        # Lazy loading
        self._embedder = None
        
        # Caches
        self._last_computed_embeddings = None
        self._text_features = None
        
        # Classification
        self._classes = None
        self._output_processor = None
        
        logger.info(f"Initialized Qwen3VLEmbeddingModel (device: {self.device})")
    
    # =========================================================================
    # Required properties from Model base class
    # =========================================================================
    
    @property
    def media_type(self):
        """Media type this model operates on."""
        return self.config.media_type
    
    @property
    def transforms(self):
        """Preprocessing transforms (None - embedder handles it)."""
        return None
    
    @property
    def preprocess(self):
        """Whether model should apply preprocessing."""
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Allow FiftyOne to control preprocessing."""
        self._preprocess = value
    
    @property
    def ragged_batches(self):
        """Must be False to enable batching."""
        return False
    
    # =========================================================================
    # Embedding properties
    # =========================================================================
    
    @property
    def has_embeddings(self):
        """Whether this model can generate embeddings."""
        return True
    
    @property
    def can_embed_prompts(self):
        """Whether this model can embed text prompts."""
        return True
    
    # =========================================================================
    # Classification properties
    # =========================================================================
    
    @property
    def classes(self):
        """List of class labels for zero-shot classification."""
        return self._classes
    
    @classes.setter
    def classes(self, value):
        """Set class labels and invalidate cached text features."""
        self._classes = value
        self._text_features = None
        
        if value is not None and len(value) > 0:
            self._output_processor = ClassifierOutputProcessor(classes=value)
        else:
            self._output_processor = None
    
    @property
    def text_prompt(self):
        """Text prompt prefix for classification."""
        return self.config.text_prompt
    
    @text_prompt.setter
    def text_prompt(self, value):
        """Set text prompt and invalidate cached text features."""
        self.config.text_prompt = value
        self._text_features = None
    
    # =========================================================================
    # Configurable properties (can be changed without reloading model)
    # =========================================================================
    
    @property
    def fps(self):
        """Frame sampling rate for videos."""
        return self.config.fps
    
    @fps.setter
    def fps(self, value):
        """Set frame sampling rate."""
        self.config.fps = value
    
    @property
    def max_frames(self):
        """Maximum frames to sample from video."""
        return self.config.max_frames
    
    @max_frames.setter
    def max_frames(self, value):
        """Set max frames."""
        self.config.max_frames = value
    
    @property
    def num_frames(self):
        """Target number of frames to extract."""
        return self.config.num_frames
    
    @num_frames.setter
    def num_frames(self, value):
        """Set target number of frames."""
        self.config.num_frames = value
    
    @property
    def min_pixels(self):
        """Minimum pixels for processing."""
        return self.config.min_pixels
    
    @min_pixels.setter
    def min_pixels(self, value):
        """Set minimum pixels."""
        self.config.min_pixels = value
    
    @property
    def max_pixels(self):
        """Maximum pixels for processing."""
        return self.config.max_pixels
    
    @max_pixels.setter
    def max_pixels(self, value):
        """Set maximum pixels."""
        self.config.max_pixels = value
    
    @property
    def total_pixels(self):
        """Total pixel budget for video frames."""
        return self.config.total_pixels
    
    @total_pixels.setter
    def total_pixels(self, value):
        """Set total pixel budget."""
        self.config.total_pixels = value
    
    @property
    def max_length(self):
        """Maximum token sequence length for the tokenizer."""
        return self.config.max_length
    
    @max_length.setter
    def max_length(self, value):
        """Set max token length.
        
        If the embedder is already loaded, updates it directly so the change
        takes effect immediately without reloading the model.
        """
        self.config.max_length = value
        if self._embedder is not None:
            self._embedder.max_length = value
    
    @media_type.setter
    def media_type(self, value):
        """Set media type (video or image).
        
        Args:
            value: "video" or "image"
        
        Raises:
            ValueError: If value is not "video" or "image"
        """
        if value not in ("video", "image"):
            raise ValueError(f"media_type must be 'video' or 'image', got '{value}'")
        self.config.media_type = value
    
    # =========================================================================
    # TorchModelMixin properties
    # =========================================================================
    
    @property
    def has_collate_fn(self):
        """Whether this model has custom collation."""
        return True
    
    @property
    def collate_fn(self):
        """Custom collate function - returns batch as-is."""
        def identity_collate(batch):
            return batch
        return identity_collate
    
    # =========================================================================
    # SupportsGetItem methods
    # =========================================================================
    
    def build_get_item(self, field_mapping=None):
        """Build GetItem transform for data loading."""
        return Qwen3VLEmbeddingGetItem(field_mapping=field_mapping)
    
    # =========================================================================
    # Model loading
    # =========================================================================
    
    def _load_model(self):
        """Load Qwen3VLEmbedder."""
        logger.info(f"Loading Qwen3-VL-Embedding from {self.config.model_path}")
        
        embedder_kwargs = {
            "max_length": self.config.max_length,
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
            "total_pixels": self.config.total_pixels,
            "fps": self.config.fps,
            "num_frames": self.config.num_frames,
            "max_frames": self.config.max_frames,
            "torch_dtype": "auto",
        }
        
        if self.device == "cuda" and is_flash_attn_2_available():
            embedder_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        self._embedder = Qwen3VLEmbedder(
            model_name_or_path=self.config.model_path,
            **embedder_kwargs
        )
        
        logger.info("Model loaded successfully")
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _extract_media_path(self, media):
        """Extract filepath from media input.
        
        Args:
            media: Dict from GetItem, string path, or video reader object
        
        Returns:
            str: Filepath to media file
        """
        if isinstance(media, dict):
            return media["filepath"]
        if isinstance(media, str):
            return media
        # FFmpegVideoReader stores path in inpath
        return media.inpath

    def _prepare_embedder_input(self, media_type, filepath):
        """Prepare input dict for Qwen3VLEmbedder.process().
        
        Uses config.media_type to determine input format.
        
        Args:
            filepath: Path to media file
        
        Returns:
            dict: Input dict for embedder.process()
        """
        if media_type == "video":
            return {
                "video": filepath,
                "fps": self.config.fps,
                "max_frames": self.config.max_frames,
            }

        return {"image": filepath}
    
    # =========================================================================
    # Video/Image embedding methods
    # =========================================================================

    def embed(self, media):
        if self._embedder is None:
            self._load_model()

        filepath, cleanup, media_type = self._extract_media_path_and_type(media)
        input_dict = self._prepare_embedder_input(media_type, filepath)

        try:
            with torch.no_grad():
                embedding = self._embedder.process([input_dict])
                result = embedding[0].cpu().float().numpy()
        finally:
            if cleanup is not None:
                try:
                    os.remove(cleanup)
                except Exception:
                    pass

        self._last_computed_embeddings = result.reshape(1, -1)
        return result

    def embed_all(self, medias):
        if self._embedder is None:
            self._load_model()

        embeddings = []
        cleanups = []

        try:
            with torch.no_grad():
                for media in medias:
                    filepath, cleanup, media_type = self._extract_media_path_and_type(media)
                    if cleanup is not None:
                        cleanups.append(cleanup)

                    input_dict = self._prepare_embedder_input(media_type, filepath)
                    embedding = self._embedder.process([input_dict])
                    embeddings.append(embedding[0].cpu().float().numpy())
        finally:
            for p in cleanups:
                try:
                    os.remove(p)
                except Exception:
                    pass

        result = np.stack(embeddings, axis=0)
        self._last_computed_embeddings = result
        return result
    
    def get_embeddings(self):
        """Get the last computed embeddings.
        
        Returns:
            numpy.ndarray: Last computed embeddings
        """
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
        return self._last_computed_embeddings
    
    # =========================================================================
    # Text embedding methods (PromptMixin)
    # =========================================================================
    
    def embed_prompt(self, prompt):
        """Embed a single text prompt.
        
        Args:
            prompt: Text string to embed
        
        Returns:
            numpy.ndarray: 1D embedding vector
        """
        if self._embedder is None:
            self._load_model()
        
        with torch.no_grad():
            embedding = self._embedder.process([{"text": prompt}])
        return embedding[0].cpu().float().numpy()
    
    def embed_prompts(self, prompts):
        """Embed multiple text prompts.
        
        Args:
            prompts: List of text strings to embed
        
        Returns:
            numpy.ndarray: 2D embeddings with shape (batch_size, hidden_dim)
        """
        if self._embedder is None:
            self._load_model()
        
        with torch.no_grad():
            inputs = [{"text": p} for p in prompts]
            embeddings = self._embedder.process(inputs)
        return embeddings.cpu().float().numpy()
    
    # =========================================================================
    # Zero-shot classification methods
    # =========================================================================
    
    def _get_text_features(self):
        """Get or compute cached text embeddings for class labels.
        
        Returns:
            numpy.ndarray: Text embeddings for all classes
        """
        if self._text_features is None:
            prompts = [f"{self.text_prompt} {c}".strip() for c in self.classes]
            self._text_features = self.embed_prompts(prompts)
        return self._text_features
    
    def predict(self, media):
        """Predict on a single media item (required by Model base class).
        
        Args:
            media: Video/image to classify
        
        Returns:
            Classification label
        """
        results = self.predict_all([media])
        return results[0]
    
    def predict_all(self, medias):
        """Predict on a batch of media items.
        
        Args:
            medias: List of videos/images to classify
        
        Returns:
            List of Classification labels
        """
        if self.classes is None or len(self.classes) == 0:
            raise ValueError(
                "Cannot perform classification without classes. "
                "Set model.classes = ['class1', 'class2', ...]"
            )
        
        media_embeddings = self.embed_all(medias)
        text_embeddings = self._get_text_features()
        
        # Cosine similarity (embeddings are L2-normalized)
        logits = media_embeddings @ text_embeddings.T
        
        # Convert to torch tensor for output processor
        output = torch.from_numpy(logits)
        
        # Get frame size (not used for classification, but required by processor)
        frame_size = (1, 1)
        
        return self._output_processor(
            output,
            frame_size,
            confidence_thresh=self.config.confidence_thresh,
        )
    
    # =========================================================================
    # Context manager
    # =========================================================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit - clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False
