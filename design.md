# Design Document: TruthLens Enhancements

## Overview

This design document specifies enhancements to the TruthLens AI Content Risk Analyzer system. The enhancements maintain TruthLens's core philosophy of honest uncertainty and transparent analysis while significantly improving detection accuracy, performance, and user experience.

The design focuses on five key areas:
1. **Enhanced Detection Algorithms**: Improved frequency analysis, CLIP integration, and advanced metadata checking
2. **Sophisticated Text Analysis**: Better AI text detection with language support and perplexity modeling
3. **Intelligent Confidence Scoring**: Adaptive thresholds and nuanced confidence calculations
4. **Performance & Scalability**: Caching, batch processing, and resource management
5. **User Experience**: Enhanced UI, snip mode, and comprehensive explainability

### Design Philosophy

All enhancements adhere to TruthLens's founding principles:
- **Honesty over hype**: Default to UNCERTAIN when appropriate
- **Transparency over black boxes**: Show all signals and reasoning
- **Education over false confidence**: Help users understand detection limitations

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser Extension (MV3)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Popup UI   │  │  Content.js  │  │ Background   │     │
│  │  (Enhanced)  │  │  (Snip Mode) │  │   Worker     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS/JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Layer (main.py)                      │  │
│  │  Rate Limiting │ Caching │ Request Queue │ Metrics   │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Analysis Orchestration Layer                │  │
│  │  Batch Processing │ Async Execution │ Error Recovery │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────┬──────────────────┬─────────────────┐  │
│  │ Image Analyzer  │  Text Analyzer   │  Page Analyzer  │  │
│  │  (Enhanced)     │   (Enhanced)     │   (Enhanced)    │  │
│  └─────────────────┴──────────────────┴─────────────────┘  │
│           │                 │                  │            │
│  ┌─────────────────┬──────────────────┬─────────────────┐  │
│  │ Signal Modules  │  Signal Modules  │  Fusion Engine  │  │
│  │ • FFT/DCT       │  • Perplexity    │  • Adaptive     │  │
│  │ • CLIP API      │  • Burstiness    │    Thresholds   │  │
│  │ • C2PA/EXIF     │  • Language Det  │  • Confidence   │  │
│  │ • Metadata      │  • Pattern Match │    Calculation  │  │
│  └─────────────────┴──────────────────┴─────────────────┘  │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Infrastructure Layer                     │  │
│  │  Redis Cache │ Logging │ Monitoring │ Config Manager │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  External APIs   │
                  │  • CLIP Service  │
                  │  • Language Det  │
                  └──────────────────┘
```

### Key Architectural Changes

1. **Caching Layer**: Redis-based content-addressed cache for analysis results
2. **Batch Processing**: Support for concurrent analysis of multiple items
3. **Adaptive Fusion**: Dynamic threshold adjustment based on content type
4. **External API Integration**: CLIP embeddings and language detection services
5. **Enhanced Error Recovery**: Partial results when individual signals fail

## Components and Interfaces

### 1. Enhanced Image Analyzer

#### FrequencyAnalyzer Component

**Purpose**: Improved frequency domain analysis using both FFT and DCT

**Interface**:
```python
class FrequencyAnalyzer:
    def analyze_fft(self, image: np.ndarray) -> FrequencyResult:
        """Analyze using Fast Fourier Transform"""
        pass
    
    def analyze_dct(self, image: np.ndarray) -> FrequencyResult:
        """Analyze using Discrete Cosine Transform"""
        pass
    
    def detect_periodicity(self, spectrum: np.ndarray) -> PeriodicityMetrics:
        """Detect and quantify periodic patterns"""
        pass
    
    def adaptive_threshold(self, image_resolution: Tuple[int, int]) -> ThresholdConfig:
        """Calculate resolution-adaptive thresholds"""
        pass
    
    def compensate_compression(self, image: Image, quality_estimate: float) -> np.ndarray:
        """Adjust analysis for JPEG compression artifacts"""
        pass
```

**Key Methods**:
- `analyze_fft()`: Performs FFT analysis, returns high/mid/low frequency energy ratios
- `analyze_dct()`: Performs DCT analysis, detects block artifacts and quantization patterns
- `detect_periodicity()`: Uses autocorrelation to find periodic patterns from GAN upsampling
- `adaptive_threshold()`: Adjusts detection thresholds based on image resolution
- `compensate_compression()`: Estimates JPEG quality and adjusts frequency expectations

**Algorithm Details**:
```
1. Convert image to grayscale
2. Apply FFT and DCT transforms
3. Divide frequency spectrum into bands (low: 0-10%, mid: 10-30%, high: 30%+)
4. Calculate energy distribution across bands
5. Detect periodic peaks using autocorrelation
6. Compare to expected distributions for natural vs AI images
7. Adjust scores based on resolution and compression
8. Return combined FFT+DCT score with artifact types
```

#### CLIPSemanticAnalyzer Component

**Purpose**: Semantic analysis using CLIP embeddings

**Interface**:
```python
class CLIPSemanticAnalyzer:
    def __init__(self, api_endpoint: str, cache: CacheManager):
        pass
    
    def get_embedding(self, image: Image) -> np.ndarray:
        """Get CLIP embedding for image"""
        pass
    
    def calculate_drift(self, embedding: np.ndarray) -> float:
        """Calculate distance from natural image manifold"""
        pass
    
    def fallback_heuristic(self, image: Image) -> float:
        """Fallback to heuristic analysis if API unavailable"""
        pass
```

**Key Methods**:
- `get_embedding()`: Calls CLIP API (with caching), returns 512-dim embedding vector
- `calculate_drift()`: Computes cosine distance to reference distribution centroid
- `fallback_heuristic()`: Uses existing color/gradient analysis when API fails

**Reference Distribution**:
- Pre-computed from 10,000 natural images (COCO dataset)
- Stored as mean vector and covariance matrix
- Mahalanobis distance used for drift calculation

#### ProvenanceAnalyzer Component

**Purpose**: Advanced metadata and provenance verification

**Interface**:
```python
class ProvenanceAnalyzer:
    def extract_exif(self, image_data: bytes) -> Dict[str, Any]:
        """Extract EXIF metadata"""
        pass
    
    def validate_exif_consistency(self, exif: Dict) -> ConsistencyReport:
        """Check for metadata inconsistencies"""
        pass
    
    def verify_c2pa(self, image_data: bytes) -> C2PAResult:
        """Verify C2PA content credentials"""
        pass
    
    def detect_tampering(self, exif: Dict) -> TamperingIndicators:
        """Detect metadata tampering"""
        pass
    
    def classify_editing(self, software_tags: List[str]) -> EditingClassification:
        """Distinguish minor edits from AI generation"""
        pass
```

**Key Methods**:
- `validate_exif_consistency()`: Checks timestamp sequences, camera model compatibility
- `verify_c2pa()`: Validates cryptographic signatures in C2PA manifests
- `detect_tampering()`: Looks for impossible metadata combinations
- `classify_editing()`: Categorizes as: NONE, MINOR_EDIT, MAJOR_EDIT, AI_GENERATED

**Consistency Checks**:
- DateTime vs DateTimeOriginal vs DateTimeDigitized sequence
- Camera Make/Model compatibility with lens information
- GPS coordinates vs timezone in DateTime
- Software version vs camera model release date

### 2. Enhanced Text Analyzer

#### PerplexityCalculator Component

**Purpose**: Calculate text perplexity using n-gram models

**Interface**:
```python
class PerplexityCalculator:
    def __init__(self, ngram_size: int = 3):
        pass
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity score"""
        pass
    
    def build_ngram_model(self, corpus: List[str]) -> NGramModel:
        """Build n-gram language model"""
        pass
    
    def get_word_probabilities(self, text: str) -> List[float]:
        """Get probability for each word"""
        pass
```

**Algorithm**:
```
1. Tokenize text into words
2. For each word, calculate P(word | previous n-1 words)
3. Use pre-trained n-gram model (built from human text corpus)
4. Perplexity = exp(-1/N * sum(log P(word_i)))
5. Lower perplexity = more predictable = more likely AI
```

#### BurstinessAnalyzer Component

**Purpose**: Analyze sentence and paragraph structure variation

**Interface**:
```python
class BurstinessAnalyzer:
    def calculate_sentence_burstiness(self, text: str) -> float:
        """Calculate sentence length variation"""
        pass
    
    def calculate_paragraph_burstiness(self, text: str) -> float:
        """Calculate paragraph structure variation"""
        pass
    
    def analyze_structure_patterns(self, text: str) -> StructureMetrics:
        """Analyze overall structural patterns"""
        pass
```

**Metrics**:
- Sentence burstiness: `std_dev(sentence_lengths) / mean(sentence_lengths)`
- Paragraph burstiness: `std_dev(paragraph_lengths) / mean(paragraph_lengths)`
- Transition uniformity: Frequency of transition phrases per paragraph
- Structure entropy: Shannon entropy of sentence length distribution

#### LanguageDetector Component

**Purpose**: Detect language and apply language-specific analysis

**Interface**:
```python
class LanguageDetector:
    def detect_language(self, text: str) -> LanguageResult:
        """Detect text language"""
        pass
    
    def get_language_config(self, language: str) -> LanguageConfig:
        """Get language-specific analysis configuration"""
        pass
    
    def apply_language_patterns(self, text: str, language: str) -> PatternMatches:
        """Apply language-specific AI detection patterns"""
        pass
```

**Supported Languages**:
- English: Full support with AI phrase detection
- Spanish, French, German, Portuguese: Burstiness + perplexity
- Chinese, Japanese, Korean: Character-level analysis
- Arabic, Russian: Script-specific patterns
- Others: Language-agnostic metrics only

### 3. Adaptive Fusion Engine

#### ThresholdManager Component

**Purpose**: Manage adaptive thresholds based on content type

**Interface**:
```python
class ThresholdManager:
    def get_thresholds(self, content_type: ContentType) -> ThresholdConfig:
        """Get thresholds for content type"""
        pass
    
    def classify_content_type(self, metadata: Dict) -> ContentType:
        """Classify content type from metadata"""
        pass
    
    def load_config(self, config_path: str) -> None:
        """Load threshold configuration"""
        pass
```

**Content Types**:
- PHOTOGRAPH: Strict thresholds (0.75 AI, 0.35 REAL)
- DIGITAL_ART: Relaxed thresholds (0.80 AI, 0.30 REAL)
- SCREENSHOT: Very relaxed (0.85 AI, 0.25 REAL)
- WEB_GRAPHIC: Relaxed (0.80 AI, 0.30 REAL)
- UNKNOWN: Conservative defaults (0.75 AI, 0.35 REAL)

**Classification Logic**:
```
IF has_camera_exif AND has_gps:
    return PHOTOGRAPH
ELIF file_format == PNG AND no_exif:
    return SCREENSHOT or WEB_GRAPHIC
ELIF has_editing_software:
    return DIGITAL_ART
ELSE:
    return UNKNOWN
```

#### ConfidenceCalculator Component

**Purpose**: Calculate nuanced confidence scores

**Interface**:
```python
class ConfidenceCalculator:
    def calculate_overall_confidence(self, signals: Dict[str, float]) -> float:
        """Calculate overall confidence from signal variance"""
        pass
    
    def calculate_per_signal_confidence(self, signal_name: str, input_quality: Dict) -> float:
        """Calculate confidence for individual signal"""
        pass
    
    def get_confidence_breakdown(self, signals: Dict, confidences: Dict) -> ConfidenceBreakdown:
        """Get detailed confidence breakdown"""
        pass
    
    def enforce_minimum_confidence(self, score: float, confidence: float) -> Tuple[str, float]:
        """Enforce UNCERTAIN verdict for low confidence"""
        pass
```

**Confidence Calculation**:
```
1. Calculate signal variance: var = variance(signal_values)
2. Base confidence: conf_base = 1.0 - min(var * 2, 0.7)
3. Per-signal adjustments:
   - Frequency analysis: reduce if image compressed
   - Semantic analysis: reduce if API unavailable
   - Text analysis: reduce if text < 100 words
4. Overall confidence: weighted average of per-signal confidences
5. If confidence < 0.4: force verdict = UNCERTAIN
```

### 4. Caching and Performance Layer

#### CacheManager Component

**Purpose**: Content-addressed caching with Redis

**Interface**:
```python
class CacheManager:
    def __init__(self, redis_client: Redis, ttl: int = 86400):
        pass
    
    def get_cached_result(self, content_hash: str) -> Optional[AnalysisResult]:
        """Retrieve cached analysis result"""
        pass
    
    def cache_result(self, content_hash: str, result: AnalysisResult) -> None:
        """Cache analysis result"""
        pass
    
    def compute_content_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content"""
        pass
    
    def invalidate_cache(self, pattern: str = "*") -> int:
        """Invalidate cache entries"""
        pass
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache hit/miss statistics"""
        pass
```

**Caching Strategy**:
- Key: SHA-256 hash of image/text content
- Value: JSON-serialized AnalysisResult
- TTL: 24 hours (configurable)
- Eviction: LRU when memory limit reached

#### BatchProcessor Component

**Purpose**: Process multiple analyses concurrently

**Interface**:
```python
class BatchProcessor:
    def __init__(self, max_concurrent: int = 5):
        pass
    
    async def process_batch(self, items: List[AnalysisRequest]) -> List[AnalysisResult]:
        """Process batch of analysis requests"""
        pass
    
    async def process_with_cache(self, items: List[AnalysisRequest], cache: CacheManager) -> List[AnalysisResult]:
        """Process batch with cache lookup"""
        pass
```

**Batch Processing Flow**:
```
1. Receive batch of N items
2. Compute content hashes for all items
3. Check cache for each hash (parallel)
4. For cache misses:
   a. Create analysis tasks
   b. Execute up to max_concurrent tasks in parallel
   c. Collect results as they complete
5. Cache new results
6. Return all results in original order
```

### 5. Enhanced Browser Extension

#### SnipModeController Component

**Purpose**: Implement region selection for targeted analysis

**Interface**:
```javascript
class SnipModeController {
    constructor(contentDocument) {}
    
    activate() {
        // Show overlay and enable selection
    }
    
    deactivate() {
        // Hide overlay and cleanup
    }
    
    onRegionSelected(rect) {
        // Capture and analyze selected region
    }
    
    captureRegion(rect) {
        // Capture content within rectangle
    }
}
```

**Implementation**:
- Overlay: Semi-transparent div covering entire page
- Selection: Mouse drag creates resizable rectangle
- Capture: Extract DOM elements and images within bounds
- Analysis: Send captured content to backend

#### ResultsPanel Component

**Purpose**: Enhanced UI for displaying analysis results

**Interface**:
```javascript
class ResultsPanel {
    constructor(container) {}
    
    displayResult(analysisResult) {
        // Show verdict, confidence, signals
    }
    
    renderSignalBreakdown(signals) {
        // Visual breakdown of signal contributions
    }
    
    renderConfidenceExplanation(confidence, breakdown) {
        // Explain confidence score
    }
    
    renderTechnicalDetails(details) {
        // Expandable technical details section
    }
    
    enableCopyToClipboard(result) {
        // Copy JSON result to clipboard
    }
}
```

**UI Enhancements**:
- Signal contribution bar chart
- Confidence explanation with tooltips
- Expandable technical details
- Copy-to-clipboard button
- Historical results for current session

## Data Models

### Enhanced Analysis Result

```python
@dataclass
class AnalysisResult:
    verdict: str  # AI_LIKELY, UNCERTAIN, REAL_LIKELY
    ai_probability: int  # 0-100
    confidence: float  # 0.0-1.0
    confidence_breakdown: ConfidenceBreakdown
    signals: Dict[str, SignalResult]
    explanations: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: int
    cache_hit: bool

@dataclass
class SignalResult:
    score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    details: Dict[str, Any]
    explanation: str

@dataclass
class ConfidenceBreakdown:
    overall: float
    per_signal: Dict[str, float]
    variance: float
    uncertainty_sources: List[str]
```

### Frequency Analysis Result

```python
@dataclass
class FrequencyResult:
    fft_score: float
    dct_score: float
    combined_score: float
    high_freq_ratio: float
    mid_freq_ratio: float
    low_freq_ratio: float
    periodicity_strength: float
    artifact_types: List[str]  # e.g., ["GAN_UPSAMPLING", "DIFFUSION_SMOOTHING"]
    compression_adjusted: bool
```

### Provenance Analysis Result

```python
@dataclass
class ProvenanceResult:
    score: float
    has_exif: bool
    exif_consistent: bool
    c2pa_verified: Optional[bool]
    tampering_detected: bool
    editing_classification: str  # NONE, MINOR_EDIT, MAJOR_EDIT, AI_GENERATED
    camera_metadata: Optional[CameraMetadata]
    inconsistencies: List[str]
```

### Text Analysis Result

```python
@dataclass
class TextAnalysisResult:
    ai_score: float
    perplexity: float
    sentence_burstiness: float
    paragraph_burstiness: float
    language: str
    language_confidence: float
    ai_phrases_count: int
    phishing_score: float
    misinformation_score: float
    word_count: int
```

### Cache Entry

```python
@dataclass
class CacheEntry:
    content_hash: str
    result: AnalysisResult
    timestamp: datetime
    hit_count: int
    ttl_seconds: int
```

### Threshold Configuration

```python
@dataclass
class ThresholdConfig:
    ai_likely_threshold: float  # e.g., 0.75
    real_likely_threshold: float  # e.g., 0.35
    min_confidence_threshold: float  # e.g., 0.4
    content_type: str
    signal_weights: Dict[str, float]
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all acceptance criteria, I identified several areas where properties can be consolidated:

**Redundancy Analysis:**
- Properties 1.1-1.5 (frequency analysis) can be partially consolidated - they all test that frequency analysis produces expected outputs
- Properties 2.2 and 2.3 both test CLIP embedding comparison - can be combined
- Properties 3.1, 3.3, 3.4 all test metadata validation - can be consolidated into comprehensive validation property
- Properties 4.1, 4.2, 4.3 all test text analysis calculations - can verify all metrics are calculated together
- Properties 6.1, 6.2, 6.3, 6.5 all test threshold adaptation - can be combined into single adaptive threshold property
- Properties 8.1, 8.2, 8.4 are all UI element existence tests - can be verified together
- Properties 14.1-14.5 are all infrastructure tests - can be verified as examples rather than properties

**Consolidation Strategy:**
- Keep properties that test universal behaviors across all inputs
- Combine properties that test multiple aspects of the same feature
- Convert infrastructure/configuration tests to examples rather than properties
- Maintain separate properties for distinct correctness concerns

### Image Analysis Properties

**Property 1: Dual Transform Frequency Analysis**
*For any* image, frequency analysis should produce both FFT and DCT scores, with periodicity strength quantified when periodic patterns are detected, and scores should be adjusted when JPEG compression is detected.
**Validates: Requirements 1.1, 1.2, 1.5**

**Property 2: Resolution-Adaptive Thresholds**
*For any* image, the frequency analysis thresholds should vary based on image resolution, with higher resolution images using different threshold values than lower resolution images.
**Validates: Requirements 1.3**

**Property 3: Artifact Type Classification**
*For any* image where frequency artifacts are detected (score > 0.6), the result should include specific artifact type classifications (e.g., "GAN_UPSAMPLING", "DIFFUSION_SMOOTHING").
**Validates: Requirements 1.4**

**Property 4: CLIP Embedding Comparison**
*For any* image analyzed with CLIP available, the semantic analysis should compare the image embedding to the reference distribution and quantify the drift distance from the natural image manifold.
**Validates: Requirements 2.2, 2.3**

**Property 5: Embedding Cache Round-Trip**
*For any* image, analyzing it twice within the cache TTL period should return cached results on the second analysis, with the cache_hit flag set to true.
**Validates: Requirements 2.5**

**Property 6: Metadata Consistency Validation**
*For any* image with EXIF data, the provenance analyzer should validate consistency of camera metadata fields, detect timestamp sequence inconsistencies, and classify editing software appropriately (NONE, MINOR_EDIT, MAJOR_EDIT, AI_GENERATED).
**Validates: Requirements 3.1, 3.3, 3.4**

**Property 7: C2PA Signature Verification**
*For any* image containing C2PA content credentials, the provenance analyzer should attempt cryptographic signature verification and report the verification result.
**Validates: Requirements 3.2**

**Property 8: Neutral Missing Metadata**
*For any* pair of otherwise identical images where one has metadata and one doesn't, the image without metadata should not receive a higher AI score solely due to missing metadata.
**Validates: Requirements 3.5**

### Text Analysis Properties

**Property 9: Comprehensive Text Metrics**
*For any* text longer than 200 words, the analysis should calculate perplexity, sentence burstiness, paragraph burstiness, and detect AI-specific writing patterns, with all metrics included in the result.
**Validates: Requirements 4.1, 4.2, 4.3**

**Property 10: Short Text Heuristics**
*For any* text between 50-100 words, the analysis should apply specialized short-text heuristics that differ from long-text analysis, and confidence should be reduced compared to longer texts.
**Validates: Requirements 4.4, 4.5**

**Property 11: Language Detection and Adaptation**
*For any* text input, the system should automatically detect the language and apply language-specific analysis patterns when available, falling back to language-agnostic metrics for unsupported languages.
**Validates: Requirements 13.1, 13.2, 13.4**

**Property 12: Language Support Confidence Adjustment**
*For any* text in a language with limited support, the confidence score should be lower than for the same text in a fully supported language.
**Validates: Requirements 13.5**

### Fusion and Confidence Properties

**Property 13: Variance-Based Confidence**
*For any* analysis with multiple signals, when signal scores have high variance (disagreement), the overall confidence should be reduced proportionally, with higher variance resulting in lower confidence.
**Validates: Requirements 5.1**

**Property 14: Per-Signal Confidence**
*For any* analysis, each signal should have its own confidence score based on input quality metrics, and these should be included in the confidence breakdown.
**Validates: Requirements 5.2, 5.4**

**Property 15: Minimum Confidence Threshold**
*For any* analysis where overall confidence is below 0.4, the verdict must be UNCERTAIN regardless of the AI probability score.
**Validates: Requirements 5.3**

**Property 16: Adaptive Content-Type Thresholds**
*For any* two images of different content types (e.g., PHOTOGRAPH vs SCREENSHOT), the system should apply different threshold values, with photographs using stricter thresholds than screenshots.
**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

### Performance and Caching Properties

**Property 17: Content-Addressed Cache Round-Trip**
*For any* content (image or text), analyzing the same content twice within 24 hours should return cached results on the second analysis, with identical analysis results except for the cache_hit flag.
**Validates: Requirements 7.2**

**Property 18: Batch Concurrent Processing**
*For any* batch of N images, processing them as a batch should complete faster than processing them sequentially, demonstrating concurrent execution.
**Validates: Requirements 7.4**

**Property 19: Large Image Downsampling**
*For any* image with dimensions exceeding 2048 pixels in either direction, the analysis should downsample the image to a maximum dimension of 2048 pixels before processing.
**Validates: Requirements 11.3**

### Error Handling Properties

**Property 20: Partial Results on Signal Failure**
*For any* analysis where one or more signals fail with exceptions, the system should return partial results from successful signals rather than failing completely.
**Validates: Requirements 10.2, 10.3**

**Property 21: CLIP API Fallback**
*For any* image analysis when the CLIP API is unavailable or returns an error, the semantic analyzer should fall back to heuristic analysis and complete successfully.
**Validates: Requirements 2.4**

### UI and Explainability Properties

**Property 22: Low Confidence Warning Display**
*For any* analysis result with confidence below 0.5, the extension UI should prominently display uncertainty warnings with explanations.
**Validates: Requirements 8.3**

**Property 23: Session History Tracking**
*For any* page session, performing multiple analyses should result in all analysis results being stored and displayable in the historical results view.
**Validates: Requirements 8.5**

**Property 24: Snip Mode Region Capture**
*For any* rectangular region selected in snip mode, the captured content should include only elements whose bounding boxes intersect with the selected region.
**Validates: Requirements 9.3**

**Property 25: Snip Mode Dual Analysis**
*For any* region selected in snip mode, the analysis should include both text content and images within that region, with both types reflected in the results.
**Validates: Requirements 9.4**

**Property 26: Signal Explanations**
*For any* analysis result, each signal should include a human-readable explanation string describing why it received its score.
**Validates: Requirements 12.1**

**Property 27: Uncertain Verdict Explanation**
*For any* analysis with UNCERTAIN verdict, the explanation should identify which signals disagreed and describe the source of uncertainty.
**Validates: Requirements 12.2**

**Property 28: Confidence Intervals**
*For any* analysis result, the probability estimate should include confidence intervals indicating the range of plausible values.
**Validates: Requirements 12.4**

## Error Handling

### Error Categories

**1. External API Failures**
- CLIP API unavailable or timeout
- Language detection service failure
- C2PA verification service unavailable

**Handling Strategy:**
- Graceful degradation to fallback methods
- Log warning with details
- Continue analysis with remaining signals
- Reduce confidence score appropriately
- Include warning in result explanations

**2. Input Validation Errors**
- Invalid image format
- Corrupted image data
- Text encoding issues
- Oversized inputs

**Handling Strategy:**
- Return HTTP 400 with descriptive error message
- Log validation failure details
- Do not attempt analysis
- Provide user-friendly error guidance

**3. Resource Exhaustion**
- Memory limits exceeded
- Rate limits exceeded
- Cache storage full
- Concurrent request limit reached

**Handling Strategy:**
- Return HTTP 429 (rate limit) or 503 (resource exhaustion)
- Include Retry-After header
- Queue requests when possible
- Log resource metrics
- Implement circuit breaker pattern

**4. Analysis Failures**
- Signal analyzer exception
- Fusion algorithm error
- Unexpected data format

**Handling Strategy:**
- Catch exceptions at signal level
- Continue with remaining signals
- Return partial results when possible
- Log full stack trace with context
- Include error details in response warnings

**5. Cache Failures**
- Redis connection lost
- Cache corruption
- Serialization errors

**Handling Strategy:**
- Fall back to non-cached analysis
- Log cache error details
- Continue normal operation
- Monitor cache health
- Auto-reconnect on transient failures

### Error Recovery Patterns

**Circuit Breaker for External APIs:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("Service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

**Partial Result Aggregation:**
```python
async def analyze_with_partial_results(image: Image) -> AnalysisResult:
    signals = {}
    errors = []
    
    # Try each signal independently
    for signal_name, analyzer in signal_analyzers.items():
        try:
            signals[signal_name] = await analyzer.analyze(image)
        except Exception as e:
            logger.error(f"Signal {signal_name} failed: {e}")
            errors.append(f"{signal_name}: {str(e)}")
    
    # Require at least 2 signals for valid result
    if len(signals) < 2:
        raise AnalysisError("Insufficient signals for analysis")
    
    # Fuse available signals
    result = fusion_engine.fuse(signals)
    result.warnings.extend(errors)
    result.confidence *= (len(signals) / len(signal_analyzers))  # Reduce confidence
    
    return result
```

### Timeout Configuration

**Per-Operation Timeouts:**
- Image download: 10 seconds
- CLIP API call: 5 seconds
- Frequency analysis: 2 seconds
- Text analysis: 1 second
- Cache operations: 500ms
- Overall request: 30 seconds

**Implementation:**
```python
async def analyze_with_timeout(content, timeout=30):
    try:
        return await asyncio.wait_for(
            analyze_internal(content),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise AnalysisTimeout(f"Analysis exceeded {timeout}s timeout")
```

## Testing Strategy

### Dual Testing Approach

TruthLens enhancements require both unit tests and property-based tests for comprehensive coverage:

**Unit Tests** focus on:
- Specific examples and edge cases
- Integration points between components
- Error conditions and failure modes
- Configuration loading and validation
- API endpoint behavior
- UI component rendering

**Property-Based Tests** focus on:
- Universal properties across all inputs
- Comprehensive input coverage through randomization
- Invariants that must hold for all valid data
- Round-trip properties (cache, serialization)
- Metamorphic properties (threshold adaptation)

### Property-Based Testing Configuration

**Library Selection:**
- Python backend: **Hypothesis** (mature, well-integrated with pytest)
- JavaScript extension: **fast-check** (TypeScript support, browser compatible)

**Test Configuration:**
- Minimum 100 iterations per property test
- Configurable via environment variable: `PROPERTY_TEST_ITERATIONS`
- Shrinking enabled for failure case minimization
- Deterministic mode for CI/CD (fixed seed)

**Property Test Tagging:**
Each property test must include a comment referencing the design property:
```python
@given(st.images())
def test_dual_transform_analysis(image):
    """
    Feature: truthlens-enhancements, Property 1: Dual Transform Frequency Analysis
    For any image, frequency analysis should produce both FFT and DCT scores.
    """
    result = frequency_analyzer.analyze(image)
    assert 'fft_score' in result
    assert 'dct_score' in result
    assert 'periodicity_strength' in result
```

### Unit Test Coverage

**Image Analysis Tests:**
- Test FFT analysis with known periodic patterns
- Test DCT analysis with block artifacts
- Test JPEG compression detection at various quality levels
- Test EXIF parsing with various camera models
- Test C2PA verification with valid/invalid signatures
- Test metadata tampering detection with crafted examples
- Test CLIP API integration with mocked responses
- Test fallback to heuristic analysis when API fails

**Text Analysis Tests:**
- Test perplexity calculation with known AI/human text
- Test burstiness with uniform vs varied sentence lengths
- Test language detection with samples from 10+ languages
- Test AI phrase detection with known patterns
- Test short text handling (<100 words)
- Test phishing detection with known phishing emails
- Test misinformation risk with emotional language

**Fusion Engine Tests:**
- Test confidence calculation with various signal variances
- Test threshold adaptation for different content types
- Test minimum confidence enforcement (< 0.4 → UNCERTAIN)
- Test signal weight application
- Test partial result handling when signals fail

**Caching Tests:**
- Test cache hit/miss behavior
- Test cache TTL expiration
- Test cache invalidation
- Test content hash collision handling
- Test cache statistics accuracy

**API Tests:**
- Test rate limiting enforcement
- Test batch processing endpoint
- Test error response formats
- Test timeout handling
- Test CORS configuration

**Extension Tests:**
- Test snip mode region selection
- Test result panel rendering
- Test signal breakdown visualization
- Test copy-to-clipboard functionality
- Test session history tracking
- Test error message display

### Integration Tests

**End-to-End Workflows:**
1. Upload image → Analyze → Verify all signals executed → Check result format
2. Submit text → Analyze → Verify language detection → Check confidence adjustment
3. Batch request → Verify concurrent processing → Check cache population
4. Analyze same content twice → Verify cache hit on second request
5. Exceed rate limit → Verify 429 response with retry-after header
6. Simulate CLIP API failure → Verify fallback to heuristic analysis
7. Snip mode selection → Capture region → Analyze → Verify correct content captured

**Performance Tests:**
- Measure analysis time for various image sizes
- Measure batch processing speedup vs sequential
- Measure cache hit rate under realistic load
- Measure memory usage with concurrent requests
- Measure API response time percentiles (p50, p95, p99)

### Test Data

**Image Test Set:**
- 100 AI-generated images (Midjourney, DALL-E, Stable Diffusion)
- 100 real photographs with EXIF data
- 50 edited images (Photoshop, GIMP)
- 50 screenshots and web graphics
- 20 images with C2PA credentials
- 20 images with tampered metadata

**Text Test Set:**
- 100 AI-generated texts (ChatGPT, Claude, various lengths)
- 100 human-written texts (news, blogs, social media)
- 50 phishing emails (known corpus)
- 50 texts in non-English languages
- 20 short texts (50-100 words)
- 20 texts with AI-typical phrases

### Continuous Integration

**CI Pipeline:**
1. Run unit tests (fast, < 2 minutes)
2. Run property tests with 100 iterations (medium, < 10 minutes)
3. Run integration tests (slow, < 5 minutes)
4. Generate coverage report (target: 80%+ coverage)
5. Run linting and type checking
6. Build extension package
7. Deploy to staging environment

**Property Test Failure Handling:**
- Capture shrunk failure case
- Log full input that caused failure
- Create GitHub issue with reproduction steps
- Add failure case to regression test suite

### Manual Testing Checklist

**Before Release:**
- [ ] Test with 10 AI images from each major generator
- [ ] Test with 10 real photos from different cameras
- [ ] Test text analysis in all 10 supported languages
- [ ] Test snip mode on complex web pages
- [ ] Test rate limiting with burst traffic
- [ ] Test cache behavior over 24-hour period
- [ ] Test error handling with network disconnected
- [ ] Test extension on restricted pages (chrome://)
- [ ] Test batch processing with 20+ images
- [ ] Verify all UI tooltips and explanations display correctly

