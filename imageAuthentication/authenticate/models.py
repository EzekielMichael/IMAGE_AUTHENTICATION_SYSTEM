from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

# ===============================
# 1. Uploaded Images
# ===============================
class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    filename = models.CharField(max_length=255)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    megapixels = models.FloatField(null=True, blank=True)
    
    # Link to the user who uploaded the image
    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name="uploaded_images",
        null=True, 
        blank=True
    )
    
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename


# ===============================
# 2. Metadata Analysis
# ===============================
class MetadataAnalysis(models.Model):
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="metadata_analysis"
    ) 
    
    # Core Device & Source
    camera_make = models.CharField(max_length=100, blank=True)
    camera_model = models.CharField(max_length=100, blank=True)
    software = models.CharField(max_length=200, blank=True)
    
    # Critical Timestamps
    date_original = models.CharField(max_length=100, blank=True)
    modify_date = models.CharField(max_length=100, blank=True)
    
    # Basic Camera Settings
    exposure_time = models.CharField(max_length=50, blank=True)
    f_number = models.FloatField(null=True, blank=True)
    iso = models.IntegerField(null=True, blank=True)
    focal_length = models.FloatField(null=True, blank=True)
    
    # Location
    gps_latitude = models.FloatField(null=True, blank=True)
    gps_longitude = models.FloatField(null=True, blank=True)
    
    # Technical Info
    resolution_width = models.IntegerField(null=True, blank=True)
    resolution_height = models.IntegerField(null=True, blank=True)
    
    # Advanced
    c2pa_present = models.BooleanField(default=False)
    
    # Storage
    exif_full = models.JSONField(null=True, blank=True)
    
    # Analysis
    ai_tool = models.CharField(max_length=100, blank=True)
    verdict = models.TextField(blank=True)
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Metadata for {self.image.filename}"


# ===============================
# 3. AI Image Detection
# ===============================
class AIDetection(models.Model):
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="ai_detection"
    )
    label = models.CharField(max_length=50)
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    model_name = models.CharField(max_length=100, default="ResNet50")
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence}%)"


# ===============================
class HumanEditDetection(models.Model):
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="human_edit_detection"
    )
    heatmap = models.ImageField(upload_to="outputs/")
    overlay = models.ImageField(upload_to="outputs/")
    detection_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    model_name = models.CharField(max_length=100, default="TruFor")
    analyzed_at = models.DateTimeField(auto_now_add=True)
    
    # Cache fields
    heatmap_analyzed = models.BooleanField(default=False)
    heatmap_last_accessed = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Edit detection for {self.image.filename}"


# ===============================
# 5. Heatmap Region Analysis (NEW)
# ===============================
class HeatmapAnalysis(models.Model):
    """Pre-computed heatmap region analysis for fast Q&A"""
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="heatmap_analysis"
    )
    
    # Grid-based analysis (3x3 grid)
    grid_scores = models.JSONField(null=True, blank=True)
    # Format: {"top_left": 0.12, "top_center": 0.45, "top_right": 0.08,
    #          "middle_left": 0.23, "middle_center": 0.89, "middle_right": 0.34,
    #          "bottom_left": 0.05, "bottom_center": 0.11, "bottom_right": 0.03}
    
    # Overall statistics
    max_edit_score = models.FloatField(default=0.0)
    avg_edit_score = models.FloatField(default=0.0)
    edited_percentage = models.FloatField(default=0.0)  # % of image with editing > threshold
    
    # Edit concentration
    primary_region = models.CharField(max_length=50, blank=True)
    secondary_region = models.CharField(max_length=50, blank=True)
    
    # Region-specific scores
    region_scores = models.JSONField(null=True, blank=True)
    # Format: {"face": 0.85, "background": 0.12, "text": 0.45}
    
    # Multi-zone detection
    edit_zones_count = models.IntegerField(default=0)
    zones_details = models.JSONField(null=True, blank=True)  # Coordinates and scores
    
    # Severity level
    severity_level = models.CharField(
        max_length=20,
        choices=[
            ('LOW', 'Low'),
            ('MEDIUM', 'Medium'),
            ('HIGH', 'High'),
            ('VERY_HIGH', 'Very High')
        ],
        default='LOW'
    )
    
    # Cache metadata
    analyzed_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['image', '-analyzed_at']),
        ]
    
    def __str__(self):
        primary = self.primary_region if self.primary_region else "None"
        return f"Heatmap analysis for {self.image.filename} (Primary: {primary})"


# ===============================
# 6. Q&A History (NEW)
# ===============================
class QnAHistory(models.Model):
    """Stores questions and answers for each image with caching"""
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="qna_history"
    )
    
    question = models.TextField()
    answer = models.TextField()
    question_type = models.CharField(
        max_length=50,
        choices=[
            ('LOCATION', 'Where edited'),
            ('REASON', 'Why AI/edited'),
            ('CONFIDENCE', 'How confident'),
            ('SEVERITY', 'How severe'),
            ('GENERAL', 'General question'),
            ('COMPARISON', 'Compare regions'),
        ],
        default='GENERAL'
    )
    
    # Context used for generating answer
    used_ai_confidence = models.FloatField(null=True, blank=True)
    used_human_score = models.FloatField(null=True, blank=True)
    used_primary_region = models.CharField(max_length=50, blank=True)
    used_verdict = models.CharField(max_length=100, blank=True)
    
    # Metadata
    asked_at = models.DateTimeField(auto_now_add=True)
    response_time_ms = models.IntegerField(default=0)
    was_from_cache = models.BooleanField(default=False)
    helpful_rating = models.IntegerField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )  # Optional user rating
    
    class Meta:
        ordering = ['-asked_at']
        indexes = [
            models.Index(fields=['image', '-asked_at']),
            models.Index(fields=['image', 'question_type']),
        ]
    
    def __str__(self):
        return f"Q: {self.question[:50]}... (asked {self.asked_at})"


# ===============================
# 7. Final Verdict
# ===============================
class FinalVerdict(models.Model):
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="final_verdict"
    )
    verdict = models.CharField(max_length=100)
    explanation = models.TextField()
    confidence_score = models.FloatField(default=0.0)  # Add this field
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Final verdict for {self.image.filename}"

class GeneratedReport(models.Model):
    """Stores generated PDF reports for images"""
    image = models.ForeignKey(
        UploadedImage,
        on_delete=models.CASCADE,
        related_name="generated_reports"
    )
    
    # The actual PDF file
    report_file = models.FileField(upload_to="reports/")
    
    # Who generated it
    generated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="generated_reports"
    )
    
    # Report configuration
    include_heatmap = models.BooleanField(default=False)
    include_images = models.BooleanField(default=True)
    
    # Snapshot of results at time of generation
    verdict_snapshot = models.CharField(max_length=100, blank=True)
    ai_score_snapshot = models.FloatField(null=True, blank=True)
    human_score_snapshot = models.FloatField(null=True, blank=True)
    
    # Metadata
    generated_at = models.DateTimeField(auto_now_add=True)
    file_size = models.IntegerField(null=True, blank=True)  # Size in bytes

    # NEW SHARING FIELDS
    is_public = models.BooleanField(default=False)  # Allow sharing
    share_token = models.CharField(max_length=100, unique=True, null=True, blank=True)
    shared_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="shared_reports"
    )
    shared_at = models.DateTimeField(null=True, blank=True)
    view_count = models.IntegerField(default=0)
    
    # Add these methods
    def generate_share_token(self):
        import uuid
        self.share_token = str(uuid.uuid4()).replace('-', '')[:20]
        self.is_public = True
        self.shared_at = timezone.now()
        self.save()
    
    def get_share_url(self, request):
        return request.build_absolute_uri(f'/report/shared/{self.share_token}/')
    
    
    class Meta:
        ordering = ['-generated_at']
        indexes = [
            models.Index(fields=['image', '-generated_at']),
            models.Index(fields=['generated_by', '-generated_at']),
        ]
    
    def __str__(self):
        return f"Report for {self.image.filename} ({self.generated_at.strftime('%Y-%m-%d %H:%M')})"
    
