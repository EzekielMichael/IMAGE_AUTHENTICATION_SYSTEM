from django.contrib import admin
from django.utils.html import format_html
from .models import (
    UploadedImage,
    MetadataAnalysis,
    AIDetection,
    HumanEditDetection,
    FinalVerdict,
    QnAHistory,
    HeatmapAnalysis,
)


@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ("id", "filename", "width", "height", "megapixels", "uploaded_by", "uploaded_at")
    list_display_links = ("id", "filename")
    list_filter = ("uploaded_at", "uploaded_by")
    search_fields = ("filename", "uploaded_by__username")
    readonly_fields = ("uploaded_at",)
    ordering = ("-uploaded_at",)
    
    fieldsets = (
        ("Image Information", {
            "fields": ("image", "filename", "uploaded_by")
        }),
        ("Dimensions", {
            "fields": ("width", "height", "megapixels")
        }),
        ("Metadata", {
            "fields": ("uploaded_at",)
        }),
    )


@admin.register(MetadataAnalysis)
class MetadataAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "camera_make", "camera_model", "iso", "f_number", 
                   "gps_latitude", "gps_longitude", "ai_tool", "verdict_preview", "analyzed_at")
    list_display_links = ("id", "image")
    list_filter = ("camera_make", "camera_model", "ai_tool", "c2pa_present", "analyzed_at")
    search_fields = ("image__filename", "camera_make", "camera_model", "software", "verdict")
    readonly_fields = ("analyzed_at",)
    
    def verdict_preview(self, obj):
        return obj.verdict[:50] + "..." if len(obj.verdict) > 50 else obj.verdict
    verdict_preview.short_description = "Verdict"
    
    fieldsets = (
        ("Image Reference", {
            "fields": ("image",)
        }),
        ("Camera & Device", {
            "fields": ("camera_make", "camera_model", "software")
        }),
        ("Timestamps", {
            "fields": ("date_original", "modify_date")
        }),
        ("Camera Settings", {
            "fields": ("exposure_time", "f_number", "iso", "focal_length")
        }),
        ("Location Data", {
            "fields": ("gps_latitude", "gps_longitude"),
            "classes": ("collapse",)
        }),
        ("Technical Info", {
            "fields": ("resolution_width", "resolution_height")
        }),
        ("Analysis", {
            "fields": ("c2pa_present", "ai_tool", "verdict", "exif_full")
        }),
        ("Metadata", {
            "fields": ("analyzed_at",)
        }),
    )


@admin.register(AIDetection)
class AIDetectionAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "label", "confidence_colored", "model_name", "analyzed_at")
    list_display_links = ("id", "image")
    list_filter = ("label", "model_name", "analyzed_at")
    search_fields = ("image__filename", "label")
    readonly_fields = ("analyzed_at",)
    ordering = ("-confidence",)
    
    def confidence_colored(self, obj):
        color = "green" if obj.confidence < 30 else "orange" if obj.confidence < 75 else "red"
        return format_html('<span style="color: {}; font-weight: bold;">{}%</span>', color, obj.confidence)
    confidence_colored.short_description = "Confidence"
    
    fieldsets = (
        ("Image Reference", {
            "fields": ("image",)
        }),
        ("Detection Results", {
            "fields": ("label", "confidence", "model_name")
        }),
        ("Metadata", {
            "fields": ("analyzed_at",)
        }),
    )


@admin.register(HumanEditDetection)
class HumanEditDetectionAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "detection_score_colored", "model_name", "heatmap_preview", "analyzed_at")
    list_display_links = ("id", "image")
    list_filter = ("model_name", "heatmap_analyzed", "analyzed_at")
    search_fields = ("image__filename",)
    readonly_fields = ("analyzed_at",)
    
    def detection_score_colored(self, obj):
        color = "green" if obj.detection_score < 0.3 else "orange" if obj.detection_score < 0.7 else "red"
        # Convert to percentage string first, then format
        percentage = f"{obj.detection_score * 100:.1f}%"
        return format_html('<span style="color: {}; font-weight: bold;">{}</span>', color, percentage)
    detection_score_colored.short_description = "Detection Score"
    
    def heatmap_preview(self, obj):
        if obj.heatmap and obj.heatmap.url:
            return format_html('<img src="{}" width="50" height="50" style="border-radius: 4px;" />', obj.heatmap.url)
        return "No heatmap"
    heatmap_preview.short_description = "Heatmap"
    
    fieldsets = (
        ("Image Reference", {
            "fields": ("image",)
        }),
        ("Detection Outputs", {
            "fields": ("heatmap", "overlay", "detection_score", "model_name")
        }),
        ("Cache Information", {
            "fields": ("heatmap_analyzed", "heatmap_last_accessed"),
            "classes": ("collapse",)
        }),
        ("Metadata", {
            "fields": ("analyzed_at",)
        }),
    )


@admin.register(HeatmapAnalysis)
class HeatmapAnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "severity_colored", "max_edit_score", "avg_edit_score", 
                   "edited_percentage", "primary_region", "edit_zones_count", "analyzed_at")
    list_display_links = ("id", "image")
    list_filter = ("severity_level", "primary_region", "analyzed_at", "last_accessed")
    search_fields = ("image__filename", "primary_region", "secondary_region")
    readonly_fields = ("analyzed_at", "last_accessed")
    
    def severity_colored(self, obj):
        colors = {
            'LOW': 'green',
            'MEDIUM': 'orange',
            'HIGH': 'red',
            'VERY_HIGH': 'darkred'
        }
        color = colors.get(obj.severity_level, 'black')
        return format_html('<span style="color: {}; font-weight: bold;">{}</span>', color, obj.get_severity_level_display())
    severity_colored.short_description = "Severity"
    
    fieldsets = (
        ("Image Reference", {
            "fields": ("image",)
        }),
        ("Grid Analysis (3x3)", {
            "fields": ("grid_scores",)
        }),
        ("Statistics", {
            "fields": ("max_edit_score", "avg_edit_score", "edited_percentage")
        }),
        ("Region Detection", {
            "fields": ("primary_region", "secondary_region", "region_scores")
        }),
        ("Multi-Zone Detection", {
            "fields": ("edit_zones_count", "zones_details")
        }),
        ("Classification", {
            "fields": ("severity_level",)
        }),
        ("Metadata", {
            "fields": ("analyzed_at", "last_accessed")
        }),
    )


@admin.register(QnAHistory)
class QnAHistoryAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "question_preview", "answer_preview", "question_type", 
                   "response_time_ms", "was_from_cache", "helpful_rating", "asked_at")
    list_display_links = ("id", "image")
    list_filter = ("question_type", "was_from_cache", "helpful_rating", "asked_at")
    search_fields = ("image__filename", "question", "answer")
    readonly_fields = ("asked_at",)
    ordering = ("-asked_at",)
    
    def question_preview(self, obj):
        return obj.question[:60] + "..." if len(obj.question) > 60 else obj.question
    question_preview.short_description = "Question"
    
    def answer_preview(self, obj):
        return obj.answer[:60] + "..." if len(obj.answer) > 60 else obj.answer
    answer_preview.short_description = "Answer"
    
    fieldsets = (
        ("Image Reference", {
            "fields": ("image",)
        }),
        ("Q&A Content", {
            "fields": ("question", "answer", "question_type")
        }),
        ("Context Used", {
            "fields": ("used_ai_confidence", "used_human_score", "used_primary_region", "used_verdict")
        }),
        ("Performance", {
            "fields": ("response_time_ms", "was_from_cache", "helpful_rating")
        }),
        ("Metadata", {
            "fields": ("asked_at",)
        }),
    )


@admin.register(FinalVerdict)
class FinalVerdictAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "verdict", "confidence_score", "explanation_preview", "created_at")
    list_display_links = ("id", "image")
    list_filter = ("verdict", "created_at")
    search_fields = ("image__filename", "verdict", "explanation")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
    
    def explanation_preview(self, obj):
        return obj.explanation[:80] + "..." if len(obj.explanation) > 80 else obj.explanation
    explanation_preview.short_description = "Explanation"
    
    # Add colored confidence display
    def confidence_score(self, obj):
        confidence = obj.confidence_score
        if confidence >= 80:
            color = "#10b981"  # Green
        elif confidence >= 60:
            color = "#f59e0b"  # Orange
        elif confidence >= 40:
            color = "#f97316"  # Orange-red
        else:
            color = "#ef4444"  # Red
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}%</span>',
            color,
            round(confidence, 1)
        )
    confidence_score.short_description = "Confidence"
    
    fieldsets = (
        ("Image Reference", {
            "fields": ("image",)
        }),
        ("Verdict", {
            "fields": ("verdict", "confidence_score", "explanation")
        }),
        ("Metadata", {
            "fields": ("created_at",)
        }),
    )