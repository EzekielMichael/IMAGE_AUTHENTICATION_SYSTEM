    # ====================== FIX MATPLOTLIB BACKEND ======================
import matplotlib
matplotlib.use('Agg')  # Must be BEFORE importing pyplot
import matplotlib.pyplot as plt
import os
import uuid
import gc, time
from django.db.models import Q
from django.utils import timezone
import torch
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django import forms
from django.http import Http404, JsonResponse
from django.conf import settings
from photoholmes.methods.trufor.method import TruFor
from photoholmes.methods.trufor.preprocessing import trufor_preprocessing
from photoholmes.utils.image import read_image, overlay_mask
import exiftool
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.contrib import messages
from .utils import encode_id, decode_id
from django.contrib.auth.decorators import login_required
from django.db.models import Prefetch, Q, Exists, OuterRef
from .forms import CustomUserCreationForm
from .heatmap_analyzer import analyze_heatmap_from_path
from .qa_system import qa_system
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
import json
from django.core.paginator import Paginator
from django.db.models import Prefetch, Exists, OuterRef
from django.http import Http404, HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML
from .models import UploadedImage, MetadataAnalysis, AIDetection, HumanEditDetection, FinalVerdict, HeatmapAnalysis, GeneratedReport
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import user_passes_test

from .models import (
    UploadedImage,
    MetadataAnalysis,
    AIDetection,
    HumanEditDetection,
    FinalVerdict,
    HeatmapAnalysis
)

YELLOW_COLOR = '\033[93m'  # Yellow text
END_COLOR = '\033[0m'

# ====================== LOAD MODELS ONCE ======================
BASE_DIR = Path(__file__).resolve().parent
IMAGE_MODELS_DIR = BASE_DIR / "Image_models"

# 2. Path for TruFor model weights
trufor_weights_path = IMAGE_MODELS_DIR / "Localization.pth.tar"
Human_Edit_model = TruFor(
    weights=str(trufor_weights_path),
    device="cpu"
)
Human_Edit_model.eval()
# #=========================== NORMAL MODELS =============================

ai_model = models.resnet50(weights=None)
DEVICE = torch.device("cpu")

# Match EXACT architecture from training
ai_model.fc = nn.Sequential(
    nn.Linear(2048, 256),  # fc.0
    nn.ReLU(),             # fc.1
    nn.Dropout(0.4),       # fc.2
    nn.Linear(256, 2)      # fc.3
)

print(f"{YELLOW_COLOR}Loading AI detection Model... {END_COLOR}")
# Load the trained weights
# 1. Path for PyTorch checkpoint
resnet_path = IMAGE_MODELS_DIR / "best_resnet50_ai_detector_Eze_04.pth"
checkpoint = torch.load(str(resnet_path), map_location=DEVICE)

# Load state dict
ai_model.load_state_dict(checkpoint)
ai_model.to(DEVICE)
ai_model.eval()

# Classes (order matters: index 0 = Fake/AI, index 1 = Real)
classes = ["AI-generated", "Real Image"]

# IMPORTANT: Use SAME transform as validation (with normalization)
ai_transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 256 first
    # transforms.Resize((224, 224)),  # Force square
    # transforms.CenterCrop(224),  # Then crop to 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


MAX_WIDTH = 708
MAX_HEIGHT = 1080


# ====================== USER ROLES =====================
def is_superuser(user):
    """Check if user is superuser"""
    return user.is_authenticated and user.is_superuser

def is_staff(user):
    """Check if user is staff member (includes superusers)"""
    return user.is_authenticated and user.is_staff

def is_active_user(user):
    """Check if user is active"""
    return user.is_authenticated and user.is_active

def is_admin_or_staff(user):
    """Check if user is superuser or staff"""
    return user.is_authenticated and (user.is_superuser or user.is_staff)

# ====================== REGISTRATION ======================
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import EmailMessage
from django.contrib.auth import get_user_model

User = get_user_model()

def register(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()

            mail_subject = 'Activate your account.'
            message = render_to_string('acc_active_email.html', {
                'user': user,
                'domain': request.get_host(), # Dynamically gets current domain
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': default_token_generator.make_token(user),
                'protocol': 'https' if request.is_secure() else 'http',
            })
            email = EmailMessage(mail_subject, message, to=[user.email])
            email.send()

            messages.success(request, "Registration successful! Please check your email to activate your account.")
            return redirect('login')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = CustomUserCreationForm()

    return render(request, 'register.html', {'form': form})


def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        messages.success(request, "Your account has been activated successfully! You can now log in.")
        return redirect('login')
    else:
        messages.error(request, "Activation link is invalid or has expired!")
        return redirect('register')


class EmailOrUsernameAuthenticationForm(AuthenticationForm):
    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username and password:
            # Try to authenticate with username first
            self.user_cache = authenticate(self.request, username=username, password=password)
            
            # If not found, try with email
            if self.user_cache is None:
                try:
                    # Get user by email
                    user_by_email = User.objects.get(email=username)
                    # Authenticate using the username from found email
                    self.user_cache = authenticate(self.request, username=user_by_email.username, password=password)
                except User.DoesNotExist:
                    pass
            
            # Handle invalid login
            if self.user_cache is None:
                raise forms.ValidationError(
                    "Invalid email/username or password.",
                    code='invalid_login',
                )
            else:
                self.confirm_login_allowed(self.user_cache)
        
        return self.cleaned_data

def user_login(request):
    if request.method == "POST":
        form = EmailOrUsernameAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.first_name or user.username}!")
            return redirect('index')
        else:
            messages.error(request, "Invalid email/username or password.")
    else:
        form = EmailOrUsernameAuthenticationForm()
    
    return render(request, 'login.html', {'form': form})


def user_logout(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('index')


@login_required
def profile_view(request):
    """Display user profile page"""
    from .models import UploadedImage  # Import the correct model
    
    # Get user statistics
    stats = {
        'total_analyses': UploadedImage.objects.filter(uploaded_by=request.user).count(),
        'member_since': request.user.date_joined.strftime('%b %Y') if request.user.date_joined else 'N/A',
    }
    
    return render(request, 'profile.html', {
        'user': request.user,
        'stats': stats
    })


@login_required
def update_profile(request):
    """Handle profile updates via AJAX"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user = request.user
            
            # Update fields if provided
            if 'first_name' in data:
                user.first_name = data['first_name'].strip()
            if 'last_name' in data:
                user.last_name = data['last_name'].strip()
            if 'email' in data:
                new_email = data['email'].strip()
                # Check if email is already taken by another user
                if User.objects.exclude(pk=user.pk).filter(email=new_email).exists():
                    return JsonResponse({
                        'success': False, 
                        'error': 'Email already exists'
                    }, status=400)
                user.email = new_email
            
            user.save()
            
            return JsonResponse({
                'success': True,
                'user': {
                    'username': user.username,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'email': user.email
                }
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)


def process_ai_image(image_path):
    """Process image with new model"""
    image = Image.open(image_path).convert("RGB")
    img_tensor = ai_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = ai_model(img_tensor)
    
    probs = F.softmax(outputs, dim=1)
    confidence_val, predicted = torch.max(probs, 1)
    label = classes[predicted.item()]
    confidence = round(confidence_val.item() * 100, 2)
    
    # Cap confidence at 99.99%
    if confidence >= 99.9:
        confidence = 99.9
    
    return {"label": label, "confidence": confidence}


# Handle checkpoint format (could be dict or state_dict)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    ai_model.load_state_dict(checkpoint['model_state_dict'])
else:
    ai_model.load_state_dict(checkpoint)


# ====================== HELPER FUNCTIONS ======================
def cleanup_figure(fig):
    """Properly close and cleanup matplotlib figure"""
    try:
        plt.close(fig)
        gc.collect()  # Force garbage collection
    except Exception as e:
        print(f"Error cleaning up figure: {e}")


def safe_float(value):
    try:
        if value is None:
            return None
        return float(str(value).split()[0])
    except:
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(float(str(value).split()[0]))
    except:
        return None

# ====================== HELPER FUNCTIONS ======================
def resize_if_needed(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        img.save(image_path)
    return image_path


def process_human_edit(image_path):
    """Process image with TruFor and save heatmap/overlay"""
    image_path = resize_if_needed(image_path)
    image = read_image(image_path)
    inputs = trufor_preprocessing(image=image)
    
    with torch.no_grad():
        heatmap, confidence, detection, _ = Human_Edit_model.predict(**inputs)
    
    heatmap = heatmap.cpu().numpy()
    
    unique_id = uuid.uuid4().hex[:12]
    output_dir = os.path.join(settings.MEDIA_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Heatmap
    heatmap_filename = f"heatmap_{unique_id}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    
    fig1 = None
    try:
        fig1 = plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")
        plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0, dpi=100)
    finally:
        if fig1:
            cleanup_figure(fig1)
    
    # Generate Overlay
    overlay_filename = f"overlay_{unique_id}.png"
    overlay_path = os.path.join(output_dir, overlay_filename)
    
    fig2 = None
    try:
        overlay = overlay_mask(image.permute(1, 2, 0).numpy(), heatmap)
        fig2 = plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.axis("off")
        plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0, dpi=100)
    finally:
        if fig2:
            cleanup_figure(fig2)
    
    # Return relative paths for database storage
    return {
        "heatmap": f"outputs/{heatmap_filename}",
        "overlay": f"outputs/{overlay_filename}",
        "detection": float(detection)
    }


# ====================== UPDATED METADATA PROCESSING ======================
def process_metadata(image_path):
    from datetime import datetime
    import os
    
    try:
        with exiftool.ExifToolHelper() as et:
            meta = et.get_metadata(image_path)[0]
    except Exception:
        meta = {}

    # ================= BASIC FIELDS =================
    make = meta.get("EXIF:Make")
    model = meta.get("EXIF:Model")
    software = meta.get("EXIF:Software") or meta.get("XMP:CreatorTool") or meta.get("PNG:CreatorTool")

    # ================= TIMESTAMP LOGIC =================
    date_original = meta.get("EXIF:DateTimeOriginal") or meta.get("EXIF:CreateDate")
    
    # Get EXIF modify date
    exif_modify_date = meta.get("EXIF:ModifyDate")
    
    # Get file modification time from filesystem
    current_time = datetime.now()
    file_modified_time = None
    try:
        file_stat = os.stat(image_path)
        file_modified_time = datetime.fromtimestamp(file_stat.st_mtime)
    except:
        pass
    
    # Parse modify date if it exists
    modify_date = None
    
    if date_original:
        # Has capture date - store modify date if available (even if same as capture)
        if exif_modify_date:
            modify_date = exif_modify_date
        elif file_modified_time:
            modify_date = file_modified_time.strftime("%Y:%m:%d %H:%M:%S")
    else:
        # No capture date - only use modify date if it's NOT the current upload time
        valid_modify_date = None
        
        # Check EXIF modify date
        if exif_modify_date:
            try:
                # Parse EXIF modify date
                if isinstance(exif_modify_date, str):
                    for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"]:
                        try:
                            parsed_date = datetime.strptime(exif_modify_date, fmt)
                            # Check if it's within 5 seconds of current time (upload time)
                            time_diff = abs((current_time - parsed_date).total_seconds())
                            if time_diff > 5:
                                valid_modify_date = exif_modify_date
                            break
                        except:
                            continue
            except:
                pass
        
        # If no valid EXIF modify date, check file modification time
        if not valid_modify_date and file_modified_time:
            time_diff = abs((current_time - file_modified_time).total_seconds())
            if time_diff > 5:
                valid_modify_date = file_modified_time.strftime("%Y:%m:%d %H:%M:%S")
        
        modify_date = valid_modify_date

    exposure_time = meta.get("EXIF:ExposureTime")
    f_number = meta.get("EXIF:FNumber")
    iso = meta.get("EXIF:ISO") or meta.get("EXIF:ISOSpeedRatings")
    focal_length = meta.get("EXIF:FocalLength")

    # ================= GPS =================
    gps_lat = (
        meta.get("Composite:GPSLatitude")
        or meta.get("EXIF:GPSLatitude")
        or meta.get("GPS:GPSLatitude")
    )

    gps_lon = (
        meta.get("Composite:GPSLongitude")
        or meta.get("EXIF:GPSLongitude")
        or meta.get("GPS:GPSLongitude")
    )

    # ================= IMAGE SIZE =================
    width = (
        meta.get("EXIF:ExifImageWidth")
        or meta.get("PNG:ImageWidth")
        or meta.get("File:ImageWidth")
    )

    height = (
        meta.get("EXIF:ExifImageHeight")
        or meta.get("PNG:ImageHeight")
        or meta.get("File:ImageHeight")
    )

    # ================= C2PA DETECTION =================
    has_c2pa = any(
        "c2pa" in str(k).lower() or "jumbf" in str(k).lower()
        for k in meta.keys()
    )

    # ================= AI TOOL DETECTION =================
    ai_tool = ""

    if meta.get("JUMBF:ActionsSoftwareAgentName"):
        ai_tool = str(meta.get("JUMBF:ActionsSoftwareAgentName"))
    elif meta.get("JUMBF:Claim_Generator_InfoName"):
        ai_tool = str(meta.get("JUMBF:Claim_Generator_InfoName"))
    elif "GPT-4o" in str(meta):
        ai_tool = "GPT-4o"
    elif "ChatGPT" in str(meta):
        ai_tool = "ChatGPT"
    elif has_c2pa and any("trainedAlgorithmicMedia" in str(v) for v in meta.values()):
        ai_tool = "AI Generated (C2PA)"

    # ================= VERDICT =================
    camera_str = f"{make or ''} {model or ''}".strip()

    verdict_parts = []
    if ai_tool:
        verdict_parts.append(f"AI-generated using {ai_tool}")
    elif has_c2pa:
        verdict_parts.append("C2PA signed")
    elif camera_str:
        verdict_parts.append(f"Real camera photo ({camera_str})")
    elif software:
        verdict_parts.append(f"Edited with {software}")
    else:
        verdict_parts.append("No camera metadata (possible AI or social media)")

    return {
        "verdict": " | ".join(verdict_parts),
        "summary": {
            "make": make,
            "model": model,
            "software": software,
            "date_original": date_original,
            "modify_date": modify_date,
            "exposure_time": exposure_time,
            "f_number": f_number,
            "iso": iso,
            "focal_length": focal_length,
            "gps_latitude": gps_lat,
            "gps_longitude": gps_lon,
            "resolution_width": width,
            "resolution_height": height,
            "c2pa_present": has_c2pa,
            "ai_tool": ai_tool.strip(),
            "exif_full": meta
        }
    }


# Helper functions for safe conversion
def safe_float(value):
    try:
        return float(value) if value else None
    except (ValueError, TypeError):
        return None

def safe_int(value):
    try:
        return int(value) if value else None
    except (ValueError, TypeError):
        return None


# ====================== UPDATED INDEX VIEW ======================
def index(request):
    if request.method == "POST":
        file = request.FILES.get("image")

        if not file:
            messages.error(request, "Please select an image to upload.")
            return render(request, "index.html", {"error": "Please upload an image."})

        # ==================== IMAGE VALIDATION ====================
        allowed_extensions = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'heic']

        # Fix: Check if file is valid UploadedFile object
        if not hasattr(file, 'name'):
            messages.error(request, "Invalid file uploaded.")
            return render(request, "index.html", {"error": "Invalid file."})

        validator = FileExtensionValidator(allowed_extensions=allowed_extensions)

        try:
            validator(file)   # Pass the file object, not file.name
        except ValidationError:
            messages.error(request, "Invalid file type. Only JPG, JPEG, PNG, WEBP, and GIF are allowed.")
            return render(request, "index.html", {"error": "Only image files are allowed."})

        # Optional: File size check (max 10MB)
        if file.size > 10 * 1024 * 1024:
            messages.error(request, "File is too large. Maximum size allowed is 10MB.")
            return render(request, "index.html", {"error": "File is too large."})

        # Create image record
        uploaded = UploadedImage.objects.create(
            image=file,
            filename=file.name,
            uploaded_by=request.user if request.user.is_authenticated else None
        )

        image_path = uploaded.image.path

        try:
            metadata_result = process_metadata(image_path)
            summary = metadata_result["summary"]

            MetadataAnalysis.objects.create(
                image=uploaded,
                camera_make=summary.get("make") or "",
                camera_model=summary.get("model") or "",
                software=summary.get("software") or "",
                date_original=str(summary.get("date_original") or ""),
                modify_date=str(summary.get("modify_date") or ""),
                exposure_time=str(summary.get("exposure_time") or ""),
                f_number=safe_float(summary.get("f_number")),
                iso=safe_int(summary.get("iso")),
                focal_length=safe_float(summary.get("focal_length")),
                gps_latitude=safe_float(summary.get("gps_latitude")),
                gps_longitude=safe_float(summary.get("gps_longitude")),
                resolution_width=safe_int(summary.get("resolution_width")),
                resolution_height=safe_int(summary.get("resolution_height")),
                c2pa_present=summary.get("c2pa_present") or False,
                exif_full=summary.get("exif_full"),
                ai_tool=summary.get("ai_tool") or "",
                verdict=metadata_result["verdict"]
            )

            messages.success(request, "Image uploaded successfully!")
            return redirect("image_detail", hash_id=encode_id(uploaded.id))

        except Exception as e:
            uploaded.delete()  # cleanup if processing fails
            messages.error(request, f"Error processing image: {str(e)}")
            return render(request, "index.html", {"error": "Error processing the image."})

    # GET request
    if request.user.is_authenticated:
        images = UploadedImage.objects.filter(uploaded_by=request.user).order_by("-uploaded_at")[:3]
        # Prefetch related data
        images = images.prefetch_related(
            Prefetch('final_verdict', queryset=FinalVerdict.objects.order_by('-created_at')),
            Prefetch('ai_detection', queryset=AIDetection.objects.order_by('-analyzed_at')),
            Prefetch('human_edit_detection', queryset=HumanEditDetection.objects.order_by('-analyzed_at'))
        )
        
        # Calculate AI confidence and human edit percentage for each image
        for img in images:
            # Calculate AI confidence (always show AI percentage)
            ai_detection = img.ai_detection.first()
            if ai_detection and ai_detection.confidence > 0:
                if 'AI' in ai_detection.label.upper():
                    img.ai_confidence_percentage = ai_detection.confidence
                else:
                    # If it's 'Real', show inverse as AI confidence
                    img.ai_confidence_percentage = 100 - ai_detection.confidence
            else:
                img.ai_confidence_percentage = None
            
            # Calculate human edit percentage (convert decimal to percentage)
            human_edit = img.human_edit_detection.first()
            if human_edit and human_edit.detection_score > 0:
                # Convert from decimal (0.35) to percentage (35.0)
                img.human_edit_percentage = human_edit.detection_score * 100
            else:
                img.human_edit_percentage = None
    else:
        images = []

    return render(request, "index.html", {"images": images})



def get_image_characteristics(image, ai_detection, human_detection, metadata, heatmap, verdict):
    """
    Generate logical characteristics for an image based on available data.
    Returns a list of characteristics with explanations and metadata support.
    """
    characteristics = []
    
    # Helper to add characteristic with metadata if available
    def add_char(characteristic, metadata_info=None, is_positive=False, is_negative=False):
        characteristics.append({
            'text': characteristic,
            'metadata': metadata_info,
            'is_positive': is_positive,
            'is_negative': is_negative
        })
    
    verdict_text = verdict.verdict if verdict else ""
    ai_confidence = ai_detection.confidence if ai_detection else 0
    human_score = human_detection.detection_score if human_detection else 0
    c2pa_present = metadata.c2pa_present if metadata else False
    
    # ============================================================
    # AI GENERATED IMAGE CHARACTERISTICS
    # ============================================================
    if "AI" in verdict_text.upper() or c2pa_present or ai_confidence > 70:
        
        # 1. C2PA Detection
        if c2pa_present:
            add_char(
                "C2PA Content Credentials detected - indicates AI generation or editing tool usage",
                metadata_info="C2PA manifest present in metadata",
                is_negative=True
            )
        else:
            add_char(
                "Synthetic texture patterns detected",
                metadata_info=None,
                is_negative=True
            )
        
        # 2. AI Confidence
        if ai_confidence > 0:
            add_char(
                f"GAN/diffusion frequency artifacts present ({ai_confidence:.1f}% confidence)",
                metadata_info=f"AI Detection Model: {ai_detection.model_name if ai_detection else 'ResNet50'}",
                is_negative=True
            )
        
        # 3. Camera Metadata Check
        if not metadata or not (metadata.camera_make or metadata.camera_model):
            add_char(
                "No camera metadata detected - typical for AI-generated content",
                metadata_info="Camera make/model missing from EXIF",
                is_negative=True
            )
        else:
            add_char(
                "Missing authentic camera sensor noise pattern",
                metadata_info=f"Camera: {metadata.camera_make} {metadata.camera_model} would have distinct noise pattern",
                is_negative=True
            )
        
        # 4. AI Tool Detection
        if metadata and metadata.ai_tool:
            add_char(
                f"AI generation tool identified: {metadata.ai_tool}",
                metadata_info=f"Tool signature found in metadata",
                is_negative=True
            )
        
        # 5. Software Mismatch
        if metadata and metadata.software and "AI" in metadata.software.upper():
            add_char(
                f"AI software detected: {metadata.software}",
                metadata_info="Software field indicates AI generation",
                is_negative=True
            )
        
        # 6. GPS Data
        if not metadata or not (metadata.gps_latitude and metadata.gps_longitude):
            add_char(
                "No GPS location data (unusual for authentic photos)",
                metadata_info="GPS coordinates missing",
                is_negative=True
            )
        
        # 7. Timestamp Inconsistency
        if metadata and metadata.date_original and metadata.modify_date:
            if metadata.date_original != metadata.modify_date:
                add_char(
                    "Timestamp inconsistency detected between capture and modification",
                    metadata_info=f"Original: {metadata.date_original}, Modified: {metadata.modify_date}",
                    is_negative=True
                )
        
        # 8. Heatmap Analysis
        if heatmap:
            if heatmap.severity_level == "HIGH" or heatmap.severity_level == "VERY_HIGH":
                add_char(
                    f"Widespread artifacts detected (Severity: {heatmap.severity_level})",
                    metadata_info=f"{heatmap.edited_percentage:.1f}% of image shows artifacts",
                    is_negative=True
                )
            if heatmap.primary_region:
                add_char(
                    f"Artifacts concentrated in {heatmap.primary_region} region",
                    metadata_info=f"Primary affected area with score {heatmap.max_edit_score:.2f}",
                    is_negative=True
                )
        
        # 9. Missing optical characteristics
        if not metadata or not metadata.f_number:
            add_char(
                "Lack of optical lens characteristics",
                metadata_info="No aperture/focal length data",
                is_negative=True
            )
        
        # 10. Unnatural color distribution
        add_char(
            "Unnatural color distribution patterns",
            metadata_info=None,
            is_negative=True
        )
    
    # ============================================================
    # HUMAN EDITED IMAGE CHARACTERISTICS
    # ============================================================
    if "EDIT" in verdict_text.upper() or human_score > 0.35 or (heatmap and heatmap.edited_percentage > 20):
        
        # 1. Localized Manipulation
        if heatmap:
            add_char(
                f"Localized manipulation regions detected ({heatmap.edit_zones_count} zones)",
                metadata_info=f"Edit score: {human_score:.1%}, Edited: {heatmap.edited_percentage:.1f}%",
                is_negative=True
            )
        
        # 2. Primary Edit Region
        if heatmap and heatmap.primary_region:
            add_char(
                f"Object insertion or modification in {heatmap.primary_region} area",
                metadata_info=f"Region edit score: {heatmap.max_edit_score:.2f}",
                is_negative=True
            )
        
        # 3. Secondary Region
        if heatmap and heatmap.secondary_region:
            add_char(
                f"Additional edits detected in {heatmap.secondary_region}",
                metadata_info="Multiple manipulation zones",
                is_negative=True
            )
        
        # 4. Software Detection
        if metadata and metadata.software:
            editing_software = ["Photoshop", "GIMP", "Lightroom", "Affinity", "Pixelmator", "Paint", "Editor"]
            if any(sw in metadata.software for sw in editing_software):
                add_char(
                    f"Editing software detected: {metadata.software}",
                    metadata_info="Software metadata indicates post-processing",
                    is_negative=True
                )
        
        # 5. Copy-move evidence
        if heatmap and heatmap.edit_zones_count >= 2:
            add_char(
                "Possible copy-move or clone stamp artifacts",
                metadata_info=f"Multiple similar edit zones detected",
                is_negative=True
            )
        
        # 6. Compression mismatch
        if heatmap and heatmap.avg_edit_score > 0.3:
            add_char(
                "Compression artifacts mismatch",
                metadata_info=f"Average edit score: {heatmap.avg_edit_score:.2f}",
                is_negative=True
            )
        
        # 7. Timestamp inconsistency
        if metadata and metadata.date_original and metadata.modify_date:
            if metadata.date_original != metadata.modify_date:
                add_char(
                    "Modification date differs from original capture",
                    metadata_info=f"Original: {metadata.date_original}, Modified: {metadata.modify_date}",
                    is_negative=True
                )
        
        # 8. Splice boundaries
        if heatmap and heatmap.grid_scores:
            import json
            scores = heatmap.grid_scores
            if isinstance(scores, str):
                scores = json.loads(scores)
            # Check for edge-based manipulations
            edge_regions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
            if any(scores.get(region, 0) > 0.3 for region in edge_regions):
                add_char(
                    "Splice boundaries visible at image edges",
                    metadata_info="Higher edit scores detected at corners",
                    is_negative=True
                )
        
        # 9. Resolution inconsistency
        if heatmap and heatmap.max_edit_score > 0.5:
            add_char(
                "Resolution inconsistencies across image",
                metadata_info="Edited regions show different compression patterns",
                is_negative=True
            )
    
    # ============================================================
    # MIXED (AI + HUMAN EDIT) CHARACTERISTICS
    # ============================================================
    if ("AI" in verdict_text.upper() and "EDIT" in verdict_text.upper()) or (ai_confidence > 60 and human_score > 0.3):
        
        add_char(
            f"Base AI generation detected ({ai_confidence:.1f}%) with human modifications ({human_score:.1%})",
            metadata_info="Combined AI + manual editing detected",
            is_negative=True
        )
        
        if heatmap and heatmap.edit_zones_count > 1:
            add_char(
                "Multiple generation and editing layers detected",
                metadata_info=f"{heatmap.edit_zones_count} separate manipulation zones",
                is_negative=True
            )
        
        if heatmap and heatmap.region_scores:
            add_char(
                "Inconsistent noise patterns (AI base + human corrections)",
                metadata_info="Different regions show different generation characteristics",
                is_negative=True
            )
        
        if metadata and metadata.software and metadata.ai_tool:
            add_char(
                f"Mixed metadata: {metadata.ai_tool} + {metadata.software}",
                metadata_info="Both AI generation and editing tools detected",
                is_negative=True
            )
    
    # ============================================================
    # AUTHENTIC IMAGE CHARACTERISTICS
    # ============================================================
    elif "AUTHENTIC" in verdict_text.upper() or (ai_confidence < 40 and human_score < 0.25):
        
        # 1. Camera Metadata
        if metadata and (metadata.camera_make or metadata.camera_model):
            add_char(
                f"Natural texture patterns from authentic camera capture",
                metadata_info=f"Camera: {metadata.camera_make} {metadata.camera_model}",
                is_positive=True
            )
        else:
            add_char(
                "Natural texture patterns throughout",
                metadata_info=None,
                is_positive=True
            )
        
        # 2. Consistent Lighting
        add_char(
            "Consistent lighting and shadow geometry",
            metadata_info="No lighting inconsistencies detected",
            is_positive=True
        )
        
        # 3. Sensor Noise
        if metadata and (metadata.camera_make or metadata.camera_model):
            add_char(
                "Camera sensor noise pattern present",
                metadata_info=f"{metadata.camera_make} sensors have characteristic noise patterns",
                is_positive=True
            )
        
        # 4. Camera Settings
        if metadata and (metadata.f_number or metadata.iso or metadata.focal_length):
            settings = []
            if metadata.f_number: settings.append(f"f/{metadata.f_number}")
            if metadata.iso: settings.append(f"ISO {metadata.iso}")
            if metadata.focal_length: settings.append(f"{metadata.focal_length}mm")
            add_char(
                f"Authentic camera settings: {', '.join(settings)}",
                metadata_info="Natural optical characteristics",
                is_positive=True
            )
        
        # 5. GPS Data
        if metadata and metadata.gps_latitude and metadata.gps_longitude:
            add_char(
                f"GPS location data available",
                metadata_info=f"Coordinates: {metadata.gps_latitude}, {metadata.gps_longitude}",
                is_positive=True
            )
        
        # 6. Natural Edges
        add_char(
            "Natural edge transitions without artifacts",
            metadata_info=None,
            is_positive=True
        )
        
        # 7. No Manipulation
        if human_score < 0.2:
            add_char(
                f"No manipulation artifacts detected (edit score: {human_score:.1%})",
                metadata_info="Clean authentic image",
                is_positive=True
            )
        
        # 8. Color Gradation
        add_char(
            "Natural color gradation and tone curves",
            metadata_info=None,
            is_positive=True
        )
        
        # 9. Physical Plausibility
        if metadata and (metadata.f_number and metadata.focal_length):
            add_char(
                "Physically plausible lighting and depth of field",
                metadata_info=f"Aperture f/{metadata.f_number} at {metadata.focal_length}mm creates realistic blur",
                is_positive=True
            )
        
        # 10. Consistent Timestamps
        if metadata and metadata.date_original:
            add_char(
                f"Consistent capture timestamp: {metadata.date_original}",
                metadata_info="No date manipulation detected",
                is_positive=True
            )
    
    return characteristics

@property
def display_url(self):
    """Returns a viewable Google Drive direct URL or local fallback."""
    if self.drive_file_id:
        return (
            f"https://drive.google.com/uc?export=view&id={self.drive_file_id}"
        )
    elif self.image:
        return self.image.url
    return ""

# The rest of your views remain unchanged
@login_required(login_url='login')
def image_detail(request, hash_id):
    real_id = decode_id(hash_id)
    if real_id is None:
        from django.http import Http404
        raise Http404("Image not found")
    image = get_object_or_404(UploadedImage, id=real_id)
    metadata = MetadataAnalysis.objects.filter(image=image).first()
    ai = AIDetection.objects.filter(image=image).first()
    human = HumanEditDetection.objects.filter(image=image).first()
    verdict = FinalVerdict.objects.filter(image=image).first()
    heatmap_analysis = HeatmapAnalysis.objects.filter(image=image).first()
    existing_reports = GeneratedReport.objects.filter(image=image).order_by('-generated_at')

    ai_percentage = 0
    human_percentage = 0
    dashoffset_ai = 339.29
    dashoffset_human = 339.29

    if ai:
        ai_percentage = ai.confidence if ai.label == "AI-generated" else (100 - ai.confidence)
        dashoffset_ai = round(339.29 * (1 - ai_percentage / 100), 2)

    if human:
        human_percentage = round(human.detection_score * 100, 2)
        dashoffset_human = round(339.29 * (1 - human_percentage / 100), 2)
    
    # Generate characteristics - FIXED variable names
    characteristics = get_image_characteristics(
        image, ai, human, metadata, heatmap_analysis, verdict
    )

    return render(request, "image_detail.html", {
        "image": image,
        "metadata": metadata,
        "ai": ai,
        "human": human,
        "verdict": verdict,
        "heatmap_analysis": heatmap_analysis,
        "existing_reports": existing_reports,
        "ai_percentage": round(ai_percentage, 1),
        "human_percentage": human_percentage,
        "dashoffset_ai": dashoffset_ai,
        "dashoffset_human": dashoffset_human,
        "characteristics": characteristics,
        # Added missing variables for template
        "ai_detection": ai,
        "human_detection": human,
        "heatmap": heatmap_analysis,
        "c2pa_present": metadata.c2pa_present if metadata else False,
    })


def get_image_ids_for_navigation(request):
    """Get all image IDs for current user for navigation"""
    
    # Base queryset based on user role (same as all_images view)
    if request.user.is_superuser:
        images = UploadedImage.objects.all()
    elif request.user.is_staff:
        images = UploadedImage.objects.filter(
            Q(uploaded_by=request.user) |
            Q(uploaded_by__is_staff=False, uploaded_by__is_superuser=False)
        ).distinct()
    else:
        images = UploadedImage.objects.filter(uploaded_by=request.user)
    
    # Apply current filter if any
    filter_type = request.GET.get('filter', 'all')
    
    if filter_type == 'authentic':
        images = images.filter(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='AUTHENTIC'
            ))
        )
    elif filter_type == 'ai_generated':
        images = images.filter(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='AI'
            )) | Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='GENERATED'
            ))
        )
    elif filter_type == 'human_edited':
        images = images.filter(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='EDITED'
            )) | Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='MANIPULATED'
            ))
        )
    elif filter_type == 'not_analyzed':
        images = images.filter(
            ~Exists(FinalVerdict.objects.filter(image=OuterRef('pk')))
        )
    
    # Apply current sorting
    sort_by = request.GET.get('sort', 'newest')
    if sort_by == 'name_asc':
        images = images.order_by('filename')
    elif sort_by == 'name_desc':
        images = images.order_by('-filename')
    elif sort_by == 'oldest':
        images = images.order_by('uploaded_at')
    else:
        images = images.order_by('-uploaded_at')
    
    # Get hashed IDs
    image_ids = [encode_id(img.id) for img in images]
    
    return JsonResponse({'image_ids': image_ids})


def calculate_overall_confidence(ai_label, ai_confidence, human_score, verdict):
    """Calculate overall confidence based on verdict type"""
    
    # Convert AI confidence based on label
    # If label is "Real Image", then AI confidence is actually confidence in REAL, not AI
    # So we need to invert it when we care about AI probability
    if ai_label == "Real Image":
        # Confidence that it's REAL - so AI probability is the inverse
        ai_probability = 100 - ai_confidence
        real_confidence = ai_confidence  # How sure it's real
    else:
        # Confidence that it's AI-generated
        ai_probability = ai_confidence
        real_confidence = 100 - ai_confidence  # How sure it's real (inverse)
    
    human_edit_confidence = human_score * 100
    human_authentic_confidence = (1 - human_score) * 100
    
    if verdict == "AUTHENTIC IMAGE":
        # For authentic: confidence is how sure it's NOT manipulated
        # Use real confidence from AI model and human authentic confidence
        overall = (real_confidence + human_authentic_confidence) / 2
        
    elif verdict == "AI Generated Image":
        # For AI: confidence is how sure it's AI-generated
        overall = ai_probability
        
    elif verdict == "Human Edited Image":
        # For Human Edit: confidence is the human edit score
        overall = human_edit_confidence
        
    elif verdict == "AI Generated & Human Edited":
        # For both: average of AI probability and human edit score
        overall = (ai_probability + human_edit_confidence) / 2
        
    else:
        overall = 0
        
    # Round and ensure within 0-100 range
    return max(0.1, min(99.9, round(overall)))

@login_required(login_url='login')
def run_analysis(request, hash_id):
    """Main analysis view that coordinates both detections"""
    real_id = decode_id(hash_id)
    if real_id is None:
        raise Http404("Image not found")
    
    image = get_object_or_404(UploadedImage, id=real_id)
    image_path = image.image.path
    
    # AI Detection - Always run or check if needed
    try:
        ai_result = process_ai_image(image_path)
        AIDetection.objects.update_or_create(
            image=image,
            defaults={"label": ai_result["label"], "confidence": ai_result["confidence"]}
        )
        print(f"✅ AI Detection saved for {image.filename}")
    except Exception as e:
        print(f"AI Detection failed: {e}")
        messages.error(request, f"AI Detection failed: {str(e)}")
        return redirect("image_detail", hash_id=encode_id(image.id))
    
    # Human Edit Detection - Check if already exists
    existing_human_edit = HumanEditDetection.objects.filter(image=image).first()
    
    if existing_human_edit:
        # Use existing results - skip heavy model
        print(f"⏭️ Using existing Human Edit Detection for {image.filename} (skipping heavy model)")
        human_score = existing_human_edit.detection_score
        print(f"📊 Retrieved human_score: {human_score}")
        
        # Check if heatmap analysis exists, if not, create it
        if not HeatmapAnalysis.objects.filter(image=image).exists():
            print(f"📊 Creating heatmap analysis from existing data for {image.filename}")
            heatmap_full_path = os.path.join(settings.MEDIA_ROOT, existing_human_edit.heatmap.name)
            analyze_result = analyze_heatmap_from_path(image.id, heatmap_full_path, human_score)
            if analyze_result:
                print(f"✅ Heatmap analysis saved for {image.filename}")
            else:
                print(f"⚠️ Heatmap analysis failed for {image.filename}")
    else:
        # Run heavy model - no existing results
        try:
            print(f"🔄 Running Human Edit Detection (heavy model) for {image.filename}")
            human_result = process_human_edit(image_path)
            human_edit, created = HumanEditDetection.objects.update_or_create(
                image=image,
                defaults={
                    "heatmap": human_result["heatmap"],
                    "overlay": human_result["overlay"],
                    "detection_score": human_result["detection"]
                }
            )
            print(f"✅ Human Edit Detection saved for {image.filename}")
            human_score = human_result["detection"]
            print(f"📊 New human_score: {human_score}")
            
            # Analyze heatmap
            heatmap_full_path = os.path.join(settings.MEDIA_ROOT, human_result["heatmap"])
            print(f"📊 Analyzing heatmap at: {heatmap_full_path}")
            analyze_result = analyze_heatmap_from_path(image.id, heatmap_full_path, human_score)
            if analyze_result:
                print(f"✅ Heatmap analysis saved for {image.filename}")
            else:
                print(f"⚠️ Heatmap analysis failed for {image.filename}")
                
        except Exception as e:
            print(f"Human Edit Detection failed: {e}")
            messages.error(request, f"Human Edit Detection failed: {str(e)}")
            return redirect("image_detail", hash_id=encode_id(image.id))
    
    # Make sure human_score is defined (fallback)
    if 'human_score' not in locals():
        print(f"⚠️ Warning: human_score not defined, using default 0.0")
        human_score = 0.0
    
    # Final Verdict Logic (using existing or new scores)
    ai_label = ai_result["label"]
    ai_confidence = ai_result["confidence"]
    is_ai = ai_label == "AI-generated"
    
    print(f"📊 AI Label: {ai_label}, AI Confidence: {ai_confidence}%")
    print(f"📊 Human Score: {human_score}")
    
    # Get C2PA status from metadata
    metadata = MetadataAnalysis.objects.filter(image=image).first()
    c2pa_present = metadata.c2pa_present if metadata else False
    
    # Determine verdict based on scores
    if c2pa_present:
        verdict_text = "AI Generated Image"
    elif human_score > 0.35:
        if is_ai and ai_confidence >= 96:
            verdict_text = "AI Generated Image"
        elif is_ai and ai_confidence >= 80:
            verdict_text = "AI Generated & Human Edited"           
        else:
            verdict_text = "Human Edited Image"
    elif is_ai and ai_confidence >= 80:
        verdict_text = "AI Generated Image"
    else:
        verdict_text = "AUTHENTIC IMAGE"
    
    print(f"📊 Final Verdict: {verdict_text}")
    
    # Calculate overall confidence
    # If C2PA is present, set confidence to 99.9%
    if c2pa_present:
        overall_confidence = 99.9
        print(f"📊 C2PA Present - Setting confidence to: {overall_confidence}%")
    else:
        overall_confidence = calculate_overall_confidence(ai_label, ai_confidence, human_score, verdict_text)
        print(f"📊 Overall Confidence: {overall_confidence}%")
    
    # Also update AI Detection confidence if C2PA is present
    if c2pa_present:
        # Update AI detection confidence to 99.9 to reflect C2PA finding
        AIDetection.objects.update_or_create(
            image=image,
            defaults={
                "label": "AI-generated",  # Force label to AI-generated
                "confidence": 99.9
            }
        )
        print(f"📊 Updated AI Detection to 99.9% due to C2PA presence")
    
    FinalVerdict.objects.update_or_create(
        image=image,
        defaults={
            "verdict": verdict_text,
            "explanation": "Auto-generated from ResNet50 + TruFor analysis" + (" (C2PA detected)" if c2pa_present else ""),
            "confidence_score": overall_confidence
        }
    )
    
    messages.success(request, f"Analysis complete! Verdict: {verdict_text} (Confidence: {overall_confidence}%)")
    return redirect("image_detail", hash_id=encode_id(image.id))


#  #===================== RUN AI ANALYSIS FOR ALL IMAGES ==================
@login_required(login_url='login')
def run_analysis_for_all_images(request):
    """Run AI analysis for all images without human edit detection"""
    
    # Get all images based on user role
    if request.user.is_superuser:
        images = UploadedImage.objects.all()
    elif request.user.is_staff:
        images = UploadedImage.objects.filter(
            Q(uploaded_by=request.user) |
            Q(uploaded_by__is_staff=False) & ~Q(uploaded_by__is_superuser=True)
        ).distinct()
    else:
        images = UploadedImage.objects.filter(uploaded_by=request.user)
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    c2pa_count = 0
    
    for image in images:
        print(f"\n--- Processing: {image.filename} ---")
        image_path = image.image.path
        
        try:
            # Run AI Detection only
            ai_result = process_ai_image(image_path)
            AIDetection.objects.update_or_create(
                image=image,
                defaults={"label": ai_result["label"], "confidence": ai_result["confidence"]}
            )
            print(f"✅ AI Detection saved for {image.filename}")
            
        except Exception as e:
            print(f"❌ AI Detection failed for {image.filename}: {e}")
            error_count += 1
            continue
        
        # Get existing human edit detection (don't run new one)
        existing_human_edit = HumanEditDetection.objects.filter(image=image).first()
        
        if existing_human_edit:
            human_score = existing_human_edit.detection_score
            print(f"📊 Using existing human_score: {human_score}")
            
            # Create heatmap analysis if not exists
            if not HeatmapAnalysis.objects.filter(image=image).exists():
                heatmap_full_path = os.path.join(settings.MEDIA_ROOT, existing_human_edit.heatmap.name)
                analyze_result = analyze_heatmap_from_path(image.id, heatmap_full_path, human_score)
                if analyze_result:
                    print(f"✅ Heatmap analysis saved for {image.filename}")
        else:
            human_score = 0.0
            print(f"⚠️ No human edit data for {image.filename}")
            skipped_count += 1
        
        # Final Verdict Logic
        ai_label = ai_result["label"]
        ai_confidence = ai_result["confidence"]
        is_ai = ai_label == "AI-generated"
        
        # Get C2PA status from metadata
        metadata = MetadataAnalysis.objects.filter(image=image).first()
        c2pa_present = metadata.c2pa_present if metadata else False
        
        # Determine verdict
        if c2pa_present:
            verdict_text = "AI Generated Image"
        elif human_score > 0.35:
            if is_ai and ai_confidence >= 96:
                verdict_text = "AI Generated Image"
            elif is_ai and ai_confidence >= 80:
                verdict_text = "AI Generated & Human Edited"           
            else:
                verdict_text = "Human Edited Image"
        elif is_ai and ai_confidence >= 80:
            verdict_text = "AI Generated Image"
        else:
            verdict_text = "AUTHENTIC IMAGE"
        
        # Calculate overall confidence
        # If C2PA is present, set confidence to 99.9%
        if c2pa_present:
            overall_confidence = 99.9
            c2pa_count += 1
            # Update AI detection to reflect C2PA finding
            AIDetection.objects.update_or_create(
                image=image,
                defaults={
                    "label": "AI-generated",
                    "confidence": 99.9
                }
            )
            print(f"📊 C2PA Present - Setting confidence to 99.9% for {image.filename}")
        else:
            overall_confidence = calculate_overall_confidence(ai_label, ai_confidence, human_score, verdict_text)
        
        # Save final verdict
        FinalVerdict.objects.update_or_create(
            image=image,
            defaults={
                "verdict": verdict_text,
                "explanation": "Auto-generated from ResNet50 analysis (batch)" + (" (C2PA detected)" if c2pa_present else ""),
                "confidence_score": overall_confidence
            }
        )
        
        print(f"📊 Final Verdict for {image.filename}: {verdict_text} (Confidence: {overall_confidence}%)")
        success_count += 1
    
    # Summary message
    summary = f"Batch analysis complete! Processed: {success_count} images, Errors: {error_count}, No human edit data: {skipped_count}, C2PA detected: {c2pa_count}"
    messages.success(request, summary)
    print(f"\n{'='*50}\n{summary}\n{'='*50}")
    
    return redirect("all_images")


#===================== FOR TESTING ONLY ===========================
def get_pagination_mode(request):
    """Get and set pagination mode from session"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            request.session['use_infinite_scroll'] = data.get('use_infinite_scroll', False)
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
    # GET request - return current mode
    return JsonResponse({
        'use_infinite_scroll': request.session.get('use_infinite_scroll', False)
    })


@login_required(login_url='login')
def all_images(request):
    """
    View all images with role-based access control:
    - Superuser: Can view ALL images (including other superusers)
    - Staff: Can view their own images + all non-staff user images
    - Regular users: Can only view their own images
    """
    
    # Base queryset based on user role
    if request.user.is_superuser:
        all_images = UploadedImage.objects.all()
        print(f"👑 Superuser {request.user.username} - viewing ALL images")
        
    elif request.user.is_staff:
        all_images = UploadedImage.objects.filter(
            Q(uploaded_by=request.user) |
            (Q(uploaded_by__is_staff=False) & ~Q(uploaded_by__is_superuser=True))
        ).distinct()
        print(f"🛡️ Staff {request.user.username} - viewing own + non-staff user images")
        
    else:
        all_images = UploadedImage.objects.filter(uploaded_by=request.user)
        print(f"👤 Regular user {request.user.username} - viewing only their own images")
    
    # ========== CALCULATE COUNTS FOR FILTER BADGES ==========
    from django.db.models import Count, Q, Exists, OuterRef
    
    # Base queryset for counts (without filtering)
    if request.user.is_superuser:
        base_queryset = UploadedImage.objects.all()
    elif request.user.is_staff:
        base_queryset = UploadedImage.objects.filter(
            Q(uploaded_by=request.user) |
            (Q(uploaded_by__is_staff=False) & ~Q(uploaded_by__is_superuser=True))
        ).distinct()
    else:
        base_queryset = UploadedImage.objects.filter(uploaded_by=request.user)
    
    # Efficient single-query aggregation for all counts
    from django.db.models import Value, IntegerField
    from django.db.models.functions import Coalesce
    
    counts_result = base_queryset.aggregate(
        total_all=Count('id'),
        total_authentic=Count('id', filter=Exists(
            FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='AUTHENTIC'
            )
        )),
        total_ai_generated=Count('id', filter=(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='AI'
            )) | Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='GENERATED'
            ))
        )),
        total_human_edited=Count('id', filter=(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='EDITED'
            )) | Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='MANIPULATED'
            ))
        )),
        total_not_analyzed=Count('id', filter=~Exists(
            FinalVerdict.objects.filter(image=OuterRef('pk'))
        )),
    )
    
    # Format the counts dictionary with percentages
    total_counts = {
        'all': counts_result['total_all'],
        'authentic': counts_result['total_authentic'],
        'ai_generated': counts_result['total_ai_generated'],
        'human_edited': counts_result['total_human_edited'],
        'not_analyzed': counts_result['total_not_analyzed'],
    }
    
    # Calculate percentages
    total_all = total_counts['all']
    if total_all > 0:
        total_counts['authentic_percent'] = round((total_counts['authentic'] / total_all) * 100, 1)
        total_counts['ai_generated_percent'] = round((total_counts['ai_generated'] / total_all) * 100, 1)
        total_counts['human_edited_percent'] = round((total_counts['human_edited'] / total_all) * 100, 1)
        total_counts['not_analyzed_percent'] = round((total_counts['not_analyzed'] / total_all) * 100, 1)
    else:
        total_counts['authentic_percent'] = 0
        total_counts['ai_generated_percent'] = 0
        total_counts['human_edited_percent'] = 0
        total_counts['not_analyzed_percent'] = 0
    
    print(f"📊 Image Counts - All: {total_counts['all']}, Authentic: {total_counts['authentic']} ({total_counts['authentic_percent']}%), "
          f"AI: {total_counts['ai_generated']} ({total_counts['ai_generated_percent']}%), "
          f"Human Edited: {total_counts['human_edited']} ({total_counts['human_edited_percent']}%), "
          f"Not Analyzed: {total_counts['not_analyzed']} ({total_counts['not_analyzed_percent']}%)")
    
    # Prefetch related data for performance
    all_images = all_images.prefetch_related(
        Prefetch('final_verdict', queryset=FinalVerdict.objects.order_by('-created_at')),
        Prefetch('ai_detection', queryset=AIDetection.objects.order_by('-analyzed_at')),
        Prefetch('human_edit_detection', queryset=HumanEditDetection.objects.order_by('-analyzed_at'))
    )
    
    # Get sorting parameter
    sort_by = request.GET.get('sort', 'newest')
    
    # Apply sorting
    if sort_by == 'name_asc':
        all_images = all_images.order_by('filename')
    elif sort_by == 'name_desc':
        all_images = all_images.order_by('-filename')
    elif sort_by == 'oldest':
        all_images = all_images.order_by('uploaded_at')
    elif sort_by == 'newest':
        all_images = all_images.order_by('-uploaded_at')
    else:
        all_images = all_images.order_by('-uploaded_at')
    
    # Apply filtering based on verdict
    filter_type = request.GET.get('filter', 'all')
    
    if filter_type == 'authentic':
        all_images = all_images.filter(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='AUTHENTIC'
            ))
        )
    elif filter_type == 'ai_generated':
        all_images = all_images.filter(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='AI'
            )) | Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='GENERATED'
            ))
        )
    elif filter_type == 'human_edited':
        all_images = all_images.filter(
            Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='EDITED'
            )) | Exists(FinalVerdict.objects.filter(
                image=OuterRef('pk'),
                verdict__icontains='MANIPULATED'
            ))
        )
    elif filter_type == 'not_analyzed':
        all_images = all_images.filter(
            ~Exists(FinalVerdict.objects.filter(image=OuterRef('pk')))
        )
    
    # Annotate scores for sorting
    from django.db.models import Case, When, Value, FloatField, F
    from django.db.models.functions import Coalesce
    
    all_images = all_images.annotate(
        ai_score_annotated=Coalesce(
            Case(
                When(ai_detection__label='AI-generated', then=F('ai_detection__confidence')),
                When(ai_detection__label='Real', then=100 - F('ai_detection__confidence')),
                default=Value(0.0),
                output_field=FloatField()
            ),
            Value(0.0)
        ),
        edit_score_annotated=Coalesce(
            F('human_edit_detection__detection_score') * 100,
            Value(0.0),
            output_field=FloatField()
        )
    )
    
    # Apply score-based sorting
    if sort_by == 'ai_score_high':
        all_images = all_images.order_by('-ai_score_annotated')
    elif sort_by == 'ai_score_low':
        all_images = all_images.order_by('ai_score_annotated')
    elif sort_by == 'edit_score_high':
        all_images = all_images.order_by('-edit_score_annotated')
    elif sort_by == 'edit_score_low':
        all_images = all_images.order_by('edit_score_annotated')
    
    # Get pagination mode from session
    use_infinite_scroll = request.session.get('use_infinite_scroll', False)
    
    # ========== AJAX INFINITE SCROLL REQUEST ==========
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' and request.GET.get('infinite'):
        page = int(request.GET.get('page', 1))
        items_per_page = 9
        
        # Calculate offset
        start = (page - 1) * items_per_page
        end = start + items_per_page
        
        # Get total count and slice
        total_images = all_images.count()
        sliced_images = all_images[start:end]
        
        images_data = []
        for img in sliced_images:
            # Get AI score
            ai_detection_obj = img.ai_detection.first() if hasattr(img, 'ai_detection') else None
            if ai_detection_obj and ai_detection_obj.confidence > 0:
                if 'AI' in ai_detection_obj.label.upper():
                    ai_score = float(ai_detection_obj.confidence)
                else:
                    ai_score = float(100 - ai_detection_obj.confidence)
            else:
                ai_score = None
            
            # Get human edit score
            human_edit_obj = img.human_edit_detection.first() if hasattr(img, 'human_edit_detection') else None
            if human_edit_obj and human_edit_obj.detection_score > 0:
                edit_score = float(human_edit_obj.detection_score * 100)
            else:
                edit_score = None
            
            # Get verdict
            verdict_obj = img.final_verdict.first() if hasattr(img, 'final_verdict') else None
            
            # Calculate uploaded ago
            from django.utils.timesince import timesince
            uploaded_ago = timesince(img.uploaded_at)
            
            images_data.append({
                'id': img.id,
                'hashid': encode_id(img.id),
                'filename': img.filename,
                'image_url': img.image.url,
                'uploaded_ago': uploaded_ago,
                'verdict': verdict_obj.verdict if verdict_obj else None,
                'ai_score': ai_score,
                'edit_score': edit_score,
                'has_analysis': bool(verdict_obj),
                'uploaded_by': img.uploaded_by.username if img.uploaded_by else None,
                'uploader_role' : get_uploader_role(img.uploaded_by)
            })
        
        # Calculate if there are more pages
        has_next = end < total_images
        
        return JsonResponse({
            'success': True,
            'images': images_data,
            'has_next': has_next,
            'current_page': page,
            'total': total_images
        })
    
    # ========== REGULAR PAGINATION (non-AJAX) ==========
    paginator = Paginator(all_images, 9)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    # Calculate AI confidence and human edit percentage for each image
    for img in page_obj:
        ai_detection_obj = img.ai_detection.first() if hasattr(img, 'ai_detection') else None
        if ai_detection_obj and ai_detection_obj.confidence > 0:
            if 'AI' in ai_detection_obj.label.upper():
                img.ai_confidence_percentage = ai_detection_obj.confidence
            else:
                img.ai_confidence_percentage = 100 - ai_detection_obj.confidence
        else:
            img.ai_confidence_percentage = None
        
        human_edit_obj = img.human_edit_detection.first() if hasattr(img, 'human_edit_detection') else None
        if human_edit_obj and human_edit_obj.detection_score > 0:
            img.human_edit_percentage = human_edit_obj.detection_score * 100
        else:
            img.human_edit_percentage = None
        
        if request.user.is_superuser or request.user.is_staff:
            img.uploaded_by_username = img.uploaded_by.username if img.uploaded_by else "Unknown"
            img.uploader_role = "Superuser" if img.uploaded_by and img.uploaded_by.is_superuser else "Staff" if img.uploaded_by and img.uploaded_by.is_staff else "User"
    
    # Add role info to template context
    role_info = {
        'is_superuser': request.user.is_superuser,
        'is_staff': request.user.is_staff,
        'role': 'superuser' if request.user.is_superuser else 'staff' if request.user.is_staff else 'user'
    }
    
    return render(request, 'all_images.html', {
        'images': page_obj,
        'page_obj': page_obj,
        'paginator': paginator,
        'use_infinite_scroll': use_infinite_scroll,
        'role_info': role_info,
        'filter_type': filter_type,
        'current_sort': sort_by,
        'total_counts': total_counts,  # Add counts to template context
    })

def get_uploader_role(user):
    """Get user role as string"""
    if not user:
        return 'Guest'
    if user.is_superuser:
        return 'Superuser'
    if user.is_staff:
        return 'Staff'
    return 'User'

@csrf_exempt
def set_scroll_mode(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        request.session['use_infinite_scroll'] = data.get('use_infinite_scroll', False)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


# views.py
@login_required(login_url='login')
@require_http_methods(["POST"])
def ask_question_api(request, hash_id):
    real_id = decode_id(hash_id)
    if real_id is None:
        return JsonResponse({"error": "Invalid image ID"}, status=404)
    
    question = request.POST.get('question', '').strip()
    if not question:
        return JsonResponse({"error": "Question is required"}, status=400)
    
    try:
        from .llm_qa import LLMQA
        llm = LLMQA(provider="gemini")
        answer, from_cache = llm.get_answer(real_id, question)
    except Exception as e:
        # Log the error
        import logging
        logging.error(f"ask_question_api error: {e}")
        # Fallback to rule-based
        from .qa_system import qa_system
        answer, from_cache = qa_system.get_answer(real_id, question)
    
    return JsonResponse({
        "question": question,
        "answer": answer,
        "from_cache": from_cache,
        "image_id": hash_id
    })

# views.py
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import QnAHistory

@login_required
def get_chat_history(request, hash_id):
    """Fetch all Q&A history for an image"""
    real_id = decode_id(hash_id)
    if real_id is None:
        return JsonResponse({"error": "Invalid image ID"}, status=404)
    
    # Fetch all history entries for this image, ordered oldest first
    history = QnAHistory.objects.filter(image_id=real_id).order_by('asked_at')
    
    messages = []
    for entry in history:
        messages.append({"text": entry.question, "isUser": True})
        messages.append({"text": entry.answer, "isUser": False})
    
    return JsonResponse({"messages": messages, "count": len(messages)})


def get_analysis_status(request, hash_id):
    """Check if analysis is complete for an image"""
    real_id = decode_id(hash_id)
    if real_id is None:
        return JsonResponse({"error": "Invalid image ID"}, status=404)
    
    try:
        image = UploadedImage.objects.get(id=real_id)
        has_ai = AIDetection.objects.filter(image=image).exists()
        has_human = HumanEditDetection.objects.filter(image=image).exists()
        has_heatmap_analysis = HeatmapAnalysis.objects.filter(image=image).exists()
        has_verdict = FinalVerdict.objects.filter(image=image).exists()
        
        is_analysed = all([has_ai, has_human, has_heatmap_analysis, has_verdict])
        
        return JsonResponse({
            "is_analysed": is_analysed,
            "has_ai": has_ai,
            "has_human": has_human,
            "has_heatmap_analysis": has_heatmap_analysis,
            "has_verdict": has_verdict
        })
        
    except UploadedImage.DoesNotExist:
        return JsonResponse({"error": "Image not found"}, status=404)
    
@login_required(login_url='login')
def image_report_pdf(request, hash_id):
    real_id = decode_id(hash_id)
    if real_id is None:
        raise Http404("Image not found")

    image = get_object_or_404(UploadedImage, id=real_id)
    metadata = MetadataAnalysis.objects.filter(image=image).first()
    ai = AIDetection.objects.filter(image=image).first()
    human = HumanEditDetection.objects.filter(image=image).first()
    verdict = FinalVerdict.objects.filter(image=image).first()
    heatmap_analysis = HeatmapAnalysis.objects.filter(image=image).first()

    # Read user preferences
    include_heatmap = request.GET.get('include_heatmap', '0') == '1'

    # Compute percentages and offsets
    ai_percentage = 0
    human_percentage = 0
    if ai:
        ai_percentage = ai.confidence if ai.label == "AI-generated" else (100 - ai.confidence)
    if human:
        human_percentage = round(human.detection_score * 100, 2)
    
    # Get C2PA status
    c2pa_present = metadata.c2pa_present if metadata else False

    # Generate characteristics
    characteristics = get_image_characteristics(
        image, ai, human, metadata, heatmap_analysis, verdict
    )

    base_url = request.build_absolute_uri('/')
    context = {
        'image': image,
        'metadata': metadata,
        'ai': ai,
        'human': human,
        'verdict': verdict,
        'heatmap_analysis': heatmap_analysis,
        'ai_percentage': round(ai_percentage, 1),
        'human_percentage': human_percentage,
        'c2pa_present': c2pa_present,
        'base_url': base_url,
        'include_heatmap': include_heatmap,
        'characteristics': characteristics,
        'ai_detection': ai,
        'human_detection': human,
        'heatmap': heatmap_analysis,
    }

    html_string = render_to_string('image_report_pdf.html', context)
    html = HTML(string=html_string, base_url=base_url)
    pdf_file = html.write_pdf()

    response = HttpResponse(pdf_file, content_type='application/pdf')
    
    if request.GET.get('download') == '1':
        response['Content-Disposition'] = f'attachment; filename="{image.filename}_forensix_report.pdf"'
    else:
        response['Content-Disposition'] = f'inline; filename="{image.filename}_forensix_report.pdf"'
    
    return response


@login_required(login_url='login')
def generate_report(request, hash_id):
    real_id = decode_id(hash_id)
    if real_id is None:
        raise Http404("Image not found")

    image = get_object_or_404(UploadedImage, id=real_id)
    metadata = MetadataAnalysis.objects.filter(image=image).first()
    ai = AIDetection.objects.filter(image=image).first()
    human = HumanEditDetection.objects.filter(image=image).first()
    verdict = FinalVerdict.objects.filter(image=image).first()
    heatmap_analysis = HeatmapAnalysis.objects.filter(image=image).first()

    include_heatmap = request.GET.get('include_heatmap', '0') == '1'

    ai_percentage = 0
    human_percentage = 0
    if ai:
        ai_percentage = ai.confidence if ai.label == "AI-generated" else (100 - ai.confidence)
    if human:
        human_percentage = round(human.detection_score * 100, 2)
    
    # Get C2PA status
    c2pa_present = metadata.c2pa_present if metadata else False

    # Check if report with same settings already exists
    existing_report = GeneratedReport.objects.filter(
        image=image,
        include_heatmap=include_heatmap,
        verdict_snapshot=verdict.verdict if verdict else "",
        ai_score_snapshot=round(ai_percentage, 1),
        human_score_snapshot=human_percentage,
    ).first()
    
    if existing_report:
        # Report with same settings exists
        if request.GET.get('download') == '1':
            response = HttpResponse(existing_report.report_file, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{image.filename}_forensix_report.pdf"'
            return response
        else:
            # Redirect to reports page
            return redirect('image_reports', hash_id=encode_id(image.id))

    base_url = request.build_absolute_uri('/')

    # Generate characteristics
    characteristics = get_image_characteristics(
        image, ai, human, metadata, heatmap_analysis, verdict
    )
    
    context = {
        'image': image,
        'metadata': metadata,
        'ai': ai,
        'human': human,
        'verdict': verdict,
        'heatmap_analysis': heatmap_analysis,
        'ai_percentage': round(ai_percentage, 1),
        'human_percentage': human_percentage,
        'c2pa_present': c2pa_present,
        'base_url': base_url,
        'include_heatmap': include_heatmap,
        'characteristics': characteristics,
        'ai_detection': ai,
        'human_detection': human,
        'heatmap': heatmap_analysis,
    }

    html_string = render_to_string('image_report_pdf.html', context)
    html = HTML(string=html_string, base_url=base_url)
    pdf_file = html.write_pdf()

    report = GeneratedReport(
        image=image,
        generated_by=request.user if request.user.is_authenticated else None,
        include_heatmap=include_heatmap,
        include_images=True,  # Always include images in generated reports
        verdict_snapshot=verdict.verdict if verdict else "",
        ai_score_snapshot=round(ai_percentage, 1),
        human_score_snapshot=human_percentage,
        file_size=len(pdf_file)
    )
    report.save()
    report.report_file.save(
        f"{image.filename}_report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.pdf",
        ContentFile(pdf_file)
    )
    report.save()

    if request.GET.get('download') == '1':
        response = HttpResponse(pdf_file, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{image.filename}_forensix_report.pdf"'
        return response
    else:
        # Redirect to reports page
        return redirect('image_reports', hash_id=encode_id(image.id))


@login_required(login_url='login')
def view_saved_report_ajax(request, report_hash_id):
    """View a previously saved report (AJAX version for same-page viewing)"""
    real_id = decode_id(report_hash_id)
    if real_id is None:
        raise Http404("Report not found")
    
    report = get_object_or_404(GeneratedReport, id=real_id)
    
    response = HttpResponse(report.report_file, content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="{report.image.filename}_report.pdf"'
    return response


@login_required(login_url='login')
def download_saved_report(request, report_hash_id):
    """Download a previously saved report"""
    real_id = decode_id(report_hash_id)
    if real_id is None:
        raise Http404("Report not found")
    
    report = get_object_or_404(GeneratedReport, id=real_id)
    
    response = HttpResponse(report.report_file, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{report.image.filename}_report.pdf"'
    return response

@login_required(login_url='login')
def image_reports(request, hash_id):
    """Show all reports for an image"""
    real_id = decode_id(hash_id)
    if real_id is None:
        raise Http404("Image not found")
    
    image = get_object_or_404(UploadedImage, id=real_id)
    reports = GeneratedReport.objects.filter(image=image).order_by('-generated_at')
    
    return render(request, 'image_reports.html', {
        'image': image,
        'reports': reports,
    })


@login_required(login_url='login')
def delete_specific_report(request, report_hash_id):
    """Delete a specific report and return to the same page"""
    real_id = decode_id(report_hash_id)
    if real_id is None:
        raise Http404("Report not found")
    
    report = get_object_or_404(GeneratedReport, id=real_id)
    image_hash = encode_id(report.image.id)
    
    if request.method == 'POST':
        # Delete the file from storage
        if report.report_file:
            report.report_file.delete(save=False)
        report.delete()
        
        # Check if AJAX request (from modal delete)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'message': 'Report deleted successfully'})
        
        # Regular form POST - redirect back to the image reports page
        return redirect('image_reports', hash_id=image_hash)
    
    # GET request - show confirmation page (optional)
    return render(request, 'confirm_delete_report.html', {
        'report': report,
        'image_hash': image_hash,
    })

 
@login_required
def my_all_reports(request):
    """View all reports generated by me across all my images"""
    reports = GeneratedReport.objects.filter(
        generated_by=request.user
    ).select_related('image').order_by('-generated_at')
    
    # Add filter by image
    image_id = request.GET.get('image')
    if image_id:
        decoded_id = decode_id(image_id)
        if decoded_id:
            reports = reports.filter(image_id=decoded_id)
    
    # Add filter by heatmap inclusion
    include_heatmap = request.GET.get('include_heatmap')
    if include_heatmap is not None:
        reports = reports.filter(include_heatmap=include_heatmap == 'true')
    
    return render(request, 'my_all_reports.html', {
        'reports': reports,
        'total_count': reports.count(),
    })


@login_required
def shared_reports(request):
    """View reports shared by others"""
    reports = GeneratedReport.objects.filter(
        is_public=True,
        share_token__isnull=False
    ).exclude(
        generated_by=request.user  # Don't show my own shared reports here
    ).select_related('image', 'generated_by').order_by('-shared_at')
    
    # Add filter by report owner
    owner_id = request.GET.get('owner')
    if owner_id:
        decoded_id = decode_id(owner_id)
        if decoded_id:
            reports = reports.filter(generated_by_id=decoded_id)
    
    return render(request, 'shared_reports.html', {
        'reports': reports,
        'total_count': reports.count(),
    })


@login_required
def share_report(request, report_hash_id):
    """Make a report public and generate shareable link"""
    real_id = decode_id(report_hash_id)
    if real_id is None:
        return JsonResponse({'error': 'Invalid report'}, status=400)
    
    report = get_object_or_404(GeneratedReport, id=real_id, generated_by=request.user)
    
    if not report.share_token:
        report.generate_share_token()
    else:
        # Toggle sharing off if already on
        report.is_public = not report.is_public
        if not report.is_public:
            report.share_token = None
        report.save()
    
    return JsonResponse({
        'success': True,
        'is_public': report.is_public,
        'share_url': report.get_share_url(request) if report.is_public else None,
    })


def view_shared_report(request, share_token):
    """Public view for shared reports (no login required)"""
    report = get_object_or_404(GeneratedReport, share_token=share_token, is_public=True)
    
    # Increment view count
    report.view_count += 1
    report.save()
    
    # Return PDF directly
    response = HttpResponse(report.report_file, content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="shared_{report.image.filename}_report.pdf"'
    return response


@login_required
def toggle_report_sharing(request, report_hash_id):
    """AJAX endpoint to toggle sharing status"""
    from django.http import JsonResponse
    import json
    
    real_id = decode_id(report_hash_id)
    if real_id is None:
        return JsonResponse({'error': 'Invalid report'}, status=400)
    
    report = get_object_or_404(GeneratedReport, id=real_id, generated_by=request.user)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            share = data.get('share', False)
            
            if share:
                if not report.share_token:
                    import uuid
                    report.share_token = str(uuid.uuid4()).replace('-', '')[:20]
                report.is_public = True
                report.shared_at = timezone.now()
                report.save()
            else:
                report.is_public = False
                report.share_token = None
                report.save()
            
            # Build share URL
            share_url = None
            if report.is_public and report.share_token:
                share_url = request.build_absolute_uri(f'/report/shared/{report.share_token}/')
            
            return JsonResponse({
                'success': True,
                'is_public': report.is_public,
                'share_url': share_url,
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid method'}, status=405)

@csrf_exempt
def shared_report_view_count(request, share_token):
    """Track view count for shared reports (AJAX)"""
    report = get_object_or_404(GeneratedReport, share_token=share_token, is_public=True)
    report.view_count += 1
    report.save()
    return JsonResponse({'success': True})


@login_required
def delete_report(request, report_hash_id):
    """Delete a specific report"""
    real_id = decode_id(report_hash_id)
    if real_id is None:
        raise Http404("Report not found")
    
    report = get_object_or_404(GeneratedReport, id=real_id, generated_by=request.user)
    
    if request.method == 'POST':
        # Delete the file from storage
        if report.report_file:
            report.report_file.delete(save=False)
        report.delete()
        
        # Check if AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'message': 'Report deleted successfully'})
        
        return redirect('my_all_reports')
    
    return JsonResponse({'error': 'Invalid method'}, status=405)


@login_required
def delete_image(request, image_id):
    """Delete an uploaded image and all its associated analyses"""
    # Decode the hashed ID
    decoded_id = decode_id(image_id)
    if not decoded_id:
        messages.error(request, "Invalid image ID")
        return redirect('all_images')
    
    # Get the image and check ownership
    image = get_object_or_404(UploadedImage, id=decoded_id, uploaded_by=request.user)
    
    # Store filename for message
    filename = image.filename
    
    # Delete the image file from storage
    if image.image:
        image.image.delete(save=False)
    
    # Delete the database record (cascades to related analyses)
    image.delete()
    
    messages.success(request, f'Image "{filename}" deleted successfully')
    return redirect('all_images')

@login_required
@require_http_methods(["POST"])
def bulk_delete_images(request):
    """Delete multiple images at once"""
    try:
        import json
        data = json.loads(request.body)
        image_ids = data.get('image_ids', [])
        
        deleted_count = 0
        failed_ids = []
        
        for hashed_id in image_ids:
            try:
                decoded_id = decode_id(hashed_id)
                if decoded_id:
                    image = UploadedImage.objects.get(id=decoded_id, uploaded_by=request.user)
                    if image.image:
                        image.image.delete(save=False)
                    image.delete()
                    deleted_count += 1
                else:
                    failed_ids.append(hashed_id)
            except Exception:
                failed_ids.append(hashed_id)
        
        return JsonResponse({
            'success': True,
            'deleted_count': deleted_count,
            'failed_ids': failed_ids,
            'message': f'Successfully deleted {deleted_count} image(s)'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
