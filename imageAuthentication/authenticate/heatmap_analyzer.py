# authenticate/heatmap_analyzer.py
import numpy as np
from PIL import Image
from django.conf import settings
from scipy import ndimage

# Import models - use relative import
from .models import UploadedImage, HeatmapAnalysis

def analyze_heatmap_from_path(image_id, heatmap_path, detection_score):
    """Analyze saved heatmap and store results in database"""
    try:
        image = UploadedImage.objects.get(id=image_id)
        
        # Load heatmap
        heatmap_img = Image.open(heatmap_path)
        heatmap_array = np.array(heatmap_img)
        
        # Convert to intensity (use red channel for jet colormap)
        if len(heatmap_array.shape) == 3:
            # For RGB heatmap, red channel indicates intensity
            heatmap_intensity = heatmap_array[:, :, 0].astype(float) / 255.0
        else:
            heatmap_intensity = heatmap_array.astype(float) / 255.0
        
        # Split into 3x3 grid
        h, w = heatmap_intensity.shape
        grid_h, grid_w = h // 3, w // 3
        
        grid_scores = {}
        regions = ['top_left', 'top_center', 'top_right',
                   'middle_left', 'middle_center', 'middle_right',
                   'bottom_left', 'bottom_center', 'bottom_right']
        
        for idx, region in enumerate(regions):
            row = idx // 3
            col = idx % 3
            y1, y2 = row * grid_h, (row + 1) * grid_h
            x1, x2 = col * grid_w, (col + 1) * grid_w
            
            # Handle case where grid division might not be perfect
            if y2 > h:
                y2 = h
            if x2 > w:
                x2 = w
                
            region_data = heatmap_intensity[y1:y2, x1:x2]
            if region_data.size > 0:
                grid_scores[region] = float(np.mean(region_data))
            else:
                grid_scores[region] = 0.0
        
        # Calculate statistics
        max_score = max(grid_scores.values())
        avg_score = np.mean(list(grid_scores.values()))
        
        # Percentage of image with editing > 0.3 threshold
        edited_percentage = float((heatmap_intensity > 0.3).sum() / heatmap_intensity.size * 100)
        
        # Find primary and secondary regions
        sorted_regions = sorted(grid_scores.items(), key=lambda x: x[1], reverse=True)
        primary_region = sorted_regions[0][0] if sorted_regions else ""
        secondary_region = sorted_regions[1][0] if len(sorted_regions) > 1 else ""
        
        # Detect distinct edit zones (connected components)
        try:
            labeled, num_features = ndimage.label(heatmap_intensity > 0.5)
            
            zones_details = []
            for i in range(1, num_features + 1):
                zone_mask = labeled == i
                if zone_mask.sum() > 100:  # Minimum zone size
                    y_coords, x_coords = np.where(zone_mask)
                    zones_details.append({
                        'zone_id': i,
                        'center_y': int(np.mean(y_coords)),
                        'center_x': int(np.mean(x_coords)),
                        'size': int(zone_mask.sum()),
                        'avg_score': float(np.mean(heatmap_intensity[zone_mask]))
                    })
        except:
            num_features = 0
            zones_details = []
        
        # Determine severity
        if detection_score > 0.7:
            severity = 'VERY_HIGH'
        elif detection_score > 0.5:
            severity = 'HIGH'
        elif detection_score > 0.3:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        # Save to database
        heatmap_analysis, created = HeatmapAnalysis.objects.update_or_create(
            image=image,
            defaults={
                "grid_scores": grid_scores,
                "max_edit_score": max_score,
                "avg_edit_score": avg_score,
                "edited_percentage": edited_percentage,
                "primary_region": primary_region,
                "secondary_region": secondary_region,
                "edit_zones_count": num_features,
                "zones_details": zones_details,
                "severity_level": severity,
                "region_scores": grid_scores
            }
        )
        
        print(f"✅ Heatmap analysis saved for {image.filename}")
        print(f"   Primary region: {primary_region}, Severity: {severity}")
        return True
        
    except Exception as e:
        print(f"❌ Heatmap analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
