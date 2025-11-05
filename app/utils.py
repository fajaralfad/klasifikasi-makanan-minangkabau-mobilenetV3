from fastapi import HTTPException
from PIL import Image
import io

def validate_image(file):
    """Validate uploaded image file"""
    try:
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset file pointer
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400, 
                detail="File size too large. Maximum size is 10MB."
            )
        
        # Try to open image
        image = Image.open(io.BytesIO(file.file.read()))
        file.file.seek(0)  # Reset file pointer
        
        # Check if it's a valid image format
        if image.format not in ['JPEG', 'PNG', 'JPG', 'WEBP']:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Supported formats: JPEG, PNG, JPG, WEBP"
            )
            
        return True
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )