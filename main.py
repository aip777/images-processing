from fastapi import FastAPI, UploadFile, File, Query, Depends
import shutil
import os
import uuid
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import ImageMetadata
from iprocessor.processor import ImageProcessor

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.tiff")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    processor = ImageProcessor(file_path)
    metadata = processor.extract_metadata()
    db_metadata = ImageMetadata(
        file_name=file.filename,
        file_path=file_path,
        image_metadata=str(metadata)
    )
    db.add(db_metadata)
    db.commit()
    return {"message": "Image uploaded", "metadata": metadata}

@app.get("/metadata")
async def get_metadata(db: Session = Depends(get_db)):
    data = db.query(ImageMetadata).all()
    return {"metadata": [entry.image_metadata for entry in data]}

@app.get("/slice")
async def get_slice(
    z: int = Query(None), time: int = Query(None), channel: int = Query(None),
    db: Session = Depends(get_db)
):
    latest_image = db.query(ImageMetadata).order_by(ImageMetadata.id.desc()).first()
    if not latest_image:
        return {"error": "No images found"}

    processor = ImageProcessor(latest_image.file_path)
    sliced_image = processor.get_slice(z, time, channel)
    return {"slice": sliced_image.tolist()}

@app.post("/analyze")
async def analyze_image(db: Session = Depends(get_db)):
    latest_image = db.query(ImageMetadata).order_by(ImageMetadata.id.desc()).first()
    if not latest_image:
        return {"error": "No images found"}

    processor = ImageProcessor(latest_image.file_path)
    pca_result = processor.perform_pca()
    return {"pca_result": pca_result.tolist()}

@app.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    latest_image = db.query(ImageMetadata).order_by(ImageMetadata.id.desc()).first()
    if not latest_image:
        return {"error": "No images found"}
    processor = ImageProcessor(latest_image.file_path)
    stats = processor.compute_statistics()
    return {"statistics": stats}
