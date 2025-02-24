from sqlalchemy import Column, Integer, String
from db.database import Base

class ImageMetadata(Base):
    __tablename__ = "image_metadata"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    file_path = Column(String)
    image_metadata = Column(String)
