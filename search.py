import os
import cv2
from ultralytics import YOLO
from database import DB

model = YOLO('best.pt')
db = DB("lancedb")


def create_database():
    parent_folder = "Bonn_Furniture_Styles_Dataset/houzz"
    for furniture in os.listdir(parent_folder):
        for style in os.listdir(f"{parent_folder}/{furniture}"):
            for img in os.listdir(f"{parent_folder}/{furniture}/{style}"):
                img_url = f"{parent_folder}/{furniture}/{style}/{img}"
                v = model.embed()
                db.create_embeddings_table(furniture,
                                           {"image_uri": img_url,
                                            "vector": v[0].cpu().numpy(),
                                            "style": style})


def get():
    img_origin = "Bonn_Furniture_Styles_Dataset/houzz/sofas/Modern/2047modern-sofas.jpg"
    detections = model.predict(img_origin)
    cords = detections[0].boxes[0].xyxy[0].tolist()
    cords = [round(x) for x in cords]
    v = model.embed(detections[0].orig_img[cords[1]:cords[3], cords[0]:cords[2]])

    # TODO: need replace bed by name
    result = db.search("bed", v[0].cpu().numpy())
    print(result)


if __name__ == '__main__':
    create_database()
    get()
