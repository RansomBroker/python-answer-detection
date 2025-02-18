import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import time
import json

def detect_and_ocr(image_path):
    try:
        # Load YOLO
        model = YOLO("model/best.pt")

        # Load TrOCR
        processor_path = "microsoft/trocr-small-handwritten"
        model_trocr_path = "microsoft/trocr-small-handwritten"

        processor = TrOCRProcessor.from_pretrained(processor_path)
        model_trocr = VisionEncoderDecoderModel.from_pretrained(model_trocr_path, local_files_only=False)

        # Load image 
        image = cv2.imread(image_path)

        # Detect objects
        results = model([image])

        # Cropped image
        cropped_images = []

        for result in results:
            for box in result.boxes:
                # Crop image
                cropped_images.append(image[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])])

        # Do OCR
        batch_size = 8  # Adjust the batch size according to your GPU memory
        all_texts = []
        inference_times = []

        for i in range(0, len(cropped_images), batch_size):
            batch_images = cropped_images[i:i + batch_size]
            pixel_values = processor(batch_images, return_tensors="pt").pixel_values

            start_time = time.time()
            generated_ids = model_trocr.generate(pixel_values)
            end_time = time.time()

            inference_time = end_time - start_time
            inference_times.append(inference_time)

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(generated_texts)

        # Sort the texts from small to large
        all_texts.sort(key=len)

        return json.dumps({"answers": all_texts, "error": None})

    except Exception as e:
        return json.dumps({"answers": [], "error": str(e)})

# Example usage
result = detect_and_ocr('images/lembar jawaban manual.jpg')
print(result)
