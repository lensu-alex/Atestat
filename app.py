import io
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

app = Flask(__name__)

# --- CONFIGURATION ---
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/owlv2-base-patch16-ensemble"

print(f"--- Starting Server on {DEVICE.upper()} ---")

# Load Model and Processor to GPU
processor = Owlv2Processor.from_pretrained(MODEL_ID)
model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# English labels for AI detection
ENGLISH_LABELS = ["scissors", "knife", "gun", "hammer"]

# Romanian labels for UI display
QUERY_LABELS = ["foarfecă", "cuțit", "armă", "ciocan"]

TEXT_QUERIES = [[f"a photo of a {label}" for label in ENGLISH_LABELS]]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # 1. Get image from request
        file = request.files['image'].read()
        image = Image.open(io.BytesIO(file)).convert("RGB")

        # Get threshold from request, fallback to default
        threshold = request.form.get("threshold", DEFAULT_CONFIDENCE_THRESHOLD)

        try:
            threshold = float(threshold)
        except ValueError:
            threshold = DEFAULT_CONFIDENCE_THRESHOLD

        # Keep threshold within valid range
        threshold = max(0.0, min(1.0, threshold))

        # 2. Process image and move tensors to GPU
        inputs = processor(
            text=TEXT_QUERIES,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # 3. Post-process using selected threshold
        target_sizes = torch.Tensor([image.size[::-1]]).to(DEVICE)

        results = processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )

        detections = []

        for result in results:
            for score, label in zip(result["scores"], result["labels"]):
                label_name = QUERY_LABELS[label.item()]
                confidence = round(score.item(), 3)

                detections.append({
                    "label": label_name,
                    "confidence": confidence
                })

        # 4. Handle "Clear" logic
        if not detections:
            return jsonify({
                "detected": ["clear"],
                "detections": [],
                "threshold": threshold
            })

        # Optional: keep only the highest-confidence detection per label
        best_detections = {}

        for detection in detections:
            label = detection["label"]
            confidence = detection["confidence"]

            if label not in best_detections:
                best_detections[label] = detection
            elif confidence > best_detections[label]["confidence"]:
                best_detections[label] = detection

        final_detections = list(best_detections.values())

        return jsonify({
            "detected": [d["label"] for d in final_detections],
            "detections": final_detections,
            "threshold": threshold
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "detected": ["error"],
            "detections": [],
            "msg": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)