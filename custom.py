from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/embedding', methods=['POST'])
def process_embedding():
    data = request.json
    model = SentenceTransformer('./hyunji_embbeding', trust_remote_code=True)
    embd = model.encode(data['data'])
    result = {"embedding": embd.tolist()}  # 임시 결과
    return jsonify(result)

if __name__ == '__main__':
        app.run(port=5000)