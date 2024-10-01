from sentence_transformers import SentenceTransformer

# 내모델돌렵보기
model = SentenceTransformer('./hyunji_embbeding', trust_remote_code=True)

embd = model.encode('hello, fine')

print(embd)