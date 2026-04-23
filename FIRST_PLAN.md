Aspect-aware Retrieval + Contrastive Embedding + Multi-task ABSA 
 

Các bước: 

Step 1: Chuẩn bị dữ liệu 

Step 2: Train embedding (contrastive) 

Step 3: Build retrieval index 

Step 4: Train ABSA model (có retrieval) 

Step 5: Fine-tune end-to-end (optional) 

Step 6: Evaluation + ablation 

🔹 STEP 1 — Chuẩn bị dữ liệu  

Dataset: SemEval 15/16  

3 dạng dữ liệu: 

(1) Token-level (cho sequence tagging) 

Sentence: "food is great but service is slow" 

Tags:     O    O  O     O   B-ASP    O  O 

(2) Classification (aspect + sentiment) 

Input: (sentence, aspect) 

Output: polarity (POS / NEG / NEU) 

(3) Contrastive pairs (cho train embedding model) 

Positive: 
"[SENT] It took forever to get our order [SEP] SERVICE" 
Negatives: 
"[SENT] The food is delicious [SEP] FOOD" 
"[SENT] The service is excellent [SEP] SERVICE" 

STEP 2 — Train embedding (contrastive learning), dùng DeBERTa 

Input format: [CLS] sentence [SEP] aspect [SEP] 

Shape 

Contrastive loss: 

L = - log ( exp(sim(q, k+) / τ) / Σ exp(sim(q, k) / τ) ) 

sim = cosine similarity 

τ = temperature 

👉 Mục tiêu: 

câu cùng aspect + sentiment → gần nhau 

khác → xa nhau 

STEP 3. Sau khi train embedding, encode toàn bộ train set theo cả sentence và aspect. Build Retrieval Index sử dụng FAISS để tìm nhanh.  

Retrieval: Với mỗi query, lấy top-k similar examples 

 STEP 4 — Train ABSA model 

Input: Query (sentence + aspect), top-k retrieved sentences 

Cách dùng retrieval  

[CLS] query [SEP] retrieved_1 [SEP] retrieved_2 ... 

Giả dụ query  đi kèm aspect FOOD thì 1 ví dụ retrieved_1 

retrieved_1: 
[SENT] The food is great but the service is slow 
[ASP] FOOD 
[SENTIMENT] POS 
 
Output (multi-task) 

(1) Token tagging: Linear + CRF (optional) 

(2) Sentiment classification: lấy vector [CLS] 

Loss   L_total = L_tagging + λ * L_classification 

 

🔹 STEP 5 — Fine-tune end-to-end (optional nhưng ăn điểm) 

Thay vì embedding model đứng riêng, nên cho phép backprop từ ABSA → embedding 

-> giúp retrieval “học lại” theo task 

 

Những lỗi dễ dính 

1. Đưa vào kết quả retrieval sai 

✔️ Fix: 

chỉ dùng top-1 hoặc top-3 

hoặc threshold similarity 

 2. Contrastive data quá ít → embedding không học được 

✔️ Fix: generate nhiều negative pairs 

 3. Overfit : SemEval khá nhỏ 

✔️ Fix: 

dropout 

early stopping 

 