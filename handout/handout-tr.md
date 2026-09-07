# RAG Eğitim Günü — Cep Kılavuzu

**7 Ekim 2026 · Helios Air / IRIS senaryosu**
Her şey lokal koşar: `qwen2.5:3b` (üretim) + `bge-m3` (embedding), Ollama üzerinden.
API key yok, cloud hesabı yok, giriş yok.

> Helios Air kurgusaldır. Bu eğitimdeki hiçbir veri gerçek bir havayoluna, Amadeus sistemine
> veya müşteriye ait değildir.

---

## 1. Günün omurgası — her adım bir öncekinin yetmediği yerde doğdu

| # | Elimdeki | Sahnede kırılan | Doğan ihtiyaç |
|---|---|---|---|
| 1 | Çıplak LLM | Aynı soruya 4 farklı uydurma rakam | Kendi verim lazım |
| 2 | Eğitilmiş ağ | Ağırlık = **donmuş fotoğraf** | Veriyi ağırlığa gömeyim |
| 3 | Fine-tune | Q2'yi bilir, Q3'te **eski** cevap, kaynak yok | Taze bilgi, retrain'siz |
| 4 | Prompt'a doldur | Sığıyor ama 30× token; 10× korpusta hiç sığmıyor | Sadece doğru parça |
| 5 | BM25 keyword | Parafrazı kaçırıyor, Türkçe→İngilizce **0.000** | Kelime değil anlam |
| 6 | Naive RAG | hit@1 **0.550**; kolon başlığı kopuk, çelişen SOP'lar | Neden çöp geliyor? |
| 7 | Doğru embedder | Varsayılan İngilizce-only; TR→EN 0.000 → **0.667** | Hâlâ 0.550 |
| 8 | Structure-aware chunking | **0.800** → temizlikle **0.850** | Sıralamaya müdahale? |
| 9 | Hybrid + rerank | **Hiçbiri kazandırmıyor** (RRF 0.450, rerank 0.650) | Ön koşul neymiş? |
| 10 | Agentic RAG | Multi-hop: 3 dokümanın **1'i → 3'ü** | Agent gününe köprü |

---

## 2. Ölçülen sayılar — tek referans tablosu

**Chunking (aynı 20 soru, aynı 3 metrik)**

| strateji | hit@1 | recall@5 | MRR | chunk |
|---|---|---|---|---|
| chunk yok (tüm doküman) | 0.550 | 0.717 | 0.655 | 28 |
| fixed-280 | 0.650 | **0.950** | 0.789 | 288 |
| fixed-280 + overlap 60 | 0.700 | 0.867 | 0.795 | 363 |
| recursive-600 | 0.700 | 0.900 | 0.816 | 195 |
| **structure-aware** | **0.800** | 0.850 | **0.846** | 152 |
| structure-aware + temizlik | **0.850** | 0.850 | **0.871** | 151 |

> ⚠️ **recall@5 en kötü stratejide en yüksek.** Tek sayı asla yeterli değil.
> ⚠️ **fixed-280 tam-token retrieval'ı 1.000 → 0.500 düşürüyor** — ortalama yükselirken.

**Embedder (6 Türkçe soru, cevap İngilizce dokümanda)**

| model | tr_en hit@1 |
|---|---|
| `nomic-embed-text` | 0.000 |
| `all-MiniLM-L6-v2` (ChromaDB varsayılanı) | prob korpusunda 2/5 |
| **`bge-m3`** | **0.667** |

**Rerank — yükseltme değil takas**

| retrieval kurulumu | önce | sonra |
|---|---|---|
| zayıf (`nomic` + fixed-280) | 0.350 | **0.450** ✅ |
| zayıf (`nomic` + structure-aware) | 0.350 | **0.450** ✅ |
| güçlü (`bge-m3` + fixed-280) | 0.650 | 0.550 ❌ |
| güçlü (`bge-m3` + structure-aware) | **0.800** | 0.650 ❌ |

> Rerank **kendi tavanına düzlüyor.** Retriever'ın rerank'inden iyiyse kaybettirir.

---

## 3. Karar rubrikleri — pazartesi sabahı

**Bana gerçekten RAG lazım mı?** Yukarıdan aşağı ilk "evet"te dur.

1. Cevap tek bir dokümanda ve doküman sabit mi → **prompt'a koy, bitti**
2. Korpus context window'a sığıyor ve maliyet umurunda değil mi → **prompt'a doldur**
3. Korpus değişiyor ya da sığmıyor mu → **RAG**
4. Modelin *davranışını* mı değiştirmek istiyorsun, bilgisini değil mi → **fine-tune**
5. Cevap tek aramayla ifade edilemiyor mu → **agentic RAG**

**Fine-tune mu RAG mı?** Dik eksenler, ikisi rakip değil.

| | RAG | Fine-tune |
|---|---|---|
| Veri sık değişiyor | ✅ | ❌ retrain |
| Kaynak göstermek şart | ✅ | ❌ yapısal olarak imkânsız |
| Ton / format / davranış | ❌ | ✅ |
| Kurulum maliyeti | düşük | GPU + veri seti |

**Chunking stratejisi nasıl seçilir?**

| Doküman şuna benziyorsa | Kullan |
|---|---|
| Düz prose, başlıksız | recursive |
| Başlıklı / numaralı kural / tablo | **structure-aware** ← ilk tercih |
| Kısa, bağımsız kayıt (FAQ, makro) | doküman = chunk |
| Her yerde tekrarlanan footer var | **önce `strip_boilerplate()`** |

**Rerank ekleyeyim mi?** Önce ölç. Retriever'ın zaten iyiyse **hayır**.
Ve eklerken: **pointwise** (her adayı ayrı puanla), listwise değil.

---

## 4. Komutlar

```bash
# Kurulum (bir kez, evde — ofis ağında 20 kişi aynı anda yapmasın)
ollama pull qwen2.5:3b        # 1.9 GB, üretim
ollama pull bge-m3            # 1.2 GB, embedding
ollama pull qwen2.5:1.5b      # 986 MB, yavaş makine için yedek
python scripts/verify_setup.py         # tek satır READY / NOT READY

# Model dosyalarının yeri (USB'ye kopyalamak için)
# macOS/Linux : ~/.ollama/models
# Windows     : %USERPROFILE%\.ollama\models

# ChromaDB server (modül 8)
podman pull docker.io/chromadb/chroma
podman run -d -p 8000:8000 chromadb/chroma
curl http://localhost:8000/api/v2/heartbeat     # v1 DEĞİL

# Ölçümü kendin koştur
python eval/run_benchmark.py
python scripts/build_notebooks.py               # .py -> .ipynb
```

**Notebook'lar `.py` olarak da var.** VS Code onları hücreli açar; Jupyter kurulu olmasa da
çalışır.

---

## 5. Odadaki zor sorulara cevaplar

**"20 soru istatistik değil."** Doğru. 20 soru iki tasarım arasında karar verdirir, genel iddia
taşımaz. 0.05'in altındaki farklar bu örneklemde gürültü — sunumda öyle işaretlendi.

**"Neden ragas / LLM-as-judge yok?"** Kota yiyor, yavaş ve skor koşular arasında oynuyor.
Buradaki her sayı deterministik: aynı korpus + aynı sorular = aynı sayı, her seferinde.

**"Daha büyük context window bunu çözmez mi?"** Bizim korpusta zaten sığıyor ve doğru
cevaplıyor — ölçüldü, bağlam arttıkça doğruluk **arttı**. Argüman doğruluk değil: 30× token
maliyeti ve 10× korpusta hiç sığmaması.

**"Hybrid search her zaman kazanır diye okudum."** Burada kaybetti: 0.800 → 0.450. RRF her iki
retriever da kendi başına sağlamsa işe yarar; chunk seviyesinde BM25 Türkçe sorularda 0.000.

**"Bunlar sizin sentetik korpusunuza özel değil mi?"** Evet, ve bu asıl mesele. Genellenen şey
sayılar değil, **yöntem**: sabit soru seti, üç metrik, tekniği benimsemeden önce koştur.

---

## 6. Bugün ölçmediğimiz şey

Buradaki her metrik **retrieval'ı** puanlıyor — doğru doküman geldi mi. Hiçbiri **cevabın**
doğru olup olmadığını puanlamıyor. İkisi farklı sistemler ve farklı şekillerde bozuluyorlar:
finalde retrieval 3/3 doğru dokümanı getirdi ve model yine de saat eşiğini euro tutarı sandı.

**Kendi projende önce retrieval eval'ini kur** (ucuz ve deterministik), **sonra cevaplar için
ikincisini** — çünkü birincisi ikincisinin bozuk olduğunu sana söylemez.

---

`github.com/kuthaygumus/amadeus-rag-training` · `amadeus-rag-training.vercel.app`
