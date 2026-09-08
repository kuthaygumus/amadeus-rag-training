# RAG Eğitim Günü — Cep Kılavuzu

**7 Ekim 2026 · Helios Air / IRIS senaryosu**
Her şey lokal koşar: `qwen2.5:3b` (üretim ve rerank) + `bge-m3` (embedding), Ollama üzerinden.
`nomic-embed-text` de kurulu, çünkü modül 6 ve 9 `bge-m3`'ü onunla karşılaştırıyor.
API key yok, cloud hesabı yok, giriş yok.

> Helios Air kurgusaldır. Bu eğitimdeki hiçbir veri gerçek bir havayoluna, Amadeus sistemine
> veya müşteriye ait değildir; korpus eğitim için yazılmış sentetik materyaldir.

Repo: `github.com/kuthaygumus/amadeus-rag-training`
Site: `amadeus-rag-training.vercel.app/tr/`

---

## 1. Günün omurgası — her adım bir öncekinin yetmediği yerde doğdu

Numaralar sitedeki modül numaralarıyla aynı.

| # | Elimdeki | Sahnede kırılan | Doğan ihtiyaç |
|---|---|---|---|
| 1 | Çıplak LLM | Helios hakkında kendinden emin uydurma, kaynak yok | Kendi verim lazım |
| 2 | Eğitilmiş ağ | Ağırlık = **donmuş fotoğraf** | Veriyi ağırlığa gömeyim |
| 3 | Fine-tune | Q2'yi bilir, Q3'te **eski** cevap, kaynak yok | Taze bilgi, retrain'siz |
| 4 | Prompt'a doldur | **Sığıyor ve doğru cevaplıyor** — kırılan maliyet, gecikme ve ölçek | Sadece doğru parça |
| 5 | BM25 keyword → naive RAG | Parafrazı kaçırıyor, TR→EN **0.000**; sonra retrieval çöp getiriyor | Retrieval neden başarısız oldu? |
| 6 | Varsayılan embedder | İngilizce-only ve **hiçbir uyarı vermiyor**: TR→EN 0.000 | Embedder'ı ölç |
| 7 | Naive chunking | Üç naive strateji de 0.700'de duruyor; ortalama yükselirken `exact_token` 1.000 → 0.750 | Ortalama neyi saklıyor? |
| 8 | ChromaDB | Process ölünce index gidiyor; varsayılan embedder sessizce İngilizce-only | Bu production'da nasıl koşar? |
| 9 | Hybrid + rerank | **Hiçbiri kazandırmıyor** (RRF 0.450, rerank 0.800 → 0.600) | Eklemeden önce ölç |
| 10 | Agentic RAG | Multi-hop q19: 3 dokümanın **0'ı → 3'ü** | Agent gününe köprü |

---

## 2. Ölçülen sayılar — tek referans tablosu

Hepsi aynı 20 altın soru, aynı korpus (`corpus/2026-Q3`, 28 doküman), aynı üç metrik.
Retrieval doküman seviyesinde puanlanıyor. **20 soru benchmark değil:** iki tasarım arasında
karar verdirir, yayımlanacak bir iddia taşımaz — 0.05'in altındaki farklar bu örneklemde gürültü.

**Chunking merdiveni (bge-m3)**

| strateji | hit@1 | recall@5 | MRR | chunk |
|---|---|---|---|---|
| chunk yok (tüm doküman) | 0.550 | 0.717 | 0.654 | 28 |
| fixed-280 | 0.700 | **0.950** | 0.814 | 294 |
| fixed-280 + overlap 60 | 0.700 | 0.883 | 0.799 | 368 |
| recursive-600 | 0.700 | 0.900 | 0.816 | 197 |
| **structure-aware** | **0.800** | 0.833 | **0.844** | 154 |
| structure-aware + boilerplate temizliği | **0.850** | 0.833 | **0.869** | 153 |

> ⚠️ **Üç naive strateji de tam olarak 0.700'de duruyor.** Sabit boyut, üst üste binen sabit
> boyut, recursive splitter — daha akıllı bir naive splitter arayışı hiçbir şey kazandırmıyor.
> 0.800'e ancak dokümanın kendi yapısına göre bölünce çıkılıyor.
>
> ⚠️ **recall@5 en kötü stratejide en yüksek** (fixed-280 0.950, structure-aware 0.833).
> İki metrik bilerek çelişiyor; tek sayı asla yeterli değil.
>
> ⚠️ **Ortalama yükselirken bir kategori düşüyor:** `exact_token` soruları tüm dokümanda
> 1.000 iken fixed-280 altında **0.750**. Kategori kırılımına bakmadan ortalamaya güvenme.

**Embedder (structure-aware chunk'lar üzerinde, aynı 20 soru)**

`tr_en` = Türkçe soru, cevap İngilizce dokümanda; altı soru.

| model | genel hit@1 | `tr_en` hit@1 |
|---|---|---|
| `nomic-embed-text` | 0.350 | 0.000 |
| `all-MiniLM-L6-v2` (ChromaDB varsayılanı) | 0.350 | 0.000 |
| **`bge-m3`** | **0.800** | **0.667** |

> ChromaDB'nin varsayılanı kazara elinize geçen modeldir. Türkçe soruda hata vermiyor,
> uyarmıyor — sadece yanlış dokümanı getiriyor.

**Fusion (structure-aware chunk'lar, bge-m3) — RRF her zaman kazanmıyor**

| | hit@1 | recall@5 | MRR |
|---|---|---|---|
| yalnız dense | **0.800** | 0.833 | **0.844** |
| yalnız BM25 | 0.300 | 0.633 | 0.467 |
| RRF (ikisinin füzyonu) | 0.450 | 0.700 | 0.586 |

**Rerank — yükseltme değil takas** (`qwen2.5:3b`, pointwise, aday başına bir çağrı)

| retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra | |
|---|---|---|---|---|---|
| zayıf (`nomic` + fixed-280) | 0.350 | **0.450** | 0.503 | **0.543** | kazandırıyor |
| zayıf (`nomic` + structure-aware) | 0.350 | **0.400** | 0.492 | **0.537** | kazandırıyor |
| güçlü (`bge-m3` + fixed-280) | 0.700 | 0.550 | 0.814 | 0.712 | kaybettiriyor |
| güçlü (`bge-m3` + structure-aware) | **0.800** | 0.600 | **0.844** | 0.717 | kaybettiriyor |

> Rerank **kendi tavanına düzlüyor.** Ayrımı yapan chunking değil, embedder.
> En güçlü kurulumda soru soru bakınca: doğru doküman zaten 1. sıradaydı **16 soruda**, rerank
> bunların **5'ini** aşağı itti; 1. sırada değildi **4 soruda**, rerank bunların **2'sini** yukarı
> çekti. Her geçiş soru başına 8 model çağrısı.

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
Ve eklerken **pointwise** sor (her adayı ayrı puanla), listwise değil — modelden altı pasajı
aklında tutup sıralamasını istemek, kötü olduğu bir işi istemektir.

> Bu pointwise/listwise karşılaştırması **20 soruluk altın sette değil**, daha erken kullandığımız
> 10 dokümanlık, **5 soruluk probe korpusunda** ölçüldü: listwise 2/5, pointwise 5/5.
> **n=5** — yönü gerçek kabul edin, büyüklüğü kanıtlanmış değil. Yukarıdaki tabloların hiçbir
> hücresiyle karşılaştırılamaz, çünkü ne korpus ne soru seti aynı.

---

## 4. Komutlar

```bash
# 1) Repo ve bağımlılıklar (evde, ofis ağında değil)
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
python -m pip install -r requirements.txt   # sadece numpy + chromadb
# git yoksa: .../archive/refs/heads/main.zip indirip açın

# 2) Modeller — üç zorunlu model, toplam 3.4 GB
ollama pull qwen2.5:3b        # 1.9 GB · üretim ve rerank
ollama pull bge-m3            # 1.2 GB · embedding, Türkçeyi kaldırıyor
ollama pull nomic-embed-text  # 274 MB · modül 6 ve 9 bunu bge-m3 ile karşılaştırıyor
ollama pull qwen2.5:1.5b      # 986 MB · yavaş makine için yedek, opsiyonel
# helios-q2 hiçbir registry'de yok: eğitmen USB ile dağıtıyor. Yoksa modül 3'ün iki probe
# hücresi "(skipped — helios-q2 not installed)" yazıp geçiyor — kayıtlı çıktı devreye girmiyor,
# çünkü öyle bir kayıt yok — ve modülün asıl konusu olan Q2/Q3 fare sheet diff'i yine koşuyor.

# 3) Model olmayan iki dosya (MNIST 11.6 MB + ChromaDB'nin ONNX embedder'ı 83 MB)
python scripts/seed_offline_assets.py

# 4) Yeşil ışık — tek satır READY / NOT READY
python scripts/verify_setup.py
# Windows'ta Python hiç kurulamadıysa:
#   powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
# "running scripts is disabled" diyorsa (Group Policy):
#   Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -

ollama serve   # ikinci bir terminalde, sunucu çalışmıyorsa
```

Hazırlığın tamamı diskte **yaklaşık 3.9 GB**: 1.9 + 1.2 + 0.274 GB model, kurulmuş paketler için
~400 MB, ONNX embedder 83 MB, MNIST 11.6 MB. `verify_setup.py` makinenizi yavaş bulur da
`qwen2.5:1.5b`'yi de çekerseniz üstüne 986 MB biniyor.

```bash
# ChromaDB server (modül 8) — compose dosyası repo kökünde
podman compose up -d                          # ya da: docker compose up -d
curl -s http://localhost:8000/api/v2/heartbeat   # v1 DEĞİL, v1 410 Gone dönüyor
podman compose down

# Ölçümü kendin koştur
python eval/run_benchmark.py --chunking       # chunking merdiveni (bölüm 2'nin ilk tablosu)
python eval/run_benchmark.py --fusion         # dense / BM25 / RRF
python eval/run_benchmark.py --rerank-sweep   # dört kurulumda rerank, 640 çağrı, dakikalar
python exercises/m6_embedding_bakeoff.py      # embedder bake-off, ~20 sn
python exercises/m7_chunking_ladder.py        # chunking merdiveni, 2-3 dk
python exercises/m9_rerank_trade.py --quick   # yavaş makinede kısaltılmış rerank koşusu
python scripts/build_notebooks.py             # .py -> .ipynb
```

**Notebook'lar `.py` dosyası.** VS Code + Microsoft Python eklentisiyle açıp blok blok
`Shift+Enter` ile koşuyorsunuz. Notebook sunucusu kurulmuyor; bağımlılık listesi iki paket.

**Modelleri USB'den kopyalayacaksanız iki uyarı:**
Model dosyaları macOS/Linux'ta `~/.ollama/models`, Windows'ta `%USERPROFILE%\.ollama\models`.
Kopyalamadan **önce Ollama'yı tamamen kapatın** (menü çubuğu / sistem tepsisi → Quit), ve
`blobs/` ile `manifests/` içeriğini **birleştirin, klasörü komple değiştirmeyin** — üstüne
kopyalamak makinede zaten olan modellerin kaydını siliyor.

---

## 5. Odadaki zor sorulara cevaplar

**"20 soru istatistik değil."** Doğru. 20 soru iki tasarım arasında karar verdirir, genel iddia
taşımaz. 0.05'in altındaki farklar bu örneklemde gürültü — sunumda öyle işaretlendi.

**"Neden ragas / LLM-as-judge yok?"** Kota yiyor, yavaş ve skor koşular arasında oynuyor.
Buradaki hiçbir skor modele not verdirmiyor: metrik yalnızca doğru dokümanın sıralamada
kaçıncı olduğuna bakıyor. Aynı korpus + aynı sorular + aynı retriever = aynı sayı.

**"Daha büyük context window bunu çözmez mi?"** Bizim korpusumuzda zaten sığıyor ve doğru
cevaplıyor — ölçüldü, bağlam arttıkça doğruluk **arttı**: sekiz tek-değerli soruda top-1 1/8,
top-3 3/8, top-5 chunk 4/8, korpusun tamamı **7/8**.
Argüman doğruluk değil, maliyet ve ölçek: korpusun tamamı **78,310 karakter** (notebook'un
kurduğu prompt, `[SOURCE: ...]` başlıklarıyla birlikte 79,309) ve modelin belleği olmadığı için
bunun parasını **her soruda yeniden** ödüyorsunuz; aynı 20 soruyu structure-aware top-5 chunk
ortalama **3,427 karakterle** cevaplıyor (min 1,965 · medyan 3,439 · maks 4,430) — soru başına
yaklaşık **23× daha az bağlam**. On kat büyük bir korpus 32,768 token'lık pencereye zaten
girmiyor.

**"Hybrid search her zaman kazanır diye okudum."** Burada kaybetti: hit@1 0.800 → 0.450,
MRR 0.844 → 0.586. RRF her iki retriever da kendi başına sağlamsa işe yarar; chunk seviyesinde
BM25 tek başına 0.300 ve Türkçe sorularda 0.000, füzyon dense sıralamayı da aşağı çekiyor.

**"Bunlar sizin sentetik korpusunuza özel değil mi?"** Evet, ve bu asıl mesele. Genellenen şey
sayılar değil, **yöntem**: sabit soru seti, üç metrik, tekniği benimsemeden önce koştur.

---

## 6. Bugün ölçmediğimiz şey

Buradaki her metrik **retrieval'ı** puanlıyor — doğru doküman geldi mi. Hiçbiri **cevabın**
doğru olup olmadığını puanlamıyor. İkisi farklı sistemler ve farklı şekillerde bozuluyorlar:
finalde retrieval q19'da 3 dokümanın 3'ünü de getirdi, model yine de altı saatlik otel eşiğini
on beş euroluk yemek tutarıyla birleştirip otel bedeli gibi sundu ve son iki cümlesinde değişim
ücreti konusunda kendisiyle çelişti. Aynı K satırının bir kolon yanındaki iki rakam arasında
kaydı: **EUR 70** değişim, **EUR 90** iptal cezası. Retrieval düzeldi, okuma düzelmedi.
(Decompose adımı üretilen bir metin olduğu için bu tek yer deterministik değil; kelimeler
değişir, şekil tekrarlar.)

**Kendi projende önce retrieval eval'ini kur** (ucuz ve deterministik), **sonra cevaplar için
ikincisini** — çünkü birincisi ikincisinin bozuk olduğunu sana söylemez.

---

`github.com/kuthaygumus/amadeus-rag-training` · `amadeus-rag-training.vercel.app/tr/`
