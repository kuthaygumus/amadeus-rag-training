<!-- TRAINER-ONLY BLOCK. Delete everything above the first `---` before printing room copies. -->

# ⚠️ EĞİTMEN İÇİN — ODAYA DAĞITILAN KOPYADA BU BLOK YOK

## Presenter mode: salon dolmadan bir kez **`Alt+Shift+P`**

Sahne yönergelerin — ne söyleyeceğin, ne izleyeceğin, demo patlarsa ne yapacağın, slotun kaç
dakika olduğu — her modül sayfasında **duruyor ama varsayılan olarak gizli**. Açmazsan tek birini
bile göremezsin. Sitede bunu ele veren **görünür hiçbir kontrol yok**: katılımcı mekanizmayı
keşfetmesin diye bilerek böyle. Yani sana hatırlatacak tek yer bu blok.

- **Aç:** `Alt+Shift+P` — ya da herhangi bir sayfanın URL'sine `?presenter=1` ekle, örneğin
  `https://amadeus-rag-training.vercel.app/tr/modules/00-setup/?presenter=1`
- **Kapat:** tekrar `Alt+Shift+P` — ya da `?presenter=0`
- **Açık olduğunu nereden bilirsin:** köşede `PRESENTER MODE · Alt+Shift+P` rozeti belirir
- **Hatırlanıyor:** `localStorage` → `rag-training-presenter`, gün boyu, sayfa sayfa değil

**09:00'dan önce yap:** eğitmen laptopunda `Alt+Shift+P`, rozeti gör, sayfayı yenile, notlar
hâlâ yerinde mi doğrula — projektörü bağlamadan önce. Gizli sekmede `localStorage` boş gelir;
orada mekanizmayı yeniden açman gerekir. Notlar arama indeksinden de çıkarıldı, yani salon
aramayla da bulamaz.

**⚠️ Odaya dağıtacağın kopyaları basmadan önce buraya kadarki her şeyi sil.**

---

# RAG Eğitim Günü — Cep Kılavuzu

**7 Ekim 2026 · Kraken Air (XX) korpusu**
Her şey lokal koşar: `qwen2.5:3b` (üretim ve rerank) + `bge-m3` (embedding), Ollama üzerinden.
`nomic-embed-text` de kurulu, çünkü modül 6 ve 9 `bge-m3`'ü onunla karşılaştırıyor.
API key yok, cloud hesabı yok, giriş yok.

Repo: `github.com/kuthaygumus/amadeus-rag-training`
Site: `amadeus-rag-training.vercel.app/tr/`

---

## 1. Günün omurgası — her adım bir öncekinin yetmediği yerde doğdu

Numaralar sitedeki modül numaralarıyla aynı.

| # | Elimdeki | Sahnede kırılan | Doğan ihtiyaç |
|---|---|---|---|
| 1 | Çıplak LLM | Kraken Air hakkında kendinden emin uydurma, kaynak yok | Kendi verim lazım |
| 2 | Eğitilmiş ağ | Ağırlık = **donmuş fotoğraf** | Veriyi ağırlığa gömeyim |
| 3 | Fine-tune | Q2'yi bilir, Q3'te **eski** cevap, kaynak yok | Taze bilgi, retrain'siz |
| 4 | Prompt'a doldur | **Sığıyor ve 8/8 doğru cevaplıyor** — kırılan maliyet, gecikme ve ölçek | Sadece doğru parça |
| 5 | BM25 keyword → naive RAG | Parafrazı kaçırıyor, TR→EN **0.000**; sonra retrieval çöp getiriyor | Retrieval neden başarısız oldu? |
| 6 | Varsayılan embedder | İngilizce-only ve **hiçbir uyarı vermiyor**: TR→EN 0.000 | Embedder'ı ölç |
| 7 | Naive chunking | fixed-280 ortalamayı yükseltirken `exact_token` 1.000 → 0.750; overlap eklemek hit@1'i **0.550'ye düşürüyor**, üstelik recall@5'i de düşürüyor | Ortalama neyi saklıyor? |
| 8 | ChromaDB | Process ölünce index gidiyor; varsayılan embedder sessizce İngilizce-only | Bu production'da nasıl koşar? |
| 9 | Hybrid + rerank | **Hiçbiri kazandırmıyor** (RRF hit@1 0.450, rerank 0.750 → 0.600) | Eklemeden önce ölç |
| 10 | Agentic RAG | Multi-hop q19: 3 dokümanın **0'ı → 2'si** | Agent gününe köprü |

---

## 2. Ölçülen sayılar — tek referans tablosu

Hepsi aynı 20 altın soru, aynı korpus (`corpus/2026-Q3`, 28 doküman), aynı üç metrik.
Retrieval doküman seviyesinde puanlanıyor. **20 soru benchmark değil:** bir soru 0.05 ediyor,
yani hit@1 zaten 0.05'lik adımlarla hareket edebiliyor. Aşağıdaki merdivende komşu iki basamak
arasındaki her fark tam olarak bir soru — overlap satırı hariç, o üç soru. **Yönü alıntılayın,
rakamı değil.** Kaynak: `eval/RESULTS.md`.

**Chunking merdiveni (bge-m3)**

| strateji | hit@1 | recall@5 | MRR | chunk |
|---|---|---|---|---|
| chunk yok (tüm doküman) | 0.600 | 0.717 | 0.677 | 28 |
| fixed-280 | 0.700 | **0.917** | 0.817 | 294 |
| fixed-280 + overlap 60 | **0.550** | 0.883 | 0.720 | 368 |
| recursive-600 | 0.700 | 0.900 | 0.806 | 197 |
| **structure-aware (900)** | **0.750** | 0.833 | **0.817** | 154 |
| **structure-aware + boilerplate temizliği** | **0.800** | 0.833 | **0.841** | 153 |

> ⚠️ **Merdivende geriye giden tek hamle overlap — ve şimdi ikisinde birden.** fixed-280'e 60
> karakter overlap eklemek hit@1'i 0.700'den 0.550'ye (üç soru), MRR'yi 0.817'den 0.720'ye,
> recall@5'i de 0.917'den 0.883'e düşürüyor. Üstelik 0.550, hiç chunk'lamadan (0.600) daha kötü.
> Overlap her yerde güvenli varsayılan diye satılıyor; burada sıralamanın en kötü basamağı, hem
> de en pahalısı (368 chunk, merdivenin en fazlası). İkisini birden puanlamadan bunu göremezsiniz.
>
> ⚠️ **En iyi recall@5 en iyi sıralayıcıya ait değil.** recall@5 kolonunun tepesi fixed-280
> (0.917) ama hit@1'i 0.700 — recursive-600'le aynı, structure-aware'in altında. Üç naive kesim
> de recall@5'te structure-aware'i geçiyor — 0.917 / 0.883 / 0.900'a karşı 0.833. Daha küçük
> parçaya bölen strateji doğru dokümanı ilk beşe daha kolay sokuyor, başa koymakta zorlanıyor.
> İki metrik bilerek çelişiyor; tek sayı asla yeterli değil.
>
> ⚠️ **structure-aware hit@1'de öne geçiyor, MRR'de fixed-280'le eşleşiyor.** hit@1 0.750 ile
> fixed-280 ve recursive-600'ün (ikisi de 0.700) bir soru önünde; MRR'si 0.817 — fixed-280'inkiyle
> birebir aynı, sıralama kalitesi metriği ikisini ayıramıyor — ve recall@5'i 0.833 ile
> chunk'lanmış stratejilerin en düşüğü (kendi boilerplate-temizlenmiş haliyle eşit). Bir soruluk
> hit@1 farkı ve MRR beraberliği kesin bir zafer değil; asıl mesele **hangi metrikte kazandığı**.
>
> ⚠️ **Sayfadaki en ucuz 0.05 boilerplate temizliği.** Tekrarlanan yasal footer'ı atmak korpusun
> **%4.2**'sini götürüyor (78 310 → 75 037 karakter) ve structure-aware'i hit@1 0.750'den 0.800'e,
> MRR'yi 0.817'den 0.841'e taşıyor — recall@5 0.833'te sabit, bir chunk eksiğiyle (153). Kazancın
> tamamı tek yere iniyor: `en_en` 0.500 → 0.750, başka hiçbir kategori kımıldamıyor.
>
> ⚠️ **Ortalama yükselirken bir kategori düşüyor:** dört `exact_token` sorusu tüm dokümanda
> **1.000** iken fixed-280 altında 0.750'ye, overlap altında **0.500**'e iniyor — ve tam da bu iki
> basamak recall@5'te structure-aware'in ulaştığından yüksek rapor ediyor. Kategori kırılımına
> bakmadan ortalamaya güvenme.

**Embedder (154 structure-aware chunk üzerinde, aynı 20 soru)**

`tr_en` = Türkçe soru, cevap İngilizce dokümanda; altı soru.

| model | hit@1 | recall@5 | MRR | `tr_en` hit@1 |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` (ChromaDB varsayılanı) | 0.350 | 0.600 | 0.467 | **0.000** |
| `nomic-embed-text` | 0.350 | 0.633 | 0.490 | **0.000** |
| **`bge-m3`** | **0.750** | **0.833** | **0.817** | **0.667** |

> Sayfadaki tek "bir soru genişliğinde olmayan" fark bu: 0.750'e karşı 0.350 **sekiz soru**,
> `tr_en` kolonu ise dörde sıfır. İki zayıf embedder hit@1'de birbirinden ayrılmıyor bile —
> ikisi de 0.350. ChromaDB'nin varsayılanı kazara elinize geçen modeldir; Türkçe soruda hata
> vermiyor, uyarmıyor — sadece yanlış dokümanı getiriyor.
>
> Karşı ağırlık: `en_en` satırında **üç embedder de 0.500**. İngilizce soru + İngilizce doküman
> testinde ucuz embedder görünür şekilde kötü değil. İngilizce-only bir test seti bu hatayı asla
> yüzeye çıkarmazdı.

**Fusion (154 structure-aware chunk, bge-m3) — RRF her zaman kazanmıyor**

| | hit@1 | recall@5 | MRR |
|---|---|---|---|
| yalnız dense | **0.750** | **0.833** | **0.817** |
| yalnız BM25 | 0.300 | 0.633 | 0.467 |
| RRF (ikisinin füzyonu) | 0.450 | 0.683 | 0.579 |

> Doküman seviyesinde tablo yumuşuyor ama karar değişmiyor: dense 0.600 / 0.717 / 0.677,
> BM25 0.400 / 0.583 / 0.515, RRF 0.500 / 0.733 / 0.611. **BM25 exact_token'da bile dense'i
> geçmiyor** — dense 1.000, BM25 0.750. "Uçuş kodu string'dir, string eşleme kazanır" sezgisini
> bu korpus tek komutta çürütüyor. BM25'in tuttuğu tek yer TR soru + TR doküman; çöktüğü yer
> diller arası, ve orada her iki granülerlikte de 0.000.

**Rerank — yükseltme değil takas** (`qwen2.5:3b`, pointwise, aday başına bir çağrı)

| retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra | |
|---|---|---|---|---|---|
| zayıf (`nomic` + fixed-280) | 0.350 | **0.450** | 0.515 | **0.544** | kazandırıyor |
| zayıf (`nomic` + structure-aware) | 0.350 | **0.400** | 0.490 | **0.540** | kazandırıyor |
| güçlü (`bge-m3` + fixed-280) | 0.700 | 0.500 | 0.817 | 0.680 | kaybettiriyor |
| güçlü (`bge-m3` + structure-aware) | **0.750** | 0.600 | **0.817** | 0.725 | kaybettiriyor |

> Rerank **kendi tavanına düzlüyor.** Önce dört kurulum 0.350–0.750 aralığına yayılıyor, sonra
> 0.400–0.600'e sıkışıyor; MRR iki zayıfta yükseliyor, iki güçlüde düşüyor, istisnasız.
> Ayrımı yapan chunking değil, embedder.
>
> En güçlü kurulumda soru soru bakınca: doğru doküman zaten 1. sıradaydı **15 soruda**, rerank
> bunların **4'ünü** aşağı itti (q06, q10, q16, q18); 1. sırada değildi **5 soruda**, rerank
> bunların **1'ini** yukarı çekti (q14). On beş eksi dört artı bir, tam olarak sonra-sütununun
> 0.600'ü ediyor. Bedeli çağrı cinsinden kesin: soru başına **8 model çağrısı**, 20 soruluk bir
> geçiş 160, yukarıdaki tabloyu doldurmak 640.

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
Ve eklerken **pointwise** sor (her adayı ayrı puanla), listwise değil.

> Bu tercih ölçülmüş bir yerden geliyor. `notebooks/06_hybrid_rerank_contextual.py`'de aynı modele
> **altı** pasaj verilip sıraya dizmesi istendiğinde cevabı `1,4,2,5` oldu — altı pasaj için dört
> indeks; ortada kullanılabilir bir sıralama yok. Aynı adaylar tek tek puanlatıldığında altı
> çağrıda altı tam sayı (4, 2, 2, 5, 5, 3) döndü, 3.3 saniyede, ve o sayılar listeyi gerçekten
> yeniden sıralıyor. `retrieval.py`'de `pointwise_rerank()` bu yüzden var, listwise karşılığı yok.
>
> `UNVERIFIED: eskiden bu kutuda duran listwise 2/5 – pointwise 5/5 tablosu (MRR 0.600 / 1.000) —
> 10 doküman, 5 soruluk emekli probe korpusunda ölçülmüştü ve bu repoda onu üreten bir komut yok.`
> Alıntılamayın; mekanizma yukarıda, canlı ve tekrar üretilebilir hâlde duruyor.

---

## 4. Komutlar

Günün **iki yüzeyi** var ve ikisinin de çapası repo kökü:

1. **Terminal, repo kökünde.** `corpus/`, `notebooks/`, `eval/` ve `exercises/` klasörlerini
   içeren klasöre `cd` yapılmış tek bir terminal. Aşağıdaki her `ollama ...`, `python scripts/...`,
   `python exercises/...` ve `python eval/...` komutu **orada** koşuyor.
2. **VS Code, açık klasör repo kökü.** `File → Open Folder` ile **repo kökünü** açın,
   `notebooks/` klasörünü değil — `notebooks/` açarsanız sonraki her `python scripts/...` ve
   `python eval/...` komutu kırılır. Notebook'lar `notebooks/` altında percent-format `.py`
   dosyaları: birini açın, imleci bir `# %%` bloğunun içine koyun, `Shift+Enter`; çıktı
   Interactive window'da beliriyor.

Jupyter sunucusu yok, tarayıcı notebook'u yok, cloud konsolu yok. Python bağımlılığı iki paket.

**Terminal (repo kökü) — hazırlık:**

```bash
# 1) Repo ve bağımlılıklar (evde, ofis ağında değil)
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training                     # bundan sonraki her komut buradan koşuyor
python -m pip install -r requirements.txt   # sadece numpy + chromadb
# git yoksa: .../archive/refs/heads/main.zip indirip açın, sonra o klasöre cd yapın

# 2) Modeller — üç zorunlu model, toplam 3.4 GB
ollama pull qwen2.5:3b        # 1.9 GB · üretim ve rerank
ollama pull bge-m3            # 1.2 GB · embedding, Türkçeyi kaldırıyor
ollama pull nomic-embed-text  # 274 MB · modül 6 ve 9 bunu bge-m3 ile karşılaştırıyor
ollama pull qwen2.5:1.5b      # 986 MB · yavaş makine için yedek, opsiyonel
# kraken-q2 hiçbir registry'de yok: eğitmen USB ile dağıtıyor. Yoksa modül 3'ün iki probe
# hücresi "(skipped — kraken-q2 not installed)" yazıp geçiyor — kayıtlı çıktı devreye girmiyor,
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

**Terminal (repo kökü) — günün ölçümleri:**

```bash
# ChromaDB server (modül 8) — compose dosyası repo kökünde
podman compose up -d                          # ya da: docker compose up -d
curl -s http://localhost:8000/api/v2/heartbeat   # v1 DEĞİL, v1 410 Gone dönüyor
podman compose down

# Ölçümü kendin koştur
python eval/run_benchmark.py --chunking       # chunking merdiveni (bölüm 2'nin ilk tablosu)
python eval/run_benchmark.py --fusion         # dense / BM25 / RRF, chunk seviyesinde
python eval/run_benchmark.py --skip-rerank    # dense / BM25 / RRF, doküman seviyesinde
python eval/run_benchmark.py --rerank-sweep   # dört kurulumda rerank, 640 çağrı, dakikalar
python exercises/m6_embedding_bakeoff.py      # embedder bake-off, ~20 sn
python exercises/m7_chunking_ladder.py        # chunking merdiveni, 2-3 dk
python exercises/m9_rerank_trade.py --quick   # yavaş makinede kısaltılmış rerank koşusu
python scripts/build_notebooks.py             # .py -> .ipynb
```

**Modelleri USB'den kopyalayacaksanız iki uyarı:**
Model dosyaları macOS/Linux'ta `~/.ollama/models`, Windows'ta `%USERPROFILE%\.ollama\models`.
Kopyalamadan **önce Ollama'yı tamamen kapatın** (menü çubuğu / sistem tepsisi → Quit), ve
`blobs/` ile `manifests/` içeriğini **birleştirin, klasörü komple değiştirmeyin** — üstüne
kopyalamak makinede zaten olan modellerin kaydını siliyor.

---

## 5. Odadaki zor sorulara cevaplar

**"20 soru istatistik değil."** Doğru. 20 soru iki tasarım arasında karar verdirir, genel iddia
taşımaz. Bir soru 0.05 ediyor ve merdivendeki farkların neredeyse hepsi tam olarak bir soru
genişliğinde — sunumda öyle işaretlendi.

**"Neden ragas / LLM-as-judge yok?"** Kota yiyor, yavaş ve skor koşular arasında oynuyor.
Buradaki hiçbir skor modele not verdirmiyor: metrik yalnızca doğru dokümanın sıralamada
kaçıncı olduğuna bakıyor. Aynı korpus + aynı sorular + aynı retriever = aynı sayı.

**"Daha büyük context window bunu çözmez mi?"** Bizim korpusumuzda zaten sığıyor ve doğru
cevaplıyor — ölçüldü, ve bağlam arttıkça doğruluk hiç düşmedi: sekiz tek-değerli soruda
top-1 chunk 1/8, top-3 3/8, top-5 3/8, korpusun tamamı **8/8**. Bu ölçekte ve bu model boyunda
aranacak bir "distractor cezası" yok; top-5'in top-3 üzerine kattığı da hiçbir şey — 888 karakter
fazla bağlam, aynı üç doğru cevap. Argüman doğruluk değil, maliyet ve ölçek: korpusun tamamı
**78 310 karakter** (notebook'un kurduğu prompt, `[SOURCE: ...]` başlıklarıyla birlikte 79 309) ve
modelin belleği olmadığı için bunun parasını **her soruda yeniden** ödüyorsunuz; aynı sekiz soruya
structure-aware top-5 chunk **2 144 karakterle** giriyor — soru başına **37× daha az bağlam** — ve
saat de aynı yöne bakıyor: sekiz soru için 75.4 sn'ye karşı 10.2 sn. On kat büyük bir korpus
(264 360 token) 32 768 token'lık pencereye zaten girmiyor.

**"Hybrid search her zaman kazanır diye okudum."** Burada kaybetti: hit@1 0.750 → 0.450,
MRR 0.817 → 0.579. RRF her iki retriever da kendi başına sağlamsa işe yarar; chunk seviyesinde
BM25 tek başına 0.300 ve Türkçe sorularda 0.000, füzyon dense sıralamayı da aşağı çekiyor.

**"Bunlar sizin sentetik korpusunuza özel değil mi?"** Evet, ve bu asıl mesele. Genellenen şey
sayılar değil, **yöntem**: sabit soru seti, üç metrik, tekniği benimsemeden önce koştur.

---

## 6. Bugün ölçmediğimiz şey

Buradaki her metrik **retrieval'ı** puanlıyor — doğru doküman geldi mi. Hiçbiri **cevabın**
doğru olup olmadığını puanlamıyor. İkisi farklı sistemler ve farklı şekillerde bozuluyorlar.
İki örnek, ikisi de bugünün koşularından: top-5 chunk yürürlükteki otel eşiğini doğru getirip
yemek fişi tutarını yanlış cevapladı; agentic modülünde ise q19'da gerekli 3 dokümanın **2'sini**
buldu (`interline_xx_yy` hâlâ eksik kaldı, arama turları yeni bir şey getirmeyince durdu), buna
rağmen üretilen cevap **EUR 70** değişim cezasını doğru şekilde K sınıfına bağlarken **EUR 90**
iptal cezasını yanlışlıkla M sınıfına yazdı — aynı cevabın iki maddesi arasında sınıf karıştı.
Retrieval kısmen düzeldi (0'dan 2'ye), okuma yine de hata yaptı. (Decompose adımı üretilen bir
metin olduğu için bu tek yer deterministik değil; kelimeler değişir, şekil tekrarlar.)

**Kendi projende önce retrieval eval'ini kur** (ucuz ve deterministik), **sonra cevaplar için
ikincisini** — çünkü birincisi ikincisinin bozuk olduğunu sana söylemez.

---

`github.com/kuthaygumus/amadeus-rag-training` · `amadeus-rag-training.vercel.app/tr/`
