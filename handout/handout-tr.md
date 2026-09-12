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

**7 Ekim 2026 · Kraken Air (XX) korpusu · her şey lokal**

Gün boyu tek bir soru sorduk: *"Kısa mesafe Avrupa bileti, CLASSIC fare, booking class K — yolcu
başına iptal cezası ne kadar?"* Doğru cevap `fare_classic_shorthaul.md`'nin RULE 2A tablosunda,
K satırında: **EUR 90** (Q3 edisyonu; Q2'de EUR 120). Önce çıplak model bir rakam uydurdu, sonra
korpusun tamamını prompt'a doldurduk ve çalıştı ama her soruda 21 000 token ödedik, sonra ingest →
retrieve → query pipeline'ını kurduk, doğru dokümanı bulup yanlış dilimi okuduk, kesimi dokümanın
yapısına göre değiştirdik ve EUR 90 geldi, en sonunda Q2'yi ingest edip 120'yi, Q3'ü ingest edip
90'ı gördük. Hiçbir weight değişmedi. Hiçbir şey cloud'a gitmedi: `gemma3:4b` cevapladı, `bge-m3`
vektörledi, ChromaDB ve lab'ın api'si iki container'da koştu, sen Bruno'da tıkladın.

---

## 1. Zincir — her adım bir öncekinin yetmediği yerde doğdu

Numaralar sitedeki modül numaralarıyla aynı. Modül 0 kurulumdu; 2 bir gate değil, bir kurulum;
9 geri sarma.

| # | Elimdeki | Sahnede kırılan | Doğan ihtiyaç |
|---|---|---|---|
| 1 | Çıplak LLM | Kraken Air'i bilmiyor — ve **bilmediğini bilmiyor**; kendinden emin bir tutar uyduruyor | Bir modelde bilgi *nerede*? |
| 2 | Sinir ağı (MNIST demo) | Weight'ler training verisinin **donmuş fotoğrafı** | Fotoğrafı kendi verimle yeniden çekebilir miyim? |
| 3 | Fine-tune (`kraken-q2`, eğitmen demosu) | Q2'yi ezberden cevaplıyor, EUR 120 — **veri değişene kadar**; kaynak gösteremiyor | Taze bilgi, retrain'siz |
| 4 | Korpusun tamamı prompt'ta | Çalışıyor — kırılan maliyet: ~21 000 token/soru, soğukta 60–120 s | Modele sadece doğru parçayı ver |
| 5 | Basit RAG: ingest → retrieve → query | Cevap kaynaklı ama **doğru doküman, yanlış dilim** | Store'da gerçekten ne duruyor? |
| 6 | Container'da ChromaDB | Chunk'lara bakabiliyoruz: `\| K \|` satırının chunk'ında üç EUR tutarı var, **header yok** | Korpusu farklı kes |
| 7 | Structure-aware chunking + boilerplate temizliği | Doğru chunk, doğru cevap — **EUR 90**, tek bir weight değişmeden | Bunu fine-tuning'in asıl kaybettiği yerde kanıtla |
| 8 | Güncellik: Q2 ingest → Q3 ingest | Q2 **120**, Q3 **90**; yeniden ingest, yeniden training değil | — |
| 9 | Geri sarma | Sekiz gate, hiçbiri tanımla açılmadı | Pazartesi hâlâ çalışan sürüm: re-ingest |

Kapanış cümlesi: *Bugünkü her adım, bir öncekinin yetmediği yerde doğdu. Sonuncusu yeniden training
değil, yeniden ingest'ti — ve Pazartesi hâlâ çalışan sürüm de o.*

---

## 2. Ortamı sıfırdan kurmak — bu kâğıt yeter

Üç program, iki model, bir repo, bir komut. Python yok, API key yok, hesap yok. İndirmeleri ofis
ağında değil evde yap (≈ 4.5 GB model + ~650 MB ChromaDB imajı).

| | Ne işe yarıyor | Nereden |
|---|---|---|
| **Ollama** | modelleri laptopunda çalıştırıyor | https://ollama.com/download |
| **Podman Desktop** | ChromaDB'yi ve lab'ın api'sini container olarak çalıştırıyor | https://podman-desktop.io |
| **Bruno** | gün boyu tıkladığın istek koleksiyonu | https://www.usebruno.com/downloads |

**Terminal (herhangi bir yerde):**

```bash
ollama pull gemma3:4b     # soruları cevaplıyor       (3.3 GB)
ollama pull bge-m3        # metni vektöre çeviriyor   (1.2 GB)
ollama list               # ikisini de görmelisin; başka model koyma, cevaplar değişir
```

**Terminal (repo kökü buradan çıkıyor):**

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-lab
cd amadeus-rag-lab
podman compose up --build     # ilk sefer: api imajını build eder (~1 dk), chromadb'yi çeker
```

git yoksa ZIP: https://github.com/kuthaygumus/amadeus-rag-lab/archive/refs/heads/main.zip —
çıkart, klasöre `cd` yap, gerisi aynı.

İki container ayağa kalkıyor: `chroma` (:8000) ve `api` (:3000). **Ollama container değil** —
host'ta kalıyor (GPU ve indirdiğin modeller orada); api ona `http://host.containers.internal:11434`
üzerinden ulaşıyor. Docker kullanıyorsan tek fark:
`OLLAMA_URL=http://host.docker.internal:11434 docker compose up --build`.

`up` ön planda kalıyor ve logları akıtıyor; işin bitince `Ctrl+C`, sonra `podman compose down`.
`down` container'ları siliyor, imajları ve `chroma-data` volume'unu tutuyor — `-v` ekleme, son
collection'ın o volume'da. Sonraki açılışlarda `--build` gerekmiyor: `podman compose up`.

**Bruno:** Bruno'yu aç → *Open Collection* → clone'un **içindeki** `bruno/amadeus-rag-lab`
klasörünü seç (repo kökünü değil, `bruno/`'yu değil) → sağ üstten environment **`local`** →
soldan **`00-health` › `health`** → Send.

`"status": "ready"` ve `checks` altında dört `"ok"` (`ollama`, `chatModel`, `embedModel`, `chroma`)
= hazırsın. Başka bir şey diyorsa eksik parçayı adıyla yazıyor — `missing — run: ollama pull …`
ise o komutu aynen koştur; `chroma unreachable` ise Podman Desktop'ta `chroma` yeşil mi bak;
Bruno "connection refused" diyorsa `podman compose up` çalışıyor mu, environment `local` seçili mi.

**Modelleri USB'den taşıyacaksan:** dosyalar macOS/Linux'ta `~/.ollama/models`, Windows'ta
`%USERPROFILE%\.ollama\models`. Önce Ollama'yı tamamen kapat (menü çubuğu / tepsi → Quit), sonra
`blobs/` ve `manifests/` içeriğini **birleştir, klasörü komple değiştirme** — üstüne kopyalamak
makinede zaten olan modellerin kaydını siler.

---

## 3. Altı Bruno klasörü — her biri bir gate

Her isteğin Bruno'da bir *Docs* sekmesi var: gate ve cevapta neye bakılacağı orada yazıyor.

| Klasör | Gate — bir önceki adım nerede yetmedi? | Ne görüyorsun |
|---|---|---|
| **`01-bare-llm`** | Model *benim* verimi biliyor mu? | `ask-about-kraken`: kendinden emin, yanlış, kaynaksız cevap. `stuff-the-whole-corpus`: korpusun tamamı tek prompt'ta — çalışıyor, soru başına ~21 000 token. |
| **`02-ingest`** | Doğru parçayı bulabilmek için veri nerede durmalı? | `ingest-q3`: read → chunk → embed → store, her aşama süreli; collection adı. `peek-chunks`: chunk'lar ChromaDB'de durdukları hâliyle. |
| **`03-retrieve`** | Modelin eline ne verilecek? | `retrieve`: top-K chunk, cosine score'larıyla. LLM yok. Doğru doküman, yanlış dilim. |
| **`04-query`** | Bu chunk'lardan cevaplayabiliyor mu? | `query`: cevap + `sources` + `debug.prompt` — modelin aldığı metin, kelimesi kelimesine. |
| **`05-chunking-and-noise`** | Dilim neden yanlıştı? | `1-reingest-structure` → `2-peek-again` → `3-query-again`: structure-aware kesim, boilerplate temizliği; aynı soru, **EUR 90**. Bunu bir splitter kazandı, model değil. |
| **`06-freshness`** | Neden fine-tune etmiyoruz? | `1-ingest-q2` → `2-query-q2` → **120**. `3-ingest-q3` → `4-query-q3` → **90**. Re-ingest, retrain değil. |

Her `/ingest` kendi collection'ına düşüyor, hiçbir şeyin üstüne yazılmıyor. İsim
`kraken-<edition>-<strategy>-<chunkSize>`; overlap varsa `-ov<n>`, boilerplate soyulmuşsa `-strip`
ekleniyor — örnek: `kraken-2026-Q3-structure-1500-strip`. `/retrieve` ve `/query` varsayılan
olarak son ingest edilen collection'a gidiyor; body'ye `collection` yazarak değiştirebilirsin.

---

## 4. Endpoint'ler — api `http://localhost:3000`

| | |
|---|---|
| `GET /health` | eksik olanı adıyla söylüyor (Ollama, bir model, Chroma) |
| `POST /chat` | `{question, stuffCorpus?, edition?}` — çıplak model ya da edisyonun tamamı tek prompt'ta |
| `POST /ingest` | `{edition, strategy: fixed\|recursive\|structure, chunkSize?, overlap?, stripBoilerplate?}` — `read → chunk → embed → store` süreleri, `chunks`, `sample`, `collection` döner |
| `GET /chunks` | `?source=<doküman id>&limit=` — Chroma'da ne duruyor (`source` = dosya adı, `.md`'siz) |
| `POST /retrieve` | `{question, topK?, threshold?, collection?}` — yalnız retrieval, LLM çağrısı yok |
| `POST /query` | aynı body — retrieve + generate; `{answer, abstained, sources:[{source, score, excerpt}], debug:{prompt, model, ms}}` |
| `DELETE /collections` | bütün collection'ları siler |

Varsayılanlar: `topK` 3, `threshold` 0.35; chunk boyutları fixed 280 / recursive 600 / structure
1500, overlap 0. `abstained: true` = eşiğin üstünde chunk yok, LLM hiç çağrılmadı. Henüz ingest
yoksa `/retrieve` ve `/query` **409** dönüyor. Korpus: `corpus/2026-Q2/` ve `corpus/2026-Q3/`
altında 28 sentetik markdown doküman; aradaki fark `corpus/DELTA.md`'de.

---

## 5. Eve götürülecek üç karar

**Karakter sayısına değil, dokümanın yapısına göre chunk'la.** Günün hatası hiçbir zaman doküman
değildi — header'ını kaybeden satırdı. Markdown başlıkları, tablo başlıkları, madde numaraları
bedava context; sabit boyutlu bir kesici bunları atar. Yazarın zaten yazdığını kullan. Lab'da bu
`strategy: "structure"`, prefix `[fare classic shorthaul > RULE 2A …]`; RULE 2A bölümü 1 140
karakter, o yüzden varsayılan chunkSize 1500 — 900'de tablo chunk'ı ilk 10'a girmiyor.

**Modele dokunmadan önce retrieval'ı doküman seviyesinde ölç.** Ucuz ve deterministik bir kontrol —
doğru doküman ilk 1'e, ilk 5'e girdi mi — hatanın nerede olduğunu herhangi bir prompt
mühendisliğinden önce söyler. Gün boyu hiçbir metrik *cevabı* puanlamadı; o ikinci eval'ı da kur,
çünkü birincisi ikincisinin bozuk olduğunu sana asla söylemeyecek.

**Deployment birimi yeniden ingest'tir, yeniden training değil.** Q3 Q2'nin yerine geçtiğinde
pipeline, model ve prompt değişmiyor; yeni edisyonun tek bir ingest'i, içinde ne olduğunu söyleyen
tek bir collection adı, ve bayat cevap gitti. Bu planlanabilen, geri alınabilen, diff'lenebilen bir
deploy. Fine-tune bunların hiçbiri değil.

Kısa rubrik — yukarıdan aşağı ilk "evet"te dur:

1. Cevap tek dokümanda ve doküman sabit mi → **prompt'a koy, bitti**
2. Korpus context window'a sığıyor ve maliyet umurunda değil mi → **prompt'a doldur**
3. Korpus değişiyor ya da sığmıyor mu → **RAG**
4. Modelin *davranışını* mı değiştirmek istiyorsun, bilgisini değil mi → **fine-tune**

---

## 6. Eğitmen tarafı ölçüm — salonda koşmadı

Lab reposunda `npm run eval` (`eval/results.md`): 20 gold soru, `bge-m3`, **doküman** seviyesinde
hit@1 / recall@5 / MRR. 12 Eylül 2026, eğitmenin M-serisi Mac'inde tek koşu — sayılar başka
makinede kayar; **yönü alıntıla, rakamı değil.** Bir soru 0.05 ediyor.

| strateji | hit@1 | recall@5 | MRR |
|---|---|---|---|
| fixed-280 | 0.700 | 0.917 | 0.817 |
| fixed-280 + overlap 60 | 0.550 | 0.883 | 0.717 |
| recursive-600 | 0.700 | 0.900 | 0.806 |
| structure-1500 | 0.700 | 0.833 | 0.781 |
| structure-1500 + strip | 0.650 | 0.850 | 0.766 |

Dürüst oku: structure-aware chunking doğru **dokümanı** daha sık bulmuyor. **Chunk'ı** düzeltiyor —
K satırını header'ıyla birlikte — ve "bilmiyorum"u EUR 90'a çeviren şey bu. Overlap her metrikte
kaybediyor. Doküman seviyesindeki tek bir sayının neden yetmediği ve `sources`'a hâlâ neden baktığın
da bu. Q3 chunk sayıları: fixed-280 → 294 · +overlap60 → 368 · recursive-600 → 197 ·
structure-1500 → 132.

---

## 7. Takılırsan

- **`00-health` `"ollama": "unreachable"` diyor ama terminalde `ollama list` çalışıyor** — Windows'ta
  Ollama bazen yalnız `127.0.0.1`'i dinliyor ve container içinden görünmüyor. Kullanıcı ortam
  değişkeni `OLLAMA_HOST=0.0.0.0` ekle (Settings → System → Environment variables, admin
  gerekmiyor), Ollama'yı tepsiden kapatıp aç, isteği tekrar gönder.
- **`podman compose up --build` `npm ci`'de patlıyor** — kurumsal proxy. İmajı evde bir kez build
  et; ofiste yeniden build edilmiyor, `podman compose up` yeter.
- **`podman compose` "no compose provider" diyor** — Podman Desktop'ı kur (içinde geliyor).
- **`/retrieve` ya da `/query` 409 dönüyor** — henüz ingest yok. `02-ingest` › `ingest-q3`'ü
  gönder. api, her restart'ta güncel collection'ı bilerek unutuyor; veri Chroma volume'unda duruyor,
  bir re-ingest ya da body'ye `collection` adı yazmak yeter.
- **`chroma` container'ı ayakta değil** — repo kökünde `podman compose down`, sonra `podman compose up`.
  Sağlık kontrolü: `curl -s http://localhost:8000/api/v2/heartbeat` (v2 — v1 410 Gone dönüyor).

Pazartesi, network yokken: Ollama'yı başlat → repo kökünde `podman compose up` → Bruno'da
`00-health` › `health`. Cevap geliyorsa koleksiyonun geri kalanı çalışır. `corpus/2026-Q3/` içinde
bir dokümanı değiştir, yeniden ingest et, yeniden sor — bütün döngü bu, ve artık senin.

---

Site: https://amadeus-rag-training.vercel.app/tr/
Lab (clone'ladığın repo): https://github.com/kuthaygumus/amadeus-rag-lab
Site kaynağı: https://github.com/kuthaygumus/amadeus-rag-training

*Kraken Air (XX) ve Wyvern Overseas Airways (YY) kurgusal havayollarıdır. `corpus/` altındaki her
doküman sentetik eğitim materyalidir; hiçbir Amadeus sistemi, müşterisi ya da production verisi bu
materyalin hiçbir yerinde geçmez.*
