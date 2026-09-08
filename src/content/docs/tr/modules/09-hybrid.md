---
title: "9. Hybrid, Rerank ve Contextual"
description: "Semantic yetmiyorsa?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Semantic yetmiyorsa?**

Ölçülmüş son retrieval sayısı **hit@1 0.800**: 154 structure-aware chunk üzerinde `bge-m3`, modül
7'nin sonunda in-memory retriever ile ölçüldü. Modül 8 aynı vektörleri ChromaDB'ye taşıdı ve yirmi
soruluk benchmark'ı bilerek yeniden çalıştırmadı — 154 vektörde arama zaten exhaustive, yani
sıralamanın aynı olması gerekir; ama modül 8 açıkça söylüyor: "gerekir" bir ölçüm değildir. Yani
0.800 hâlâ ölçülmüş son sayı ve bu modülün tartıştığı sayı o.

0.800, yirmi sorunun on altısı demek. Kaçan dördü ve doğru dokümanın gerçekte kaçıncı sırada olduğu:

| soru | tip | doğru dokümanın sırası |
|---|---|---|
| q05 | `tr_en` | 13 |
| q07 | `tr_en` | 7 |
| q19 | `multi_hop` | 6 |
| q14 | `en_en` | 2 |

Dördün üçü kıl payı kaçırmış değil. Üstteki doküman yanlış olduğunda generator da duraksamıyor —
haklıyken kullandığı aynı tonla, yanlış sayfadan cevap veriyor.

O boşluğa konacak standart bir liste var. Okuduğun her RAG yazısı aynı üç kelimeyi söylüyor: hybrid,
rerank, contextual. Vector search'ün yanına BM25 koy, sonuçları yeniden sıralaması için üstüne bir
model tak, her chunk'ın başına context yaz. Üçünü de aynı yirmi soruya karşı çalıştıracağız. Bunlardan
birini modül 7'de zaten yaptın, farkında değilsin. Diğer ikisi kaybediyor.

<div class="presenter-note">
Bu modülün ajandada <strong>38 dakikası</strong> var ve üç ölçümü — BM25 ile fusion, dört kurulumluk
rerank taraması ve soru bazındaki tablo — kesilmeyecek olanlar. Açılışta kaçan dört soruyu ekrana
koy: buradaki gate bir çökme değil, bir tavan. Sonra tahtaya <strong>hybrid · rerank ·
contextual</strong> yaz ve el kaldırt: "bu üçünden hangisi 0.800'ü yukarı taşır?" Rerank için
neredeyse bütün eller, hybrid için çoğu el kalkıyor. Sayıları kelimelerin yanına yaz ve orada bıraksın
— iki kez oraya işaret edeceksin. 4 dakika, laptoplar kapalı.
</div>

## BM25'i geri getir

BM25 modül 5'teki keyword retriever: nadir terimler yüksek skor alır, term frequency doyuma ulaşır,
uzun dokümanlar cezalandırılır. `H9 1487`'yi anında bulmuştu ve "iptal edersem ne öderim" sorusunda
tam sıfır almıştı. Şimdi onu dense retriever'ın kullandığı chunk'ların tam üstünde çalıştır — aynı
154 chunk, aynı index:

```python
bm25 = R.BM25(chunk_ids, chunk_texts)
```

**hit@1 0.300, MRR 0.467.** Türkçe sorgu / İngilizce doküman olan altı `tr_en` sorusunda **0.000**.
Zayıflamış değil. Sıfır. BM25'in `iptal`'den `Cancellation penalty`'ye giden bir yolu yok, chunking
de ona böyle bir yol vermedi.

Onu yenen Türkçe değil. Sorgunun da dokümanın da Türkçe olduğu, kelimelerin ortak olduğu dört `tr_tr`
sorusunda BM25 **0.750** alıyor. Geçemediği şey, sorgu ile doküman arasındaki dil sınırı.

## Reciprocal rank fusion ve neden seyreltiyor

Fusion herkesin ilk uzandığı çözüm ve gerçekten zarif bir fikir. İki retriever'ın skorlarının
kıyaslanabilir olması gerekmiyor — cosine similarity ile BM25 skorunun ortak birimi yok — o yüzden
skorları atıp sadece sıraları tutuyorsun:

```python
def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + position)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
```

Her retriever her dokümana `1 / (k + sıra)` puan veriyor. `k = 60` eğrinin tepesini düzleştiriyor;
birinci sıra ile üçüncü sıra neredeyse aynı değerde, kuyruk ise hiçbir şey. İki retriever'ın da
beğendiği doküman, sadece birinin beğendiğini geçiyor.

Dense ile BM25'i aynı chunk'lar üzerinde fuse et: **hit@1 0.450, MRR 0.586.** Dense tek başına 0.800
idi. İkinci retriever bize 0.350'ye mal oldu.

Kaybın nereye gittiğine bak. Altı `tr_en` sorusunda dense tek başına 0.667, BM25 0.000 alıyor — ve
fuse edilmiş sıralama **0.000** alıyor. Fusion bu ikisinin ortalamasını almadı. Başarısızlığı devraldı.

Sebep formülün içinde duruyor. RRF'in bir retriever'ın ne kadar iyi olduğuna ya da bu sorguda ne kadar
emin olduğuna dair hiçbir fikri yok. BM25'in birinci sırası, dense retriever'ın birinci sırasıyla aynı
`1/61`'i katıyor. O altı soruda BM25'in sıralaması gürültü ve o gürültü tam ağırlıkla oy kullanıyor.
Anlamadığı sorgularda daha az oy kullanamaz, çünkü teslim ettiği tek şey bir sıralama — ve sıralama
her zaman bir fikir gibi görünür.

Kimsenin yüksek sesle söylemediği önkoşul bu: **reciprocal rank fusion her iki girdinin de kendi
başına sağlam olduğunu varsayar.** Farklı sorularda düşen iki makul retriever birbirinin açığını
kapatır. İyi bir retriever ile korpusunun üçte birinde sistematik olarak yanılan bir retriever ise
sadece ortalamaya gider, ve ortalama almak tamir değildir.

<div class="presenter-note">
Fusion hücresini çalıştırmadan önce taahhüt al: "Dense 0.800, BM25 0.300. Fuse edince — 0.800'ün
üstü mü, arası mı, 0.300'ün altı mı?" Kabaca oyla. Salonun çoğu üstü der, çünkü fusion kulağa toplama
gibi geliyor. Sonra <strong>0.450</strong>'yi göster ve beş saniye hiçbir şey söyleme. Devamı `tr_en`
satırı: 0.667 ile 0.000, fuse edilince 0.000. Ollama düştüyse bu bölümdeki her sayı
<code>eval/RESULTS.md</code> içinde; oradan oku ve devam et — BM25'in kendisi hiç model çağrısı
istemiyor, yani hücrenin o yarısı her koşulda çalışır. 8 dakika, üstteki BM25 bölümü dahil.
</div>

## Chunking BM25'e ne yaptı

BM25 kötü bir retriever değil, ama iyi olduğu koşul bu değil. **Tam dokümanlar** üzerinde — yani
modül 5'teki index, hiç chunking yokken — BM25 hit@1 **0.400**, MRR **0.515** aldı. Oradaki dört
exact-token sorusunda, yani uçuş kodları, bülten id'leri ve `KSHEU26` gibi fare basis kodlarında
**0.750**'ye çıktı.

Genelde "BM25 tanımlayıcılarda embedding'i yener" diye aktarılan sayı bu. Bu korpusta yenmiyor: aynı
tam dokümanlar üzerinde `bge-m3` genelde **0.550**, aynı dört exact-token sorusunda **1.000** alıyor.
BM25 elde tutulmaya değecek kadar yakın — ve kazanması beklenen tek koşulda geride.

BM25'in tam dokümanlarda gerçekten sahip olduğu asimetri dil asimetrisi, üstelik keskin. Sorgunun da
dokümanın da Türkçe olduğu dört `tr_tr` sorusunda **1.000** alıyor — aynı dört soruda `bge-m3` ile
başa baş. Türkçe sorgu / İngilizce doküman olan altı `tr_en` sorusunda **0.000**, `bge-m3`'ün
0.333'üne karşı. Kullanabildiği tek şey kelime örtüşmesi; yani aynı korpusta hem en iyi hem en kötü
hâlinde ve farkı, dokümanın hangi dilde yazıldığı belirliyor.

Chunking o kadarını da aldı. 154 chunk üzerinde BM25'in exact-token skoru 0.750'den **0.500**'e,
genel MRR'ı 0.515'ten 0.467'ye düşüyor; iki yoldan. Uzunluk normalizasyonu çalışmayı bırakıyor: BM25
doküman uzunluğunu ortalamaya bölüyor, her chunk aynı boyda olunca bu terim sabite dönüşüyor ve
güvendiğin bir ayırt edici sessizce yok oluyor. Bir de kanıt bölünüyor — altı terimlik bir sorgu
eskiden altı terimi tek dokümanın içinde biriktiriyordu; chunking'den sonra terimler üç ayrı chunk'a
dağılıyor ve hiçbiri kayda değer puan toplamıyor.

Yani dürüst cümle "hybrid search abartılıyor" değil. Dürüst cümle şu: **chunk'lar üzerinde, diller
arası bir korpusta hybrid search, dense'in 0.800'üne karşı 0.450 ölçüldü.** Index birimini değiştir,
bu bölümdeki her sayı oynar — bugün ikinci kez bir retrieval kararı başka bir retrieval kararına
bağlı çıkıyor. Bu korpusta oynamayan tek şey, iki retriever'dan hangisinin önde olduğu.

## Reranker ve modülü belirleyen sayı

Reranker, en üstteki adayları alıp bir dil modeline değerlendirtiyor. Bizimki `qwen2.5:3b`, her adayı
tek tek 0-10 arası puanlıyor — aday başına bir model çağrısı, eşitlikte retriever'ın orijinal sırası
korunuyor, yani modelin gerçekten bir fikri olduğunda bir dokümanı yerinden oynatabiliyor. Aynı
reranker, kalitesi bilerek farklı tutulmuş dört retrieval kurulumunda.

| retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra | sonuç |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.503 | 0.543 | yardım etti |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.492 | 0.537 | yardım etti |
| `bge-m3` + fixed-280 | 0.700 | 0.550 | 0.814 | 0.712 | zarar verdi |
| `bge-m3` + structure-aware | **0.800** | 0.600 | 0.844 | 0.717 | zarar verdi |

İki zayıf kurulumu yukarı çekti, iki güçlü kurulumu aşağı indirdi. Tabloyu neyin ikiye böldüğüne bak:
chunking değil — her iki chunking stratejisi de tablonun iki yakasında birden var — **embedder**.

**Reranker seviyeler, ve kendi tavanına seviyeler.** Rerank'ten önce dört kurulum 0.350 ile 0.800
arasındaydı; aralık 0.450. Sonra 0.400 ile 0.600 arasında; aralık 0.200. Aynı model alttakini yukarı,
üsttekini aşağı çekti. 3B bir modelin alaka konusundaki fikri bu korpusta aşağı yukarı 0.40 ile 0.60
arası değerde. Retriever'ın bundan kötüyse modelin fikrini dayatmak bir yükseltmedir. Retriever'ın
zaten daha iyiyse dayatmak sadece kaybettirir. Üçüncü bir sonuç yok.

Soru bazında bakınca hiç ortalama almadan aynı şey görünüyor. En güçlü kurulumda yirmi sorunun
**on altısında** doğru doküman zaten birinci sıradaydı ve reranking bunların **beşini** aşağı taşıdı:
q06 3. sıraya, q10 2., q11 4., q16 2., q17 3. sıraya. Doğru dokümanı birinci sırada olmayan **dört**
sorunun **ikisini** yukarı çekti: q14 2'den 1'e, q19 6'dan 5'e. hit@1'i değiştiren tek hamle q14;
0.800 böyle 0.600 oluyor. İki tamir satın almak için bozuk olmayan beş şeyi bozdun.

Bu negatif sonucun bedeli: **soru başına 8 model çağrısı, kurulum başına 160** ve M serisi bir Mac'te
kurulum başına **49.3 ile 87.6 saniye** arası.

<div class="presenter-note">
Önce mekanizmayı notebook'ta kur — tek sorgu, altı aday, skorlar ekranda — ve salon bir reranker'ın
kütüphane değil, içinde prompt olan bir for döngüsü olduğunu görsün. Skor sütunundaki eşitliklere
işaret et: 3B bir modelden tam sayı isteyince elinde birkaç farklı değer kalıyor ve eşitlikte
retriever'ın sırası korunuyor. Sonra <code>exercises/m9_rerank_trade.py</code>'ı başlat ve o çalışırken
tahmini al: dört kurulumun hepsine mi yarar, hiçbirine mi, bazılarına mı? Bittiğinde ağzında
gevelemeyeceğin cümle: <strong>reranker, retriever'ın reranker'dan kötüyse kazandırır; retriever'ın
daha iyiyse kaybettirir.</strong> Bir kez, yavaşça söyle ve tahtadaki el sayılarına geri işaret et.
Biri "gerçek reranker chat modeli değil, cross-encoder olur" diye itiraz edecek. Haklı — dürüstçe
cevap ver: onu ölçmedik ve denenecek ilk şey o. Onun için sayı uydurma. Yavaş laptopta
<code>--quick</code> çağrıların dörtte biriyle bitiyor ve en üstte REDUCED yazıyor; argümanın ihtiyaç
duyduğu satırlar zaten iki güçlü satır. 20 dakika, çalıştırma dahil.
</div>

## Nasıl sorduğun — n = 5 ile ölçülmüş bir kenar notu

Reranker'ın daha ucuz bir şekli var: tek çağrı, altı pasajın hepsi birden, modelden sırala. Notebook
bu çağrıyı tek bir soruda yapıp dönen cevabı basıyor. Bizim çalıştırmamızda model `1,4,2,5` dedi —
altı pasaj için dört sayı.

Altı pasajı sıralamak, altı karşılaştırmayı aynı anda akılda tutup bir permütasyon üretmek demek. 3B
model bunda kötü, "bu tek pasaj ne kadar alakalı" sorusunda iyi — çünkü o, sabit bir ölçeğe karşı tek
bir yargı. Aynı model, aynı pasajlar, aynı bilgi; fark sorunun şeklinde.

Bu karşılaştırmanın elimizdeki tek sayısal hâli, emekliye ayrılmış on dokümanlık beş soruluk bir probe
korpusundan geliyor: listwise, "bu altısını sırala", **2/5** ve MRR 0.600; pointwise, "bu tek pasajı
0-10 arası puanla", **5/5** ve MRR 1.000. Bu `n = 5`. Yönü gerçek, büyüklüğü kanıtsız kabul et — ve
bunu "pointwise rerank işe yarıyor" kanıtı olarak okuma. Aynı pointwise reranker bu yirmi soruda güçlü
kurulumu 0.800'den 0.600'e indirdi. Bu, prompt'un şekli hakkında bir kanıt; başka bir şey değil.

## Contextual retrieval — onu zaten yaptın

Contextual retrieval, her chunk'ın başına onu tek başına anlaşılır kılacak kadar çevre bilgisi
koymak demek; genelde bir modele "bu chunk dokümanın neresinde" diye bir satır yazdırarak.
Structure-aware chunking her chunk'ın başına zaten kendi başlık yolunu koyuyor: aynı mekanizma, ama
modelle üretilmiş değil dokümandan alınmış ve inference maliyeti sıfır. K satırı ile sütun başlığının
aynı chunk'ta hayatta kalmasının ve modelin iptal cezasını yan sütundan değil doğru sütundan
okumasının sebebi bu. Modül 7'deki 0.800 aslında contextual retrieval'ın kazancı, cebe girmiş
durumda. Üstüne bir de modelle context üretmek denenmeye değer — ama başlıkları olan dokümanlarda,
Markdown'ın sana bedavaya vermediği bir şey satın aldığından emin ol.

## Ne çalıştırıyorsun

**Mekanizma — notebook'ta.** `notebooks/06_hybrid_rerank_contextual.py` dosyasını VS Code'da aç ve
blokları Shift+Enter ile çalıştır (Microsoft Python eklentisi; bu eğitimde Jupyter kurulumu yok).

```bash
ollama serve                       # çalışmıyorsa ikinci bir terminalde
python scripts/verify_setup.py     # devam etmeden önce yeşil yazmalı
```

- **ne görmelisin** — `154 chunks indexed both ways`, ardından üçlü karşılaştırma: dense 0.800,
  BM25 0.300, RRF 0.450. Sonra tek bir sorgunun altı adayı 0-10 arası puanlanmış, aldığı saniyeyle
  birlikte.
- **kabaca ne kadar sürer** — 154 chunk üzerinde bir embedding geçişi artı altı model çağrısı. Birkaç
  dakika, çoğu embedding. BM25 ve fusion hiç model çağrısı istemiyor.

**Ölçüm — tek komut.**

```bash
python exercises/m9_rerank_trade.py            # tam tarama
python exercises/m9_rerank_trade.py --quick    # yavaş laptop: sadece güçlü kurulumlar, depth 4
```

- **ne görmelisin** — her retrieval kurulumu için bir satır, her satırda kendi çağrı sayısı ve
  saniyesi: iki `nomic` satırında HELPED, iki `bge-m3` satırında hurt. Sonra soru bazındaki özet:
  16 soruda doğru doküman birinci sıradaydı, reranking bunların 5'ini aşağı taşıdı.
- **maliyeti, dersin parçası olduğu için** — 4 kurulum × 20 soru × 8 aday = **640 model çağrısı**,
  kurulum başına 160. Ölçülen süreler, kurulum başına: M serisi bir Mac'te 49.3 sn, 52.4 sn, 65.2 sn
  ve 87.6 sn — toplamda dört dakikanın biraz üstünde model zamanı, CPU-only bir laptopta epey daha
  uzun. `--quick` iki güçlü kurulumu depth 4 ile çalıştırıyor — 160 çağrı — ve en üste REDUCED
  yazıyor, böylece onun sayıları bu sayfadakilerle karıştırılmıyor.

Her şey ortak koddan geliyor. Burada yeni bir bağımlılık yok:

```python
import sys; sys.path.insert(0, "eval")
from pathlib import Path
import chunking as C, metrics, retrieval as R

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("eval/gold_questions.jsonl")

chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")   # 154 chunks
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)      # bge-m3, one embed call per chunk
bm25  = R.BM25(chunk_ids, chunk_texts)

runs = {"dense": {}, "bm25": {}, "rrf": {}}
for q in questions:
    d, b = dense.rank(q["query"]), bm25.rank(q["query"])
    runs["dense"][q["id"]], runs["bm25"][q["id"]] = d, b
    runs["rrf"][q["id"]] = R.rrf([d, b])

for name, run in runs.items():
    # score documents, not chunks: collapse each chunk ranking to its parent documents first
    scored = metrics.evaluate({k: C.to_documents(v) for k, v in run.items()}, questions)
    print(f"{name:<6} hit@1 {scored['hit@1']:.3f}  MRR {scored['MRR']:.3f}")

# the reranker itself: one model call per candidate, on one question
q = next(x for x in questions if x["id"] == "q05")
candidates = runs["dense"][q["id"]][:6]               # 6 candidates -> 6 model calls
print(R.pointwise_rerank(q["query"], candidates, chunks))
```

## Sayılar ne dedi

<div class="measured">

| aynı 154 structure-aware chunk üzerinde retriever | hit@1 | MRR |
|---|---|---|
| dense, `bge-m3` | **0.800** | **0.844** |
| BM25 | 0.300 | 0.467 |
| ikisinin RRF'i | 0.450 | 0.586 |
| 6 `tr_en` sorusunda BM25 | 0.000 | — |
| 6 `tr_en` sorusunda RRF | 0.000 | — |

| tam dokümanlar, hiç chunking yokken | hit@1 | MRR | exact_token | tr_tr | tr_en |
|---|---|---|---|---|---|
| dense, `bge-m3` | 0.550 | 0.654 | **1.000** | 1.000 | 0.333 |
| BM25 | 0.400 | 0.515 | 0.750 | 1.000 | **0.000** |

| aynı reranker, dört retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra |
|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.503 | **0.543** |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.492 | **0.537** |
| `bge-m3` + fixed-280 | 0.700 | 0.550 | 0.814 | 0.712 |
| `bge-m3` + structure-aware | **0.800** | 0.600 | 0.844 | 0.717 |

| en güçlü kurulum, soru bazında | adet |
|---|---|
| doğru doküman zaten birinci sırada | 16 |
| bunlardan reranking'in aşağı taşıdığı | 5 — q06, q10, q11, q16, q17 |
| doğru doküman birinci sıranın altında | 4 |
| bunlardan reranking'in yukarı çektiği | 2 — q14, q19 |

| depth 8'de bir rerank geçişinin maliyeti | |
|---|---|
| soru başına model çağrısı | 8 |
| kurulum başına, 20 soru | 160 |
| dört kurulumluk tam tarama | 640 |
| kurulum başına ölçülen süre | 49.3 sn – 87.6 sn |

</div>

Korpus: 28 doküman, 78,310 karakter. Gold set: 20 soru, 6'sı Türkçe sorgu / İngilizce doküman.
Generation ve reranking `qwen2.5:3b`, embedding aksi belirtilmedikçe `bge-m3`, hepsi Ollama üzerinden
lokal. **Yirmi soru iki tasarım arasında karar verdirir, genel bir iddiayı taşımaz**; 0.05'in altındaki
bir fark bu örneklemin gürültüsünün içinde. Bu sayfa "bu teknikler kötüdür" demiyor. Bu korpusta, bu
embedder ve bu reranker ile kaybettiklerini söylüyor ve her birinin hangi önkoşula ihtiyaç duyduğunu
adıyla koyuyor.

## Daha derine

RRF'teki `k = 60` bir yumuşatma sabiti. `k` küçükse birinci sıra baskın olur, fusion "hangi retriever
daha eminse ona güven" gibi davranır; `k` büyükse eğri düzleşir ve fusion bütün liste üzerinde bir
popülerlik oylamasına döner. 60 orijinal TREC çalışmasından geliyor ve neredeyse hiç ellenmiyor.
Burada onu ayarlamak bizi kurtarmazdı: yirmi sorunun altısında 0.000 alan bir sıralamayı hiçbir `k`
değeri işe yarar hale getirmez, çünkü problem ağırlık eğrisi değil, girdinin o sorgularda hiç sinyal
taşımaması. Dengesiz bir çiftle fusion yapmak sorgu bazlı ağırlıklandırma ister — fuse etmeden önce
"bu, BM25'in cevaplayabileceği türden bir sorgu mu" kararını vermek — ve o router da kurup ölçmen
gereken başka bir model.

Bizim reranker'ımız pasaj puanlayan bir chat modeli; kurulumu en kolay, reranker denilebilecek en
zayıf şey. Production cevabı cross-encoder: sorguyu ve pasajı **birlikte**, tek forward pass'te okuyup
tek bir alaka skoru üreten, prompt'la göreve ikna edilmiş değil alaka etiketleriyle eğitilmiş bir
model. `bge-reranker-v2-m3` bunun çok dilli olanı ve zaten kullandığımız embedder ile aynı aileden.
Daha güçlü bir hakem ve tavanı 0.800'ün üstüne taşıması makul — ama biz onu ölçmedik, o yüzden bunu
sonuç değil, deneyi belli bir hipotez olarak kabul et. Ders iki durumda da aynı: onun da bir tavanı
var ve o tavanın senin retriever'ının üstünde mi altında mı olduğunu ölçmen gerekiyor.

Seviyeleme sonucu reranking'in ötesine genelleniyor. Üstteki bir sıralamayı ezen her aşama, girdi ne
olursa olsun kendi doğruluğunu çıktıya dayatır. "Sen bir reranker ekle" tavsiyesinin "sen bir cache
ekle" kadar kötü olmasının sebebi bu: önkoşul, sisteminle ilgili ancak ölçerek öğrenebileceğin bir
gerçek. Sıra da önemli. Modül 7 chunking'den başka hiçbir şeyi değiştirmeden hit@1'i 0.550'den
0.800'e taşıdı — o iki sayının ikisi de `bge-m3`. Modül 6'nın kaldıracı diğeriydi: aynı
structure-aware chunk'lar üzerinde `nomic-embed-text` 0.350 alırken `bge-m3` 0.800 alıyor. Bu
modüldeki her şey iki kaldıraç da çekildikten sonra geldi, yani tam da bu tekniklerin maliyet yazdığı
noktada. Dört satırlık tablonun iki satırı gerçek bir iyileşmeye bakarak reranker'ı production'a
alırdı — `nomic` cidden 0.350'den 0.450 ve 0.400'e çıkıyor — ve tasarım yine yanlış olurdu, çünkü doğru
hamle embedder'ı düzeltmekti.

On milyon dokümanda tablo değişiyor ve BM25 bambaşka bir sebeple geri geliyor. Orada her sorguda her
chunk'ı embed edip skorlayamazsın; ucuz bir birinci aşama birkaç yüz aday döndürür, pahalı bir ikinci
aşama onları sıralar. Inverted index üzerinde BM25 güçlü bir birinci aşamadır — milisaniyenin altında,
tanımlayıcılarda birebir, güncellemesi kolay — arkasında da top 200 üzerinde bir cross-encoder. Bu,
bizim ölçtüğümüz hybrid değil: BM25 nihai cevaba oy vermiyor, aday üretiyor ve metriği hit@1 değil
recall@k. Aynı bileşen, farklı iş, farklı metrik. 28 dokümanda o mimarinin yapacak işi yok ve sorgu
başına saniyeler yazıyor.

<div class="presenter-note">
Geriden geliyorsan üç ölçümü de koru, onun yerine metni kes — n=5 listwise kenar notu tek cümleye
iner, "Daha derine" zaten okuma malzemesi. Kapatmadan önce tahtadaki el sayılarına işaret et ve üç
kararı oku: hybrid kaybetti, rerank seviyeledi, contextual zaten modül 7'de yapılmıştı. Sonra hâlâ
çalışmayanı söyle: açılış slaytındaki dört soruya geri dön, q19 bir multi-hop sorusu ve bu modüldeki
her şeyden sonra hâlâ orada. Modül 10 tam orada ve tek bir hücreyle gösterilebilecek bir
başarısızlıkla açılıyor. 6 dakika.
</div>

## Çıkış cümlesi

> Ölçtüğümüz kadarıyla hiçbiri düz retrieval'ı geçemedi. Ama multi-hop hâlâ düşüyor.
