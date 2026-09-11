---
title: "9. Hybrid, Rerank ve Contextual"
description: "Semantic yetmiyorsa?"
---

## Gate sorusu

> **Semantic yetmiyorsa?**

Ölçülmüş son retrieval sayısı **hit@1 0.750**: 154 structure-aware chunk üzerinde `bge-m3`. Modül
7'nin merdiveni bir basamak daha çıkmıştı — hukuk dipnotunu atınca 153 chunk üzerinde 0.800'e
ulaşıyordu — ama bu modüldeki her ölçüm dipnotu atılmamış 154 chunk üzerinde çalışıyor, yani
kıyaslanabilir sayı 0.750. Modül 8 aynı vektörleri ChromaDB'ye taşıdı ve bilerek yeniden ölçmedi:
154 vektörde arama zaten exhaustive ve "aynı olması gerekir" bir ölçüm değildir.

0.750, yirmi sorunun on beşi demek. Beşi kaçıyor, artık üç tipten ikişer sorunun o temiz düzeninde
değil:

| soru | tip |
|---|---|
| q05 | `tr_en` |
| q07 | `tr_en` |
| q11 | `en_en` |
| q14 | `en_en` |
| q19 | `multi_hop` |

İki `tr_en`, iki `en_en`, bir `multi_hop` — ikizi q20 artık birinci sırada. Bu modüldeki her koşu hâlâ
aynı beşini geri veriyor — `--fusion` onları kendi tablosunun altına basıyor. Bunlardan yalnızca
birinin sırası yazdırılıyor: q05'in `sop_misconnect_v4`'ü **28 içinde 12. sırada**, yani kıl payı
kaçırmak değil. `UNVERIFIED: diğer dördünün sıraları — bu repoda hiçbir komut bu koşu için soru
bazında sıra yazdırmıyor.` Üstteki doküman yanlış olduğunda generator da duraksamıyor; haklıyken
kullandığı aynı tonla, yanlış sayfadan cevap veriyor.

Okuduğun her RAG yazısı o boşluk için aynı üç kelimeyi söylüyor: hybrid, rerank, contextual — vector
search'ün yanına BM25, sonuçları yeniden sıralayacak bir model, her chunk'ın önüne bir context satırı.
Üçünü de aynı yirmi soruya karşı çalıştıracağız. Bunlardan birini modül 7'de zaten yaptın, farkında
değilsin. Diğer ikisi kaybediyor.

<div class="presenter-note">
Bu modülün ajandada <strong>38 dakikası</strong> var ve üç ölçümü — BM25 ile fusion, dört kurulumluk
rerank taraması ve soru bazındaki tablo — kesilmeyecek olanlar. Açılışta kaçan beş soruyu ekrana
koy: buradaki gate bir çökme değil, bir tavan. Sonra tahtaya <strong>hybrid · rerank ·
contextual</strong> yaz ve el kaldırt: "bu üçünden hangisi 0.750'yi yukarı taşır?" Rerank için
neredeyse bütün eller, hybrid için çoğu el kalkıyor. Sayıları kelimelerin yanına yaz ve orada bıraksın
— iki kez oraya işaret edeceksin. 4 dakika, laptoplar kapalı.
</div>

## BM25'i geri getir

BM25 modül 5'teki keyword retriever: nadir terimler yüksek skor alır, term frequency doyuma ulaşır,
uzun dokümanlar cezalandırılır. `XX 1487`'yi anında bulmuştu ve "iptal edersem ne öderim" sorusunda
tam sıfır almıştı. Şimdi onu dense retriever'ın kullandığı chunk'ların tam üstünde çalıştır — aynı
154 chunk, aynı index.

**VS Code — `notebooks/06_hybrid_rerank_contextual.py`, `chunk_ids` ile `chunk_texts`'i az önce
üretmiş olan indeksleme bloğunda:**

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
skorları atıp sadece sıraları tutuyorsun.

**`eval/retrieval.py` — fusion adımının tamamı:**

```python
def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + position)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
```

Her retriever her dokümana `1 / (k + sıra)` puan veriyor. `k = 60` seni koruyor sanmadan önce
aritmetiği yap: birinci sıra `1/61` ediyor, üçüncü sıra neredeyse aynı ve 154. sıra hâlâ **birinci
sıranın %28'i** kadar. Üstelik bizim iki girdimiz de aynı 154 chunk'ın tam permütasyonu, yani her
chunk iki sıralamada birden duruyor — burada "sadece bir retriever bulmuş" diye bir durum hiç yok.
Fusion iki yumuşatılmış ters sıranın toplamına dönüşüyor ve zayıf sıralama her dokümana, listenin en
dibine kadar oy veriyor.

Dense ile BM25'i aynı chunk'lar üzerinde fuse et: **hit@1 0.450, MRR 0.579.** Dense tek başına 0.750
idi; ikinci retriever bize 0.300'e, yani altı soruya mal oldu. Altı `tr_en` sorusunda dense 0.667,
BM25 0.000 alıyor ve fuse edilmiş sıralama **0.000** alıyor. Fusion bu ikisinin ortalamasını almadı;
başarısızlığı devraldı ve başka türlüsü elinden gelmez, çünkü BM25'in teslim ettiği tek şey bir
sıralama ve sıralama her zaman bir fikir gibi görünür. Kimsenin yüksek sesle söylemediği önkoşul bu:
**reciprocal rank fusion her iki girdinin de kendi başına sağlam olduğunu varsayar.** İyi bir
retriever ile corpus'un üçte birinde sistematik olarak yanılan bir retriever sadece ortalamaya gider
ve ortalama almak tamir değildir.

Fusion bir kez bir şey kazanıyor ve bunu kutlamak yerine fiyatlamak gerekiyor. **Doküman düzeyinde**
— yani BM25'in çalışacak kadar metin bulduğu tam dokümanlarda — sayfadaki en iyi recall@5 RRF'in
oluyor: dense'in 0.717'sine karşı **0.733**. Ama o **0.016 recall@5'i 0.100 hit@1 ile** satın alıyor,
0.600'den 0.500'e. Kazanç bir sorunun küçük bir kesri. Bedel tam iki soru. Orada bile `tr_en` satırı,
dense'in 0.333'üne karşı 0.000.

<div class="presenter-note">
Fusion hücresini çalıştırmadan önce taahhüt al: "Dense 0.750, BM25 0.300. Fuse edince — 0.750'nin
üstü mü, arası mı, 0.300'ün altı mı?" Kabaca oyla. Salonun çoğu üstü der, çünkü fusion kulağa toplama
gibi geliyor. Sonra <strong>0.450</strong>'yi göster ve beş saniye hiçbir şey söyleme. Devamı `tr_en`
satırı: 0.667 ile 0.000, fuse edilince 0.000. Ollama düştüyse bu bölümdeki her sayı
<code>eval/RESULTS.md</code> içinde ve <code>USE_CACHED=1</code> bütün notebook'u
<code>cached_runs.json</code>'dan tekrar oynatıyor; oradan oku ve devam et — BM25'in kendisi hiç
model çağrısı istemiyor, yani hücrenin o yarısı her koşulda çalışır. 8 dakika, üstteki BM25 bölümü
dahil.
</div>

## Chunking BM25'e ne yaptı

BM25 kötü bir retriever değil, ama iyi olduğu koşul bu değil. **Tam dokümanlar** üzerinde — yani
modül 5'teki index, hiç chunking yokken — hit@1 **0.400**, MRR **0.515** ve dört exact-token
sorusunda **0.750** alıyordu. Genelde "BM25 tanımlayıcılarda embedding'i yener" diye aktarılan sayı
bu. Burada yenmiyor: aynı tam dokümanlar üzerinde `bge-m3` genelde **0.600**, aynı dört soruda
**1.000** alıyor. Gerçekten sahip olduğu asimetri dil asimetrisi — dört `tr_tr` sorusunda **1.000**,
altı `tr_en` sorusunda **0.000**; aynı corpus'ta hem en iyi hem en kötü hâlinde ve farkı, dokümanın
hangi dilde yazıldığı belirliyor.

Kalanını da chunking aldı: 154 structure-aware chunk üzerinde exact-token skoru 0.750'den **0.500**'e,
MRR'i 0.515'ten **0.467**'ye düşüyor. Sebebi konusunda dikkatli ol, çünkü modül 5 sana bunun daha
güçlü hâlini anlatmıştı. Orada, 294 fixed-280 chunk üzerinde, her chunk gerçekten ortalama uzunlukta
ve BM25'in uzunluk terimi sabite dönüşüyor; structure-aware chunk'ların uzunlukları ise hâlâ
değişiyor, yani burada sinyal yok olmuyor, sıkışıyor. İkinci sebep iki durumda da geçerli: altı
terimlik bir sorgu eskiden altı terimi tek dokümanın içinde biriktiriyordu, şimdi terimler üç ayrı
chunk'a bölünüyor ve hiçbiri kayda değer puan toplamıyor.

Yani dürüst cümle "hybrid search abartılıyor" değil. Dürüst cümle şu: **chunk'lar üzerinde, diller
arası bir corpus'ta hybrid search, dense'in 0.750'sine karşı 0.450 ölçüldü.** Index birimini değiştir,
buradaki her sayı oynar — bugün ikinci kez bir retrieval kararı başka bir retrieval kararına bağlı
çıkıyor.

## Reranker ve modülü belirleyen sayı

Reranker, en üstteki adayları alıp bir dil modeline değerlendirtiyor. Bizimki `qwen2.5:3b`, her adayı
tek tek 0-10 arası puanlıyor — aday başına bir model çağrısı, eşitlikte retriever'ın orijinal sırası
korunuyor, yani modelin gerçekten bir fikri olduğunda bir dokümanı yerinden oynatabiliyor. Aynı
reranker, kalitesi bilerek farklı tutulmuş dört retrieval kurulumunda.

<div class="measured">

| retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra | sonuç |
|---|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.515 | **0.544** | HELPED, +0.029 MRR |
| `nomic` + structure-aware | 0.350 | **0.400** | 0.490 | **0.540** | HELPED, +0.050 |
| `bge-m3` + fixed-280 | 0.700 | 0.500 | 0.817 | 0.680 | hurt, −0.137 |
| `bge-m3` + structure-aware | **0.750** | 0.600 | 0.817 | 0.725 | hurt, −0.092 |

</div>

İki zayıf kurulumu yukarı çekti, iki güçlü kurulumu aşağı indirdi; iki yönde de tek bir istisna yok.
Tabloyu neyin ikiye böldüğüne bak: chunking değil — her iki chunking stratejisi de tablonun iki
yakasında birden var — **embedder**.

**Reranker düzlüyor ve kendi tavanına düzlüyor.** Rerank'ten önce dört kurulum 0.350 ile 0.750
arasındaydı; aralık 0.400. Sonra 0.400 ile 0.600 arasında; aralık 0.200. Aynı model alttakini yukarı,
üsttekini aşağı çekti. Ama bandı söylerken koşulunu da söyle: bu kalitede bir retriever'dan gelen
adaylar verildiğinde, bu reranker'ın çıktısı eline ne geçerse geçsin 0.400 ile 0.600 arasına düşüyor.
O bant modelin değil, **çiftin** özelliği — reranker yalnızca retriever'ın kendi ilk 8'ini yeniden
diziyor, yani daha kötü bir aday listesi onu da aşağı çekerdi. Retriever'ın bu çiftten kötüyse modelin
fikrini dayatmak bir yükseltmedir. Zaten daha iyiyse dayatmak sadece kaybettirir. Üçüncü bir sonuç yok.

Soru bazında bakınca hiç ortalama almadan aynı şey görünüyor. En güçlü kurulumda yirmi sorunun
**on beşinde** doğru doküman zaten birinci sıradaydı ve reranking bunların **dördünü** aşağı taşıdı:
q06, q10, q16, q18. Doğru dokümanı birinci sırada olmayan **beş** sorunun **birini** yukarı çekti,
q14 — ve sonra sütunundaki 0.600, birinci sırada on iki soru demek: on beş eksi dört artı bir. Bir
gerçek kazanca karşı dört kırık: tablodaki −0.150 hit@1'in soru soru yazılmış hâli.
`UNVERIFIED: q14'ün hangi sıraya taşındığı — exercise yönü ve adedi yazdırıyor, konumu değil.`

Bedeli: **soru başına 8 model çağrısı, kurulum başına 160, tablonun tamamı için 640.** Bu koşuda dört
kurulum, yazdırıldıkları sırayla **191 sn, 127 sn, 51 sn ve 71 sn** sürdü.
`UNVERIFIED: taramanın toplam saati ve --quick'in süresi — script yalnızca bu dört kurulum-başı
süreyi yazdırıyor, bu koşuda ikisi de kaydedilmedi.` Modelleri zaten bellekte olan tek bir M-serisi
Mac'te tek bir koşu: bir büyüklük mertebesi, bir spesifikasyon değil. Çağrı sayısı sabit, saat değil
ve CPU-only bir laptop epey daha yavaş.

<div class="presenter-note">
Önce mekanizmayı notebook'ta kur — tek sorgu, altı aday, skorlar ekranda — ve salon bir reranker'ın
kütüphane değil, içinde prompt olan bir for döngüsü olduğunu görsün. Skor sütunundaki eşitliklere
işaret et: 3B bir modelden tam sayı isteyince elinde birkaç farklı değer kalıyor ve eşitlikte
retriever'ın sırası korunuyor. Sonra <code>exercises/m9_rerank_trade.py</code>'ı başlat ve o çalışırken
tahmini al: dört kurulumun hepsine mi yarar, hiçbirine mi, bazılarına mı? Bittiğinde
<strong>ağzında gevelenmemesi gereken cümle</strong>: <strong>reranker, retriever'ın reranker'dan
kötüyse kazandırır; retriever'ın daha iyiyse kaybettirir.</strong> Bir kez, yavaşça söyle ve
tahtadaki el sayılarına geri işaret et. Biri "gerçek reranker chat modeli değil, cross-encoder olur"
diye itiraz edecek. Haklı — dürüstçe cevap ver: onu ölçmedik ve denenecek ilk şey o. Onun için sayı
uydurma. Yavaş laptopta <code>--quick</code> çağrıların dörtte biriyle bitiyor ve en üstte REDUCED
yazıyor; argümanın ihtiyaç duyduğu satırlar zaten iki güçlü satır. 20 dakika, çalıştırma dahil.
</div>

### Nasıl sorduğun — tek çağrıda mı, altı çağrıda mı

Reranker'ın daha ucuz bir şekli var: tek çağrı, altı pasajın hepsi birden, modelden sırala. Notebook
iki çağrıyı da aynı soruda, aynı altı aday üzerinde, aynı modelle yapıyor ve dönenleri basıyor.
Pointwise altı tam sayı döndürdü — 4, 2, 2, 5, 5, 3 — altı çağrıda ve 3.3 sn'de. Listwise `1,4,2,5`
dedi: altı pasaj için dört indis, içinde kullanılabilecek bir sıralama yok. Altı pasajı sıralamak, on
beş ikili yargıyı tek bir cevaba sıkıştırmak demek; tek bir pasajı sabit bir ölçeğe göre puanlamak
ise tek bir yargı.

Bu gösterimin ne *olmadığı* konusunda dürüst ol, çünkü iki yol eşit korunmuyor. Bizim pointwise
yolumuz parse ediyor — okunamayan cevap 0 oluyor, eşitlikte retriever'ın sırası korunuyor, yani en
kötü ihtimalle hiçbir şey yapmamaya iniyor — listwise hücresinin ise ne parser'ı ne fallback'i var ve
modele her pasajın 600 karakterini gösteriyor, pointwise 900 gösterirken. Ekranda düşen şey format
uyumu, yargı değil; üstelik listwise çıktısı hiç skorlanmadı. Yani bu, ucuz şeklin kullanılamaz bir
çıktı ürettiğinin gösterimi, daha kötü sıraladığının değil. Bu yönü ilk gösteren eski karşılaştırma
emekliye ayrılmış on dokümanlık bir probe corpus'unda koştu ve `eval/RESULTS.md` içinde `UNVERIFIED`
olarak işaretli; sayıları burada aktarılmıyor.

Hangi şekli seçersen seç, hiçbiri "rerank et" demiyor. Aynı pointwise reranker bu yirmi soruda güçlü
kurulumu 0.750'den 0.600'e indirdi.

## Contextual retrieval — onu zaten yaptın

Contextual retrieval, her chunk'ın başına onu tek başına anlaşılır kılacak kadar çevre bilgisi
koymak demek; genelde bir modele "bu chunk dokümanın neresinde" diye bir satır yazdırarak.
Structure-aware chunking her chunk'ın başına zaten kendi başlık yolunu koyuyor: aynı mekanizma, ama
modelle üretilmiş değil dokümandan alınmış ve inference maliyeti sıfır. K satırı ile sütun başlığının
aynı chunk'ta hayatta kalmasının sebebi bu. Modül 7'deki 0.750 aslında contextual retrieval'ın
kazancı, cebe girmiş durumda. Üstüne bir de modelle context üretmek denenmeye değer — ama başlıkları
olan dokümanlarda, Markdown'ın sana bedavaya vermediği bir şey satın aldığından emin ol.

## Ne çalıştırıyorsun

**Mekanizma — notebook'ta.** `notebooks/06_hybrid_rerank_contextual.py` dosyasını, açık klasör repo
kökü olacak şekilde VS Code'da aç; imleci bir `# %%` bloğunun içine koy ve `Shift+Enter`'a bas,
çıktı Interactive penceresinde belirir. Bu eğitimde Jupyter sunucusu yok.

**Terminal (repo kökü):**

```bash
ollama serve                       # sadece zaten çalışmıyorsa
python scripts/verify_setup.py     # üç modeli birden kontrol eder — bu taramaya nomic-embed-text de gerekiyor
```

- **ne görmen gerekiyor** — `154 chunks indexed both ways`, ardından üçlü karşılaştırma: dense
  0.750, BM25 0.300, RRF 0.450. Sonra tek bir sorgunun altı adayı 0-10 arası puanlanmış, aldığı
  saniyeyle birlikte.
- **kabaca ne kadar sürüyor** — 154 chunk üzerinde bir embedding geçişi artı altı model çağrısı.
  Birkaç dakika, çoğu embedding. BM25 ve fusion hiç model çağrısı istemiyor.

**Ölçüm — tek komut. Terminal (repo kökü):**

```bash
python exercises/m9_rerank_trade.py            # tam tarama
python exercises/m9_rerank_trade.py --quick    # yavaş laptop: sadece güçlü kurulumlar, depth 4
```

- **ne görmen gerekiyor** — her retrieval kurulumu için bir satır, her satırda kendi çağrı sayısı ve
  saniyesi: iki `nomic` satırında HELPED, iki `bge-m3` satırında hurt. Sonra soru bazındaki özet:
  15 soruda doğru doküman birinci sıradaydı, reranking bunların 4'ünü aşağı taşıdı.
- **maliyeti, dersin parçası olduğu için** — 4 kurulum × 20 soru × 8 aday = **640 model çağrısı**,
  kurulum başına 160 — bu sayfanın aktardığı koşuda sırasıyla 191 sn, 127 sn, 51 sn ve 71 sn; toplam
  yazdırılmıyor. `--quick` ise 2 kurulum × 20 soru × 4 aday = 160 çağrı, ve en üste REDUCED yazıyor,
  böylece onun sayıları bu sayfadakilerle karıştırılmıyor.

Her şey ortak koddan geliyor. Notebook'un preflight'ı çalışma dizinini `notebooks/` yapıyor; yolları
bu yüzden bir seviye yukarı çıkıyor.

**VS Code — notebook'un kendi hücreleri, sıkıştırılmış:**

```python
import sys; from pathlib import Path
sys.path[:0] = [".", "notebooks"]
import _preflight; _preflight.ready(chat=True, embed=True, replayable=True)
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")   # 154 chunk
chunks = dict(zip(chunk_ids, chunk_texts))
dense, bm25 = R.DenseRetriever(chunk_ids, chunk_texts), R.BM25(chunk_ids, chunk_texts)

q = next(x for x in questions if x["id"] == "q05")
d, b = dense.rank(q["query"]), bm25.rank(q["query"])
fused = R.rrf([d, b])                                 # chunk düzeyinde fuse et, dokümanlara sonra indir
print(R.pointwise_rerank(q["query"], d[:6], chunks))  # 6 aday -> 6 model çağrısı
```

## Sayılar ne dedi

<div class="measured">

| aynı 154 structure-aware chunk üzerinde retriever | hit@1 | recall@5 | MRR |
|---|---|---|---|
| dense, `bge-m3` | **0.750** | **0.833** | **0.817** |
| BM25 | 0.300 | 0.633 | 0.467 |
| ikisinin RRF'i | 0.450 | 0.683 | 0.579 |
| 6 `tr_en` sorusunda BM25 | 0.000 | — | — |
| 6 `tr_en` sorusunda RRF | 0.000 | — | — |

| tam dokümanlar, hiç chunking yokken | hit@1 | recall@5 | MRR | `exact_token` | `tr_tr` | `tr_en` |
|---|---|---|---|---|---|---|
| dense, `bge-m3` | **0.600** | 0.717 | **0.677** | **1.000** | 1.000 | **0.333** |
| BM25 | 0.400 | 0.583 | 0.515 | 0.750 | 1.000 | 0.000 |
| ikisinin RRF'i | 0.500 | **0.733** | 0.611 | **1.000** | 1.000 | 0.000 |

| en güçlü kurulum, soru bazında | adet |
|---|---|
| doğru doküman zaten birinci sırada | 15 |
| bunlardan reranking'in aşağı taşıdığı | 4 — q06, q10, q16, q18 |
| doğru doküman birinci sıranın altında | 5 |
| bunlardan reranking'in yukarı çektiği | 1 — q14 |

| depth 8'de bir rerank geçişinin maliyeti | |
|---|---|
| soru başına model çağrısı | 8 |
| kurulum başına, 20 soru | 160 |
| dört kurulumluk tam tarama | 640 |
| bu koşuda kurulum başına süre | 191 sn, 127 sn, 51 sn, 71 sn |
| tam koşu ve `--quick` | yazdırılmadı — yalnızca yukarıdaki dört kurulum-başı süre ölçüldü |

</div>

Corpus: 28 doküman, 78 310 karakter. Gold set: 20 soru, 6'sı Türkçe sorgu / İngilizce doküman.
Generation ve reranking `qwen2.5:3b`, embedding aksi belirtilmedikçe `bge-m3`, hepsi Ollama üzerinden
lokal. **Yirmi soru iki tasarım arasında karar verdirir, genel bir iddiayı taşımaz**; 0.05'in altındaki
bir fark bu büyüklükteki bir setin gürültüsünün içinde. Bu sayfa "bu teknikler kötüdür" demiyor. Bu
corpus'ta, bu embedder ve bu reranker ile kaybettiklerini söylüyor ve her birinin hangi önkoşula
ihtiyaç duyduğunu adıyla koyuyor.

## Daha derine

Hiç değiştirmediğimiz bir koşul var: her seferinde 154 maddelik sıralamaların tamamını fuse ettik.
Zayıf bir sıralamanın ne kadar derine kadar oy kullanacağı bir ayar ve biz onu listenin en dibinde
bıraktık.

Bizim reranker'ımız pasaj puanlayan bir chat modeli; reranker denilebilecek en zayıf şey. Production
cevabı cross-encoder: sorguyu ve pasajı **birlikte**, tek forward pass'te okuyan, prompt'la göreve
ikna edilmiş değil ilgililik etiketleriyle eğitilmiş bir model. `bge-reranker-v2-m3` bunun çok dilli
olanı ve bizim embedder ile aynı aileden. Onu ölçmedik, o yüzden "0.750'yi geçerdi" cümlesini sonuç
değil, deneyi belli bir hipotez olarak kabul et — ders iki cevapta da ayakta kalıyor, çünkü daha
güçlü bir hakemin de bir tavanı var ve o tavanın senin retriever'ının neresine düştüğünü ölçmen
gerekiyor.

Düzleme sonucu reranking'in ötesine genelleniyor: üstteki bir sıralamayı ezen her aşama kendi
doğruluğunu çıktıya dayatır. "Sen bir reranker ekle" tavsiyesinin "sen bir cache ekle" kadar kötü
olmasının sebebi bu — önkoşul, sisteminle ilgili ancak ölçerek öğrenebileceğin bir gerçek. Sıra da
önemli. Buradaki her şey, modül 7'nin chunking kaldıracı (0.600'den 0.750'e) ile modül 6'nın embedder
kaldıracı (`nomic` 0.350'ye karşı `bge-m3` 0.750) çekildikten sonra geldi; yani tam da bu tekniklerin
maliyet yazdığı noktada. Tablonun iki satırı gerçek bir iyileşmeye bakarak reranker'ı production'a
alırdı — `nomic` cidden 0.350'den 0.450'ye ve 0.350'den 0.400'e çıkıyor — ve tasarım yine yanlış
olurdu, çünkü doğru hamle embedder'ı düzeltmekti.

On milyon dokümanda BM25 bambaşka bir sebeple geri geliyor: ucuz bir birinci aşama birkaç yüz aday
döndürür, pahalı bir ikinci aşama onları sıralar ve inverted index üzerinde BM25 güçlü bir birinci
aşamadır — milisaniyenin altında, tanımlayıcılarda birebir, güncellemesi kolay. Bu, bizim ölçtüğümüz
hybrid değil: orada nihai cevaba oy vermiyor, aday üretiyor ve metriği hit@1 değil recall@k.

<div class="presenter-note">
Geriden geliyorsan üç ölçümü de koru, onun yerine metni kes — listwise kenar notu tek cümleye iner,
"Daha derine" zaten okuma malzemesi. Kapatmadan önce tahtadaki el sayılarına işaret et ve üç kararı
oku: hybrid kaybetti, rerank düzledi, contextual zaten modül 7'de yapılmıştı. Sonra hâlâ çalışmayanı
söyle: açılış slaytındaki beş soruya geri dön, q19 multi-hop'un kaçırdığı soru ve bu modüldeki her
şeyden sonra hâlâ orada — ikizi q20 chunking'le zaten çözülmüştü. Modül 10 tam orada ve tek bir
hücreyle gösterilebilecek bir başarısızlıkla açılıyor. 6 dakika.
</div>

## Çıkış cümlesi

> Ölçtüğümüz kadarıyla hiçbiri düz retrieval'ı geçemedi. Ama multi-hop hâlâ düşüyor.
