---
title: "5. Keyword Search'ten Naive RAG'e"
description: "Doğru parçayı nasıl bulacağım? Keyword search, embedding ve retrieval'ın yanlış dokümanı getirmesinin dört yolu."
---

## Gate sorusu

> **Doğru parçayı nasıl bulacağım?**

Önceki modül çalışan bir cevapla ve bir faturayla bitti. Yirmi sekiz dokümanın tamamı context
window'a sığıyor, model EUR 90'ı doğru okuyor ve sen her soruda bütün kural kitabının parasını
ödüyorsun — 78 310 karakterin içinden tek bir paragrafa ihtiyaç duyan sorularda bile. Kural kitabı
üç ayda bir yeniden yayımlanıyor. Dört çeyrek ve altı istasyon olduğunda, maliyet canını yakmadan
çok önce prompt sığmayı bırakıyor.

O yüzden seçeceğiz. Yirmi sekiz doküman yerine üç doküman göndereceğiz. Günün geri kalanı tek bir
kelimeyle ilgili — seçmek — ve bu işin kaç farklı şekilde bozulduğuyla.

<div class="presenter-note">
Bu modül 10:43–11:26 arası, 43 dakika ve asla kesilmeyecekler listesinde. Kaba bütçe:
3 dk açılış sorusu · 7 dk BM25 mekanizması · 6 dk Türkçe 14. sıra ve İngilizce sütunu ·
5 dk dense retriever'ın kısmi kurtarışı · 6 dk pipeline ve benchmark tablosu · 12 dk dört hata ·
4 dk tip bazlı satırların okunması ve çıkış.
<br/><br/>
Notebook'u açmadan önce salona sor: "Elinde 28 doküman ve bir soru var. Gönderilecek üç dokümanı
nasıl seçeceğini tek satırda yaz." İki cevabı yüksek sesle al. Biri mutlaka "kelimeleri arayarak"
diyecek. O kişi bu modülün ilk yarısını yazmış oldu, bunu da söyle. 3 dakika, laptoplar hâlâ kapalı.
</div>

## Herkesin ilk uzandığı retriever

Keyword search. Sorgudaki kelimelerin her dokümanda kaç kez geçtiğini say, sayıya göre sırala.
Makul bir ilk deneme ve bugüne kadar kullandığın kurumsal arama kutularının çoğunun arkasında bu var.

BM25 tam olarak bu fikrin üç düzeltmeyle güçlendirilmiş hali. `eval/retrieval.py` bunu bir kütüphaneden
import etmek yerine bilerek elle yazıyor, çünkü skorun parçalarına ayrıldığını ekranda görmek işin
kendisi:

```python
idf  = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
f    = self.term_freq[i].get(term, 0)
norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_length or 1)
score += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
```

Üç düzeltme, koddaki sırasıyla. **Term frequency doyuma ulaşır.** "Penalty" kelimesini dokuz kez geçiren
bir doküman, bir kez geçirenden dokuz kat daha alakalı değildir; bu yüzden `f` düzleşen bir kesrin içinde
durur, `k1 = 1.5` de ne kadar hızlı düzleşeceğini belirler. **Nadir kelimeler daha çok değer.** `idf`,
terimi içeren doküman sayısı `df` küçüldükçe büyür. `penalty` yirmi sekiz dokümanın on beşinde geçtiği
için neredeyse hiçbir şey kazandırmaz; `KSHEU26` tek bir dokümanda geçtiği için her şeyi kazandırır.
**Uzun dokümanlar cezalandırılır.** `norm`, doküman uzunluğunu corpus ortalamasına böler; böylece 9 KB'lık
interline anlaşması sırf daha çok kelime içeriyor diye her sorguyu kazanmaz. `b = 0.75` bu cezanın
şiddetini ayarlar.

Bunların hiçbiri çalışmadan önce `tokenize()` kelimenin ne olduğuna karar veriyor: alfanümerik olmayan
her karakteri boşluğa çeviriyor, kalanı da küçük harfe indiriyor. Yani `XX 1487` iki ayrı token'a
dönüşüyor, `xx` ve `1487`; `SCB-2026-0914` ise dörde. Bu kabalık bilinçli — uçuş kodu literal string
üzerinden eşleşiyor, başka hiçbir şey üzerinden değil — ama iki parça çok farklı miktarda iş yapıyor.
`xx` Kraken Air'in kendi kodu ve 28 dokümanın 19'unda geçiyor, dolayısıyla `idf`'i tabanda; `1487` ise
4'ünde geçiyor ve sorguyu asıl taşıyan token o.

**VS Code — `notebooks/04_naive_rag.py`, Part 1'in ilk bloğu:**

```python
bm25 = R.BM25(doc_ids, texts)
for query in ["XX 1487", "SCB-2026-0914"]:
    print(f"{query!r:>18} -> {bm25.rank(query)[:3]}")
```

İkisi de doğru geliyor. `'XX 1487'` birinci sıraya `bulletin_scb_2026_0914`'ü koyuyor, `'SCB-2026-0914'`
de aynısını. Keyword search zaten kesin tanımlayıcılar için var.

Şimdi bir çağrı merkezi temsilcisinin gerçekten yazdığı sorgu. Bu, notebook'un kendi string'i,
`04_naive_rag.py:45` satırında; düzgün Türkçe değil ASCII, çünkü insanlar iş terminaline böyle yazıyor:

> Musteri bileti iptal ederse ne oder?

Cevap `fare_classic_shorthaul` içinde, İngilizce bir ücret sayfasında. BM25 onu **28 doküman içinde
14. sırada** getiriyor. Onun yerine birinci sırada `macro_tr_noshow`, sonra `macro_tr_rebook`, sonra
`macro_tr_baggage` geliyor — üç farklı soruyu cevaplayan üç Türkçe doküman, sırf içlerinde `iptal`
kelimesi geçtiği için tepede.

Sebebi ne ince ne de kıl payı. Sorgunun terimleri ya corpus'un tamamında yok, o zaman döngü onları
atlıyor; ya da *bu* dokümanda yok, o zaman `f` sıfır oluyor ve terim hiçbir şey katmıyor. Her iki
durumda da niyeti anlamanın kısmi puanı yok. Ücret sayfasında `Cancellation penalty` yazıyor,
sorguda `iptal`. BM25 bunların aynı kelime olduğunu bilmiyor — sadece farklı string olduklarını biliyor.

Sadece bir Türkçe problemi de değil. Gold set'teki dört İngilizce soru / İngilizce doküman sorusunda
BM25 aynı tam dokümanlar üzerinde hit@1 **0.250** alıyor: dördün üçünde birinci sıraya cevap dışında
bir şey koyuyor. Keyword search string eşler. Kullanıcın soru sorar.

<div class="presenter-note">
Ağzında gevelemeyeceğin cümle bu: <strong>BM25 "iptal" ile "cancellation"ın aynı kelime olduğunu bilmez —
sadece farklı string olduklarını bilir.</strong> Bir kez, yavaşça söyle ve 14. sırayı ekranda bırak.
"Daha kötü çalışıyor" diye yumuşatma.
<br/><br/>
Biri gerçek skoru görmek isterse: <code>BM25.rank()</code> sadece doküman id'si döndürüyor, notebook
burada hiçbir sayı basmıyor. Skoru veren çağrı <code>bm25.scores(paraphrase)</code> ve dosyada yok;
canlı yazmak yerine ödev olarak ver.
</div>

## String yerine anlam

Modül 2'de 784 pikseli 128 sayılık bir vektöre çeviren bir ağ kurmuştuk. Embedding modeli aynı şeyi
metne yapıyor: `bge-m3`, Türkçe ya da İngilizce herhangi bir string'i 1 024 sayılık bir vektöre
çeviriyor ve aynı anlama gelenler birbirine yakın düşüyor. 28 dokümanı bir kez embed et, soruyu embed
et, cosine similarity'ye göre sırala. `DenseRetriever` bu, on iki satır.

Aynı Türkçe soruyu tekrar sor: ücret sayfası 14. sıradan **5. sıraya** çıkıyor. Dürüst sonuç bu ve
hikâyenin istediği temiz kurtarış değil. İlk üç sırada `macro_tr_noshow`, `macro_en_refund`,
`macro_tr_rebook` var — BM25'in birinci sıraya koyduğu yanlış Türkçe makro hâlâ birinci. Değişen şey,
doğru dokümanın artık ikinci yarıya gömülmek yerine ilk beşin içinde olması.

Bu kazancın büyüklüğü üzerinde biraz dur, çünkü birazdan kuracağın pipeline **k=3** gönderiyor. 5. sıra,
`recall@5`'in saydığı pencerenin içinde ve modelin gerçekten okuduğu pencerenin dışında. Embedding
açığın çoğunu kapattı ve kapatmadı. Bu, tip bazlı tablonun `tr_en` için 0.333 diye raporladığı zayıflığın
ta kendisi; onu kımıldatan şey daha iyi bir soru değil — modül 6'nın embedder seçimi ve modül 7'nin neyi
index'leyeceğine dair kararı.

Yine de generator'ı üstüne tak, pipeline hazır: sorguyu embed et, top 3'ü al, "sadece bu context'i kullan"
talimatının altına yapıştır, üret. On iki satır retriever, on satır pipeline. Çalışıyor — ve asıl mesele
skor.

<div class="presenter-note">
Benchmark bloğunu çalıştırmadan önce taahhüt al: "hit@1, getirdiğimiz ilk dokümanın doğru doküman
olması demek. 20 sorudan kaçta kaçını tutturuyoruz? Bağırın." Salon 0.9 der. Sonra <em>kendi</em>
çalıştırmanın bastığı sayıyı göster — kayıt 0.600 ve embedding çıktısı Ollama sürümleri arasında
bit düzeyinde sabit değil, yani buradaki 0.05'lik oynama bu sayfanın zaten gürültü dediği şeyin
kendisi. O aradaki fark modülün kendisi. Ollama çöktüyse aynı tablo <code>eval/RESULTS.md</code>
içinde — oradan oku ve devam et, canlı debug'a girme.
</div>

## Gözünün önünde neyi yanlış yapıyor

**0.600.** Beş soruda ikisinde modele verilen ilk doküman yanlış doküman ve model o soruyu
cevaplayamayacak bir kaynaktan cevaplıyor, üstelik bunu bilmeden. Notebook'un açıkça gösterdiği dört
hata var — hücrelerinin bastığı sırayla, ki notebook'u aşağıdaki *Ne çalıştırıyorsun* bölümünde açıyorsun.

**Hata 1 — doğru doküman, yanlış cevap.** CLASSIC K iptal cezasını sor. Dense retrieval
`fare_classic_shorthaul`'u birinci sıraya koyuyor, yani retrieval doğru. Kayıtlı çalıştırmadaki cevap
ise `For CLASSIC fare family, the cancellation penalty for booking class Q is EUR 180.` — gerçek bir
tablonun gerçek bir hücresi, ama kimsenin sormadığı bir booking class için. Pipeline prompt'a
`docs[h][:1500]` yapıştırıyor: 3 923 karakterlik bir dokümanın ilk 1 500 karakteri. Sütunları
isimlendiren başlık 1 131. karakterde ve kesiğin içinde kalıyor;
`| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` satırı ise 1 743.
karakterde başlıyor ve kesiğin dışında. Modele başlık, O ve T satırları ve Q satırının ilk üç karakteri
gidiyor — Q satırı 1 497'de başlıyor, kesik 1 500'de düşüyor — o da elindekinden cevap veriyor.
Retrieval hit aldı. Tereddüt yok, geri soru yok ve çıktıda bunu bir tahmin olarak işaretleyen hiçbir şey yok.

**Hata 2 — gerçeğin iki sürümü.** Bir misconnect'te kaç euroluk yemek fişi verileceğini sor.
`sop_misconnect_v4` EUR 15 diyor, `sop_misconnect_v3` EUR 10 diyor ve ikisi birlikte geliyor.
Aralarındaki tek fark, retriever'ın okumadığı `Version: 3 | Superseded` satırı. Retrieval'ın "güncel"
diye bir kavramı yok.

**Hata 3 — tanımlayıcılar birbirine karışıyor.** Aynı 28 tam doküman üzerinde dört kısa sorgu, dense'e
karşı BM25. `XX 1487` ikisinde de 1. sırada. `SCB-2026-0914` BM25'te 1. sırada, dense'te 6. sırada —
dense başka bir bülteni, `bulletin_scb_2026_0921`'i öne koyuyor. `YY 88` BM25'te 5. sırada, dense'te
13. sırada. `booking class K` BM25'te 1. sırada, dense'te 3. sırada; dense birinci sıraya
`policy_corporate_travel`'ı koyuyor, çünkü o doküman da booking class'lardan uzun uzun bahsediyor.
1 024 sayılık tek bir vektör koca bir pasajın anlamını taşımak zorunda ve bir bülten numarası o anlamın
minicik bir kesri. Yirmi dakika önce kenara attığımız keyword search yerini tam burada geri kazanıyor.

**Hata 4 — bambaşka bir yere düşen Türkçe soru.** `q05`'i al: Türkçe konuşan bir temsilci, dört saat
bekleyecek yolcuya ne verilmesi gerektiğini soruyor. Cevap İngilizce bir prosedürde, `sop_misconnect_v4`'te
ve **28 doküman içinde 18. sırada** geliyor. Onun yerine `policy_expense_reimbursement`, short-haul ücret
sayfası ve Türkçe bagaj makrosu geliyor — hepsi makul biçimde "seyahat aksadıktan sonra para", hiçbiri
cevap değil. Keyword search burada yapısı gereği çaresiz, ama embedding bu boşluğu kapatacaktı ve ancak
kısmen kapatıyor: yirmi sorunun altısı bu şekilde ve dense retrieval bunlarda **0.333** alıyor, yani
altıda dördü başarısız. Bunun embedder'ın suçu mu bizim suçumuz mu olduğu modül 6'nın sorusu.

Sonra ortalamaya değil, tip bazlı satırlara bak. 28 tam doküman üzerinde `exact_token` soruları **1.000**
alıyor — uçuş kodları, bülten id'leri ve K sınıfı ücret araması bu ölçekte hepsi 1. sırada bulunuyor.
Aynı corpus'u 294 adet sabit 280 karakterlik chunk'a böl: ortalama 0.600'den 0.700'e çıkarken bu kategori
**0.750**'ye düşüyor, yani dörtten biri birincilikten düşüyor. Altı schedule bülteni aynı kalıp metni
paylaşıyor ve birbirlerinden bir avuç token'la ayrılıyor; dilimledikten sonra dilimler neredeyse aynı
metne dönüşüyor ve `1487` taşıyan dilim öne çıkmayı bırakıyor. BM25 de sığınak değil: modül 7'nin
bittiği 154 structure-aware chunk üzerinde, sorgu bir tanımlayıcıyken uzanacağın retriever aynı
kategoride **0.500** alıyor; tam dokümanlarda 0.750 alıyordu. Ortalama yükseldi, bir kategori düştü.
İki fark da tek soru genişliğinde; ders rakam değil, yön. Modül 7 tam olarak bunun üstüne kurulu.

## Ne çalıştırıyorsun

**Terminal (repo kökü)** — `corpus/`, `notebooks/`, `eval/` ve `exercises/` klasörlerini içeren dizin:

```bash
ollama list                        # bge-m3 ve qwen2.5:3b ikisi de listede olmalı, zaten kurulu
python scripts/verify_setup.py     # devam etmeden önce READY yazmalı
```

Gün içinde hiçbir şey indirilmiyor. `ollama list` çıktısında bir model eksikse pull başlatmak yerine
bunu hemen söyle.

**VS Code, açık klasör repo kökü:** `notebooks/04_naive_rag.py` dosyasını aç, imleci bir `# %%` bloğunun
içine koy ve Shift+Enter'a bas. Çıktı Interactive window'da beliriyor. Bu eğitimde hiçbir yerde Jupyter
sunucusu ya da tarayıcı notebook'u yok.

**Ne görmen gerekiyor.** `Musteri bileti iptal ederse ne oder?` sorgusunun ücret sayfasını BM25'te
14. sıraya, dense retriever'da 5. sıraya koyması; sonra karşılaştırma tablosu: 28 tam doküman üzerinde
BM25 hit@1 **0.400**, bge-m3 **0.600**.

**Kabaca ne kadar sürüyor.** BM25 blokları anında dönüyor, 28 dokümanı embed etmek bir kereliğine
yaklaşık 7 sn sürüyor ve üzerinde konuşmaya değen tek şey iki generation bloğu. Aynı ölçümün tek
komutluk hali — repo kökündeki terminalden `python eval/run_benchmark.py --skip-rerank` — ısınmış bir
makinede 31.5 sn ve 42 embed çağrısı.

Notebook kodu yeniden yazmıyor, ortak kodu kullanıyor. `eval`'i paket olarak import etmiyor;
`_preflight.ready()` çalışma dizinini `notebooks/` yapıyor ve `eval/` dizinini `sys.path`'e ekliyor,
bu yüzden import'lar düz ve bütün göreli yollar `notebooks/` içinden yazılmış:

**VS Code — `notebooks/04_naive_rag.py`, import'lar ve onları kullanan bloklar:**

```python
import _preflight; _preflight.ready(chat=True, embed=True)
import retrieval as R, metrics

bm25  = R.BM25(doc_ids, texts)
dense = R.DenseRetriever(doc_ids, texts)          # bge-m3, doküman başına 1024 sayı

paraphrase = "Musteri bileti iptal ederse ne oder?"
bm25.rank(paraphrase)[:3]
dense.rank(paraphrase)[:3]

questions = metrics.load_gold("../eval/gold_questions.jsonl")
metrics.evaluate({q["id"]: dense.rank(q["query"]) for q in questions}, questions)
```

Bu sayfadaki bütün tam doküman sayıları `python eval/run_benchmark.py --skip-rerank` ile yeniden
üretiliyor; diğer flag'ler ve hangisinin neyi bastığı `eval/README.md` içinde.

<div class="presenter-note">
Blok süreleri: BM25 blokları anında dönüyor, 28 dokümanı embed etmek bir kereliğine yaklaşık 7 sn
sürüyor, üzerinde konuşulmaya değer tek blok 20 soruluk benchmark. Onu başlat, çalışırken salondan
hit@1 tahminini iste. Geriden geliyorsan BM25'in 14. sırasından doğrudan dense retriever'ın
5. sırasına geç ve `en_en` kenar notunu atla; dört hata demosunu da koru, modül 6 ve modül 7 tam
onların üstüne açılıyor.
</div>

## Sayılar ne dedi

<div class="measured">

| retriever | hit@1 | recall@5 | MRR | index boyutu |
|---|---|---|---|---|
| BM25, tam dokümanlar | 0.400 | 0.583 | 0.515 | 28 doküman |
| dense (`bge-m3`), tam dokümanlar | 0.600 | 0.717 | 0.677 | 28 doküman |
| dense, sabit 280 karakter chunk (modül 7) | 0.700 | 0.917 | 0.817 | 294 chunk |
| dense, structure-aware chunk (modül 7) | 0.750 | 0.833 | 0.817 | 154 chunk |
| aynı structure-aware chunk'lar üzerinde BM25 | 0.300 | 0.633 | 0.467 | 154 chunk |

soru tipine göre hit@1:

| soru tipi | BM25, tam dok. | dense, tam dok. | dense, sabit 280 |
|---|---|---|---|
| `tr_tr` (4) | 1.000 | 1.000 | 1.000 |
| `tr_en` (6) | 0.000 | 0.333 | 0.500 |
| `en_en` (4) | 0.250 | 0.500 | 0.750 |
| `exact_token` (4) | 0.750 | 1.000 | 0.750 |
| `multi_hop` (2) | 0.000 | 0.000 | 0.500 |

</div>

İki tam doküman satırı `notebooks/04_naive_rag.py` çıktısı; bu sayfa hiçbir şeyi chunk'lamıyor,
chunk'lı satırlar modül 7 ve modül 9'un çalıştırmaları. 294 sabit 280 karakterlik chunk üzerinde
ölçülmüş bir BM25 çalıştırması yok — repodaki tek BM25-over-chunk ölçümü `run_benchmark.py --fusion`
ve o da 154 structure-aware chunk üzerinde çalışıyor, yukarıdaki satır o.

`run_benchmark.py --skip-rerank` aynı iki tam doküman retriever'ını bir kez daha skorluyor ve BM25 için
recall@5 0.583 / MRR 0.515, `bge-m3` için 0.717 / 0.677 basıyor — notebook'la üç ondalığa kadar
aynı, bu koşuda. `bge-m3` Ollama üzerinden bit düzeyinde sabit değil, yani bu tam eşleşmeyi her
koşuda bekleme; 0.05'lik bir oynama, bu sayfanın zaten gürültü saydığı şeyin ta kendisi.

Dört `exact_token` sorusunun hepsi uçuş kodu değil. Üçü uçuş numarası veya bülten id'siyle yapılan
tarife araması; dördüncüsü, `q15`, booking class ile yapılan bir ücret tablosu araması — hata 1'deki
CLASSIC K iptal cezası. Etiketi "uçuş kodları" diye değil, "cevabı tek bir tabloda tek bir kesin string
olan sorular" diye oku.

Corpus: 28 doküman, 78 310 karakter. Gold set: 20 soru. Embedding `bge-m3`, generation `qwen2.5:3b`,
hepsi Ollama üzerinden lokal. Yirmi soru iki tasarım arasında karar vermeye yeter, yayımlamaya hiç
yetmez; bir soru 0.05 ediyor, dolayısıyla burada 0.05'in altındaki fark gürültüdür.

<div class="presenter-note">
Bu modülde hiçbir şeyi düzeltme. Hata 1 ekrana düşer düşmez biri "overlap ekle" ya da "düzgün bir
splitter kullan" diye bağıracak — ikisini de tahtaya yaz, modül 7'de ölçüldüklerini ve birinin yanlış
olduğunu söyle. Modül 7 tam olarak o tahtaya dönerek açılıyor.
</div>

## Daha derine

BM25'in `idf` terimi asıl anlaşılması gereken kısım, çünkü keyword search'ün hâlâ ölmemiş olmasının
sebebi o. `idf` bir sürpriz ölçüsü: bir dokümanda `KSHEU26` görmek o doküman hakkında güçlü bir kanıt,
`the` görmek hiçbir şey. Dense embedding'in buna karşılık gelen bir düğmesi yok. Koca bir pasajı
1 024 sayılık tek bir vektöre sıkıştırıyor ve nadir bir tanımlayıcı, içinde bulunduğu pasajın anlamının
minicik bir kesri olduğu için ortalamanın içinde eriyip gidiyor. Hata 3'ün mekanizması bu; chunk'ladıktan
sonra exact token'larda 1.000'den 0.750'ye kayışın mekanizması da bu. Yapısal bir şey, `bge-m3`'ün
bug'ı değil.

`b = 0.75` uzunluk normalizasyonu da chunking'in BM25'i neden mahvettiğinin diğer yarısı. Her şeyi chunk
boyutuna kes, her doküman ortalama uzunluğa gelir, `norm` herkes için 1 olur ve kısa bir bülteni uzun
bir anlaşmadan ayıran terim iş yapmayı bırakır. BM25 28 tam doküman üzerinde hit@1 0.400 alıyor;
154 structure-aware chunk üzerinde 0.300 — ve sahiplenmesi gereken exact-token sorularında 0.750 düşüp
0.500 oluyor. Onu bozan retriever değildi. Neyi index'leyeceğine dair verdiğin karardı.

BM25'in kendi profilini kategori kategori oku, asimetri her ortalamadan daha keskin: `tr_tr`'de 1.000,
`tr_en`'de 0.000 — hiçbirinde embedding modeli, GPU ya da ağ yok. Yapmadığı şey ise yukarıdaki tablonun
herhangi bir sütununda dense retrieval'ı yenmek; exact token'larda bile değil, orada dense tavanda.
Keyword search beş dakikasını maliyetle, `SCB-2026-0914` gibi tek tanımlayıcılı sorgularla ve hiçbir
zaman bir modele ihtiyaç duymamasıyla hak ediyor. Kazanarak değil.

Top-k bir kez verip sonra verdiğini unuttuğun bir karar. Bu sayfadaki her metrik k'ye bağlı bir ifade:
tam dokümanlarda recall@5 0.717 iken hit@1 0.600 — yani doğru doküman sık sık ilk beşte ve birinci değil;
5. sıradaki Türkçe soru tam olarak bu durum. Generator yanlış dördü güvenilir şekilde görmezden gelseydi
k=5 yapıp işi bitirirdin. Görmezden gelmiyor: context'teki yanlış doküman, alıntılanmayı bekleyen bir
yalandır ve v3/v4 hatası tam olarak budur.

On milyon dokümanda `DenseRetriever`'daki brute-force döngü ölür. Yaklaşık bir index kurarsın — genelde
cevap HNSW — ve parametreyle kendin seçtiğin bir recall kaybını, doğrusal yerine logaritmik arama
karşılığında kabul edersin. Similarity hesaplamadan önce metadata ile filtrelersin; v3/v4 problemini
gerçekten çözen şey de bu: `status = current` tek satırlık bir filtre ve hiçbir embedding iyileştirmesi
onun yerini tutmaz. Çeyrek, route band ve doküman sürümü zaten elinde olan yapısal alanlar; corpus'a düz
metin muamelesi yaparak onları çöpe atıyorsun.

Naive pipeline, kasten zayıf kurulmuş bir örnek değil. Yirmi iki satır, framework yok, vector database
yok ve ağı kapalı bir laptopta yirmi gold sorunun on ikisinde doğru dokümanı birinci sıraya koyuyor.
Bundan sonraki her şey 0.600'e karşı ölçülüyor ve popüler iyileştirmelerden ikisi bunu geçemeyecek.

## Çıkış cümlesi

> RAG çalışıyor ve retrieval çöp getiriyor — altı Türkçe soru / İngilizce doküman sorusunun dördü
> hâlâ yanlış geliyor ve izlediğimiz soru ancak 5. sıraya kadar çıkabildi. Onu düzeltmeden önce:
> bu embedding modelini ben senin yerine seçtim ve neden seçtiğimi hiç söylemedim.
