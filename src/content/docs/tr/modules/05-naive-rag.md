---
title: "5. Keyword Search'ten Naive RAG'e"
description: "Doğru parçayı nasıl bulacağım?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Doğru parçayı nasıl bulacağım?**

Önceki modül çalışan bir cevapla ve bir faturayla bitti. Yirmi sekiz dokümanın tamamı context
window'a sığıyor, model EUR 90'ı doğru okuyor ve sen her soruda bütün kural kitabının parasını
ödüyorsun — 78.310 karakterin içinden tek bir paragrafa ihtiyaç duyan sorularda bile. Kural kitabı
üç ayda bir yeniden yayımlanıyor. Dört çeyrek ve altı istasyon olduğunda, maliyet canını yakmadan
çok önce prompt sığmayı bırakıyor.

O yüzden seçeceğiz. Yirmi sekiz doküman yerine üç doküman göndereceğiz. Günün geri kalanı tek bir
kelimeyle ilgili — seçmek — ve bu işin kaç farklı şekilde bozulduğuyla.

<div class="presenter-note">
Bu modül 10:43–11:26 arası, 43 dakika, ve asla kesilmeyecekler listesinde. Kaba bütçe:
3 dk açılış sorusu · 7 dk BM25 mekanizması · 6 dk Türkçe sıfır ve İngilizce 6. sıra ·
5 dk dense retriever · 6 dk pipeline ve benchmark tablosu · 12 dk dört hata ·
4 dk tip bazlı satırların okunması ve çıkış.
<br/><br/>
Notebook'u açmadan önce salona sor: "28 dokümanın ve bir sorun var. Gönderilecek üç dokümanı nasıl
seçeceğini tek satırda yaz." İki cevabı yüksek sesle al. Biri mutlaka "kelimeleri arayarak" diyecek.
O kişi bu modülün ilk yarısını yazmış oldu, bunu da söyle. 3 dakika, laptoplar hâlâ kapalı.
</div>

## Herkesin ilk uzandığı retriever

Keyword search. Sorunun kelimeleri her dokümanda kaç kez geçiyor say, sayıya göre sırala. Makul bir
ilk deneme ve bugüne kadar kullandığın kurumsal arama kutularının çoğunun arkasında bu var.

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
terimi içeren doküman sayısı `df` küçüldükçe büyür. `penalty` ücret sayfalarının çoğunda geçtiği için
neredeyse hiçbir şey kazandırmaz; `KSHEU26` tek bir dokümanda geçtiği için her şeyi kazandırır.
**Uzun dokümanlar cezalandırılır.** `norm`, doküman uzunluğunu korpus ortalamasına böler; böylece 9 KB'lık
interline anlaşması sırf daha çok kelime içeriyor diye her sorguyu kazanmaz. `b = 0.75` bu cezanın
şiddetini ayarlar.

`"H9 1487 IST-CDG retiming"` ile çalıştır, BM25 `bulletin_scb_2026_0914`'ü 9.500 skorla birinci
sıraya koyuyor; ikinci doküman 5.705. `1487` yirmi sekiz dokümanın dördünde geçiyor, ama sadece biri
tek sayfalık bir bülten — idf'in başlattığı işi uzunluk normalizasyonu bitiriyor. Tokenizer bilerek
kaba: alfanümerik olmayan her karakterden bölüyor, dolayısıyla `h9` ve `1487` ayrı token olarak
hayatta kalıyor. Kazanmasının sebebi tam olarak bu kabalık.

Şimdi Türk bir acentenin gerçekten yazdığı soruyu yaz:

> iptal edersem ne öderim

Sıfır. Kötü bir sıralama değil — `fare_classic_shorthaul` karşısında tam olarak 0.000, çünkü o
dokümanda ne `iptal` diye bir token var ne de `öderim`. Dokümanda `Cancellation penalty` yazıyor.
Onun yerine birinci sırada `macro_tr_noshow` geliyor, 7.662 skorla: başka bir soruyu cevaplayan bir
Türkçe doküman, sırf içinde `iptal` kelimesi geçtiği için tepede. BM25 yakın bir isabete doğru
yumuşakça düşmüyor — terim yok, `df` sıfır, döngü `continue` ediyor, doğru doküman hiç puan almıyor
ve yerini kendinden emin bir alakasızlık alıyor.

Sadece bir Türkçe problemi de değil. İngilizce sor — "what do I pay to give up the ticket" — ücret
sayfası 6. sırada kalıyor. Birinci `faq_en_general`, 14.811'e karşı 3.768. Çünkü `pay` bu korpusta
tek bir dokümanda geçiyor ve idf bunu fena halde ödüllendiriyor; cevabı bulacak kelime olan
`cancellation` ise on dört dokümanda geçiyor ve sorguda hiç yok. Keyword search string eşler.
Kullanıcın soru sorar.

<div class="presenter-note">
Ağzında gevelemeyeceğin cümle bu: <strong>BM25 "iptal" ile "cancellation"ın aynı kelime olduğunu bilmez —
sadece farklı string olduklarını bilir.</strong> Bir kez, yavaşça söyle ve sıfırı ekranda bırak.
"Daha kötü çalışıyor" diye yumuşatma. Sıfır.
</div>

## String yerine anlam

Modül 2'de 784 pikseli 128 sayılık bir vektöre çeviren bir ağ kurmuştuk. Embedding modeli aynı şeyi
metne yapıyor: `bge-m3`, Türkçe ya da İngilizce herhangi bir string'i sabit uzunlukta bir vektöre
çeviriyor ve aynı anlama gelenler birbirine yakın düşüyor. 28 dokümanı bir kez embed et, soruyu embed et,
cosine similarity'ye göre sırala. `DenseRetriever` bu, on iki satır.

Şimdi "iptal edersem ne öderim" diye sor; short-haul CLASSIC sayfası birinci geliyor. Ortak token yok,
sözlük yok, çeviri adımı yok. Neden `bge-m3` de framework'ün default kurduğu model değil — bunun tamamı
modül 6. Bu modül boyunca buna güven.

Üstüne generator'ı tak: sorguyu embed et, top 3'ü al, "sadece bu context'i kullan" talimatının altına
yapıştır, üret. On iki satır retriever, on satır pipeline. Çalışıyor — ve asıl mesele skor.

<div class="presenter-note">
Benchmark bloğunu çalıştırmadan önce taahhüt al: "hit@1, getirdiğimiz ilk dokümanın doğru doküman
olması demek. 20 sorudan kaçta kaçını tutturuyoruz? Bağırın." Salon 0.9 der. Sonra 0.550'yi göster.
O aradaki fark modülün kendisi. Ollama çöktüyse aynı tablo <code>eval/RESULTS.md</code> içinde —
oradan oku ve devam et, canlı debug'a girme.
</div>

## Gözünün önünde neyi yanlış yapıyor

**0.550.** Neredeyse iki soruda birinde modele verilen ilk doküman yanlış doküman. Model o soruyu
cevaplayamayacak bir kaynaktan cevaplıyor ve bunu bilmiyor. Notebook'un açıkça gösterdiği dört
hata var.

**Hata 1 — doğru doküman, yanlış cevap.** CLASSIC K iptal cezasını sor. Dense retrieval
`fare_classic_shorthaul`'u birinci sıraya koyuyor, yani retrieval doğru. Cevap yine de yanlış.
Pipeline prompt'a `docs[h][:1500]` yapıştırıyor: 3923 karakterlik bir dokümanın ilk 1500 karakteri.
Sütunları isimlendiren başlık 1131. karakterde ve kesiğin içinde kalıyor;
`| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |` satırı ise 1743.
karakterde ve kesiğin dışında. Modele başlık, O ve T satırları ve Q'nun yarısı gidiyor; o da eldekinden
cevap veriyor. Retrieval hit aldı. Salon, kimsenin sormadığı bir booking class'ın rakamını aldı —
tereddüt yok, geri soru yok ve çıktıda bunu bir tahmin olarak işaretleyen hiçbir şey yok.

**Hata 2 — gerçeğin iki sürümü.** Beş saatlik bir misconnect'te kaç euroluk yemek fişi verileceğini sor.
`sop_misconnect_v4` EUR 15 ve 6 saat sonra otel diyor. `sop_misconnect_v3` EUR 10 ve 8 saat sonra otel
diyor. İkisi de korpusta, ikisi de misconnect hakkında, similarity skoru için ikisi de aynı görünüyor
ve sık sık aynı top-3'te birlikte geliyorlar. Aralarındaki tek fark `Version: 3 | Superseded` satırı;
retriever bunu okumuyor, modelin de bunu tartmak için bir sebebi yok. Retrieval'ın "güncel" diye bir
kavramı yok.

**Hata 3 — tanımlayıcılar birbirine karışıyor.** Aynı 28 tam doküman üzerinde dört kısa sorgu, dense'e
karşı BM25. `H9 1487` ikisinde de 1. sırada. `SCB-2026-0914` BM25'te 1. sırada, dense'te 6. sırada —
dense başka bir bülteni, `bulletin_scb_2026_0921`'i öne koyuyor. `booking class K` BM25'te 1. sırada,
dense'te 5. sırada; dense personel seyahat politikasını getiriyor, çünkü o doküman da booking class'lardan
uzun uzun bahsediyor. Birkaç yüz sayılık tek bir vektör koca bir pasajın anlamını taşımak zorunda ve bir
bülten numarası o anlamın minicik bir kesri. Yirmi dakika önce kenara attığımız keyword search yerini
tam burada geri kazanıyor.

**Hata 4 — bambaşka bir yere düşen Türkçe soru.** `q05`'i al: Türk bir acente, dört saat bekleyecek
yolcuya ne verilmesi gerektiğini soruyor. Cevap İngilizce bir prosedürde, `sop_misconnect_v4`'te ve
**28 doküman içinde 18. sırada** geliyor. Onun yerine personel masraf politikası, short-haul ücret
sayfası ve Türkçe bagaj makrosu geliyor — hepsi makul biçimde "seyahat aksadıktan sonra para", hiçbiri
cevap değil. Keyword search burada yapısı gereği çaresiz, ama embedding bu boşluğu kapatacaktı ve ancak
kısmen kapatıyor: yirmi sorunun altısı bu şekilde ve dense retrieval bunlarda **0.333** alıyor, yani
altıda dördü başarısız. Bunun embedder'ın suçu mu bizim suçumuz mu olduğu modül 6'nın sorusu.

Sonra ortalamaya değil, tip bazlı satırlara bak. 28 tam doküman üzerinde exact-token soruları **1.000**
alıyor — uçuş kodları, bülten id'leri ve K sınıfı ücret araması bu ölçekte hepsi 1. sırada bulunuyor.
Aynı korpusu 294 adet sabit 280 karakterlik chunk'a böl: ortalama 0.700'e çıkarken bu kategori
**0.750**'ye düşüyor, yani dörtten biri birincilikten düşüyor. Altı schedule bulletin aynı kalıp metni
paylaşıyor ve birbirlerinden bir avuç token'la ayrılıyor; dilimledikten sonra dilimler neredeyse aynı
metne dönüşüyor ve `1487` taşıyan dilim öne çıkmayı bırakıyor. BM25 da sığınak değil: sorgu bir
tanımlayıcıyken uzanacağın retriever, aynı kategoride tam dokümanlarda 0.750 alırken chunk'ladıktan
sonra **0.250** alıyor. Ortalama yükseldi, bir kategori düştü. Modül 7 tam olarak bunun üstüne kurulu.

## Ne çalıştırıyorsun

```bash
ollama list                        # bge-m3 ve qwen2.5:3b ikisi de listede olmalı, zaten kurulu
python scripts/verify_setup.py     # devam etmeden önce READY yazmalı
```

Gün içinde hiçbir şey indirilmiyor. `ollama list` çıktısında bir model eksikse pull başlatmak yerine
bunu hemen söyle.

Sonra `notebooks/04_naive_rag.py` dosyasını VS Code'da aç ve blokları Shift+Enter ile çalıştır.

**Ne görmen gerekiyor.** BM25'in `iptal edersem ne öderim` sorgusunda tam olarak `0.000` alması;
aynı sorunun dense retriever tarafından `fare_classic_shorthaul`'dan cevaplanması; sonra karşılaştırma
tablosu: 28 tam doküman üzerinde BM25 hit@1 **0.400**, bge-m3 **0.550**.

**Kabaca ne kadar sürüyor.** BM25 blokları anında dönüyor, 28 dokümanı embed etmek bir kereliğine
birkaç saniye sürüyor ve üzerinde konuşmaya değen tek şey iki generation bloğu. Aynı ölçümün tek
komutluk hali — `python eval/run_benchmark.py --skip-rerank` — ısınmış bir makinede yaklaşık 28 sn
ve 42 embed çağrısı.

Notebook içinde kodu yeniden yazmıyorsun, ortak kodu kullanıyorsun:

```python
from eval.retrieval import BM25, DenseRetriever, generate
from eval.metrics import load_gold, evaluate

bm25  = BM25(doc_ids, documents)
dense = DenseRetriever(doc_ids, documents)          # Ollama üzerinden bge-m3

bm25.rank("H9 1487 IST-CDG retiming")[:3]
bm25.rank("iptal edersem ne öderim")[:3]            # sıraya değil, skorlara bak
dense.rank("iptal edersem ne öderim")[:3]

questions = load_gold("eval/gold_questions.jsonl")
evaluate({q["id"]: dense.rank(q["query"]) for q in questions}, questions)
```

`python eval/run_benchmark.py --skip-rerank` bu sayfadaki tam doküman satırlarını basıyor ve hiç
chunk'lamıyor. Chunk satırları `python eval/run_benchmark.py --chunking` komutundan geliyor; o komut
merdivenin tamamını yürüyor ve bu modülün değil modül 7'nin ölçümü. Bütün flag'ler ve hangisinin neyi
ürettiği `eval/README.md` içinde.

<div class="presenter-note">
Blok süreleri: BM25 blokları anında dönüyor, 28 dokümanı embed etmek bir kereliğine birkaç saniye
sürüyor, üzerinde konuşulmaya değer tek blok 20 soruluk benchmark. Onu başlat, çalışırken salondan
hit@1 tahminini iste. Geriden geliyorsan İngilizce "give up the ticket" denemesini at ve Türkçe
sıfırdan doğrudan dense retriever'a geç; dört hata demosunu da koru, modül 6 ve modül 7 tam onların
üstüne açılıyor.
</div>

## Sayılar ne dedi

<div class="measured">

| retriever | hit@1 | recall@5 | MRR | index boyutu |
|---|---|---|---|---|
| BM25, tam dokümanlar | 0.400 | 0.583 | 0.515 | 28 doküman |
| dense, tam dokümanlar | 0.550 | 0.717 | 0.654 | 28 doküman |
| dense, sabit 280 karakter chunk | 0.700 | 0.950 | 0.814 | 294 chunk |
| aynı chunk'lar üzerinde BM25 | 0.300 | 0.617 | 0.483 | 294 chunk |
| dense, structure-aware chunk (modül 7) | 0.800 | 0.833 | 0.844 | 154 chunk |

soru tipine göre hit@1:

| soru tipi | BM25, tam dok. | dense, tam dok. | dense, sabit 280 |
|---|---|---|---|
| `tr_tr` (4) | 1.000 | 1.000 | 0.750 |
| `tr_en` (6) | 0.000 | 0.333 | 0.667 |
| `en_en` (4) | 0.250 | 0.250 | 0.750 |
| `exact_token` (4) | 0.750 | 1.000 | 0.750 |
| `multi_hop` (2) | 0.000 | 0.000 | 0.500 |

</div>

Dört `exact_token` sorusunun hepsi uçuş kodu değil. Üçü uçuş numarası veya bülten id'siyle yapılan
tarife araması; dördüncüsü, `q15`, booking class ile yapılan bir ücret tablosu araması — hata 1'deki
CLASSIC K iptal cezası. Etiketi "uçuş kodları" diye değil, "cevabı tek bir tabloda tek bir kesin string
olan sorular" diye oku.

Korpus: 28 doküman, 78.310 karakter. Gold set: 20 soru. Embedding `bge-m3`, generation `qwen2.5:3b`,
hepsi Ollama üzerinden lokal. Yirmi soru iki tasarım arasında karar vermeye yeter, yayımlamaya hiç
yetmez; burada 0.05'in altındaki fark gürültüdür.

<div class="presenter-note">
Bu modülde hiçbir şeyi düzeltme. Hata 1 ekrana düşer düşmez biri "overlap ekle" ya da "düzgün bir
splitter kullan" diye bağıracak — ikisini de tahtaya yaz, modül 7'de ölçüldüklerini ve birinin yanlış
olduğunu söyle. Modül 7 tam olarak o tahtaya dönerek açılıyor.
</div>

## Daha derine

BM25'in `idf` terimi asıl anlaşılması gereken kısım, çünkü keyword search'ün hâlâ ölmemiş olmasının
sebebi o. `idf` bir sürpriz ölçüsü: bir dokümanda `KSHEU26` görmek o doküman hakkında güçlü bir kanıt,
`the` görmek hiçbir şey. Dense embedding'in buna karşılık gelen bir düğmesi yok. Koca bir pasajı birkaç
yüz sayılık tek bir vektöre sıkıştırıyor ve nadir bir tanımlayıcı, içinde bulunduğu pasajın anlamının
minicik bir kesri olduğu için ortalamanın içinde eriyip gidiyor. Hata 3'ün mekanizması bu; chunk'ladıktan
sonra exact token'larda 1.000'den 0.750'ye kayışın mekanizması da bu. Yapısal bir şey, `bge-m3`'ün
bug'ı değil.

`b = 0.75` uzunluk normalizasyonu da chunking'in BM25'i neden mahvettiğinin diğer yarısı. Her şeyi 280
karaktere kes, her doküman ortalama uzunluğa gelir, `norm` herkes için 1 olur ve kısa bir bülteni uzun
bir anlaşmadan ayıran terim iş yapmayı bırakır. BM25 28 tam doküman üzerinde 0.400 alıyor; aynı
dokümanlardan kesilen 294 chunk üzerinde 0.300 — ve sahiplenmesi gereken exact-token sorularında 0.750
düşüp 0.250 oluyor. Onu bozan retriever değildi. Neyi index'leyeceğine dair verdiğin karardı.

BM25'in kendi profilini kategori kategori oku, asimetri her ortalamadan daha keskin. Türkçe sorgu /
Türkçe doküman şeklindeki dört soruda **1.000** alıyor — hepsi 1. sırada, ortada embedding modeli yok,
GPU yok, ağ yok. Türkçe sorgu / İngilizce doküman şeklindeki altı soruda **0.000** alıyor, hem de kıl
payıyla değil: doğru doküman 28 içinde 8. ile 21. sıralar arasında geliyor, yani bir kez bile ilk beşe
giremiyor. Bu, tek bir sorguda izlediğin `iptal` / `cancellation` boşluğunun altı soruya yayılmış hali.
BM25'in burada yapmadığı şey ise dense retrieval'ı herhangi bir sütunda yenmek: tam dokümanlar üzerinde
exact token'larda dense **1.000**, BM25 0.750; `tr_tr`'de ikisi de 1.000 ile berabere. Keyword search
beş dakikasını maliyetle, `SCB-2026-0914` gibi tek tanımlayıcılı sorgularla ve hiçbir zaman bir modele
ihtiyaç duymamasıyla hak ediyor — tabloyu kazanarak değil.

Top-k bir kez verip sonra verdiğini unuttuğun bir karar. Bu sayfadaki her metrik k'ye bağlı bir ifade:
sabit 280 chunking'de recall@5 0.950 iken hit@1 0.700 — yani doğru doküman neredeyse her zaman ilk beşte,
çoğu zaman da birinci değil. Generator yanlış dördü güvenilir şekilde görmezden gelseydi k=5 yapıp işi
bitirirdin. Görmezden gelmiyor: context'teki yanlış doküman, alıntılanmayı bekleyen bir yalandır ve
v3/v4 hatası tam olarak budur.

On milyon dokümanda `DenseRetriever`'daki brute-force döngü ölür. Yaklaşık bir index kurarsın — genelde
cevap HNSW — ve parametreyle kendin seçtiğin bir recall kaybını, doğrusal yerine logaritmik arama
karşılığında kabul edersin. Similarity hesaplamadan önce metadata ile filtrelersin; v3/v4 problemini
gerçekten çözen şey de bu: `status = current` tek satırlık bir filtre ve hiçbir embedding iyileştirmesi
onun yerini tutmaz. Çeyrek, route band ve doküman sürümü zaten elinde olan yapısal alanlar; korpusa düz
metin muamelesi yaparak onları çöpe atıyorsun.

Naive pipeline bir korkuluk değil. Yirmi iki satır, framework yok, vector database yok ve ağı kapalı bir
laptopta yirmi gold sorunun on birinde doğru dokümanı birinci sıraya koyuyor. Bundan sonraki her şey
0.550'ye karşı ölçülüyor ve popüler iyileştirmelerden ikisi bunu geçemeyecek.

## Çıkış cümlesi

> RAG çalışıyor ve retrieval çöp getiriyor. Onu düzeltmeden önce — bu embedding modelini
> ben senin yerine seçtim, ve neden seçtiğimi hiç söylemedim.
