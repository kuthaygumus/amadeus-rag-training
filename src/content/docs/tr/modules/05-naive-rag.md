---
title: "5. Keyword Search'ten Naive RAG'e"
description: "Doğru parçayı nasıl bulacağım?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Doğru parçayı nasıl bulacağım?**

Önceki modül çalışan bir cevapla ve bir faturayla bitti. 75 KB'ın tamamı context window'a
sığıyor, model EUR 90'ı doğru okuyor ve sen her soruda bütün kural kitabının parasını ödüyorsun —
yirmi sekiz dokümanın içinden tek bir paragrafa ihtiyaç duyan sorularda bile. Kural kitabı üç ayda
bir yeniden yayımlanıyor. Dört çeyrek ve altı istasyon olduğunda, maliyet canını yakmadan çok önce
prompt sığmayı bırakıyor.

O yüzden seçeceğiz. Yirmi sekiz doküman yerine üç doküman göndereceğiz. Günün geri kalanı tek bir
kelimeyle ilgili — seçmek — ve bu işin kaç farklı şekilde bozulduğuyla.

<div class="presenter-note">
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

`"H9 1487 IST-CDG retiming"` ile çalıştır, BM25 `bulletin_scb_2026_0914`'ü 9.481 skorla birinci
sıraya koyuyor; ikinci doküman 5.691. `1487` yirmi sekiz dokümanın dördünde geçiyor, ama sadece biri
tek sayfalık bir bülten — idf'in başlattığı işi uzunluk normalizasyonu bitiriyor. Tokenizer bilerek
kaba: alfanümerik olmayan her karakterden bölüyor, dolayısıyla `h9` ve `1487` ayrı token olarak
hayatta kalıyor. Kazanmasının sebebi tam olarak bu kabalık.

Şimdi Türk bir acentenin gerçekten yazdığı soruyu yaz:

> iptal edersem ne öderim

Sıfır. Kötü bir sıralama değil — `fare_classic_shorthaul` karşısında tam olarak 0.000, çünkü o
dokümanda ne `iptal` diye bir token var ne de `öderim`. Dokümanda `Cancellation penalty` yazıyor.
Onun yerine birinci sırada `macro_tr_noshow` geliyor, 7.638 skorla: başka bir soruyu cevaplayan bir
Türkçe doküman, sırf içinde `iptal` kelimesi geçtiği için tepede. BM25 yakın bir isabete doğru
yumuşakça düşmüyor — terim yok, `df` sıfır, döngü `continue` ediyor, doğru doküman hiç puan almıyor
ve yerini kendinden emin bir alakasızlık alıyor.

Sadece bir Türkçe problemi de değil. İngilizce sor — "what do I pay to give up the ticket" — ücret
sayfası 6. sırada kalıyor. Birinci `faq_en_general`, 14.736'ya karşı 3.699. Çünkü `pay` bu korpusta
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
cosine similarity'ye göre sırala. `DenseRetriever` bu, on dört satır.

Şimdi "iptal edersem ne öderim" diye sor; short-haul CLASSIC sayfası birinci geliyor. Ortak token yok,
sözlük yok, çeviri adımı yok. Neden `bge-m3` de framework'ün default kurduğu model değil — bunun tamamı
modül 6. Yirmi dakika boyunca buna güven.

Üstüne generator'ı tak: sorguyu embed et, top 3'ü al, "sadece bu context'i kullan" talimatının altına
yapıştır, üret. Uçtan uca otuz satır. Çalışıyor — ve asıl mesele skor.

<div class="presenter-note">
Benchmark hücresini çalıştırmadan önce taahhüt al: "hit@1, getirdiğimiz ilk dokümanın doğru doküman
olması demek. 20 sorudan kaçta kaçını tutturuyoruz? Bağırın." Salon 0.9 der. Sonra 0.550'yi göster.
O aradaki fark modülün kendisi. Ollama çöktüyse aynı tablo <code>eval/RESULTS.md</code> içinde —
oradan oku ve devam et, canlı debug'a girme.
</div>

## Gözünün önünde neyi yanlış yapıyor

**0.550.** Neredeyse iki soruda birinde modele verilen ilk doküman yanlış doküman. Model o soruyu
cevaplayamayacak bir kaynaktan cevaplıyor ve bunu bilmiyor.

Notebook'un açıkça gösterdiği üç hata var.

**Kendinden emin yanlış sayı.** Index'i tüm dokümanlardan sabit 280 karakterlik chunk'lara çevir — her
tutorial'ın yaptığı şey, ve ortalama skoru 0.650'ye çıkarıyor — sonra CLASSIC K iptal cezasını sor.
Gelen chunk'ın içinde `| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |`
satırı bozulmadan duruyor. Sütunları isimlendiren başlık bir önceki chunk'ta kaldı. Model iki euro
tutarı görüyor, etiket görmüyor ve **EUR 70** diyor. Doğru cevap **EUR 90**. Ne tereddüt ediyor, ne
soruyor; çıktıda bunu doğru cevapladığı anlardan ayıran hiçbir şey yok.

**Gerçeğin iki sürümü.** Beş saatlik bir misconnect'te kaç euroluk yemek fişi verileceğini sor.
`sop_misconnect_v4` EUR 15 ve 6 saat sonra otel diyor. `sop_misconnect_v3` EUR 10 ve 8 saat sonra otel
diyor. İkisi de korpusta, ikisi de misconnect hakkında, similarity skoru için ikisi de aynı görünüyor
ve sık sık aynı top-3'te birlikte geliyorlar. Aralarındaki tek fark `Version: 3 | Superseded` satırı;
retriever bunu okumuyor, modelin de bunu tartmak için bir sebebi yok. Retrieval'ın "güncel" diye bir
kavramı yok.

**Vektörlerin bulanıklaştırdığı kesin kod.** 28 tam doküman üzerinde exact-token soruları **1.000**
alıyor — bu ölçekte uçuş kodları sorunsuz. Aynı korpusu 288 küçük chunk'a böl, bu kategori **0.500**'e
düşüyor. Altı schedule bulletin aynı kalıp metni paylaşıyor ve birbirlerinden bir avuç token'la ayrılıyor;
dilimledikten sonra dilimler neredeyse aynı metne dönüşüyor ve `1487` taşıyan dilim öne çıkmayı bırakıyor.
BM25'in iyi olduğu iş, dense search'ün en kötü olduğu iş — ve chunking bunu daha da kötüleştiriyor.

## Ne çalıştırıyorsun

Notebook: `04_naive_rag.ipynb`.

```bash
ollama serve                       # çalışmıyorsa ikinci bir terminalde
ollama pull bge-m3
ollama pull qwen2.5:3b
python scripts/verify_setup.py     # devam etmeden önce yeşil yazmalı
jupyter lab notebooks/04_naive_rag.ipynb
```

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

Bütün stratejileri karşılaştıran tam koşu `python eval/run_benchmark.py` — bu sayfadaki her sayıyı
üreten şey o.

<div class="presenter-note">
Hücre süreleri: BM25 hücreleri anında dönüyor, 28 dokümanı embed etmek bir kereliğine birkaç saniye
sürüyor, üzerinde konuşulmaya değer tek hücre 20 soruluk benchmark. Onu başlat, çalışırken salondan
hit@1 tahminini iste. Geriden geliyorsan İngilizce "give up the ticket" denemesini at ve Türkçe
sıfırdan doğrudan dense retriever'a geç; üç hata demosunu da koru, modül 6 ve modül 7 tam onların
üstüne açılıyor.
</div>

## Sayılar ne dedi

<div class="measured">

| retriever | hit@1 | recall@5 | MRR | index boyutu |
|---|---|---|---|---|
| dense, tam dokümanlar | 0.550 | 0.717 | 0.655 | 28 doküman |
| dense, sabit 280 karakter chunk | 0.650 | 0.950 | 0.789 | 288 chunk |
| aynı chunk'lar üzerinde BM25 | 0.300 | — | 0.465 | 288 chunk |
| dense, structure-aware chunk (modül 7) | 0.800 | 0.850 | 0.846 | 152 chunk |

| kategori | tam dokümanlar | sabit 280 |
|---|---|---|
| exact-token sorular (uçuş kodu, bülten id) | 1.000 | 0.500 |
| Türkçe sorgu / İngilizce doküman sorularında BM25 | — | 0.000 |

</div>

Korpus: 28 doküman, 75 KB. Gold set: 20 soru. Embedding `bge-m3`, generation `qwen2.5:3b`, hepsi
Ollama üzerinden lokal. Yirmi soru iki tasarım arasında karar vermeye yeter, yayımlamaya hiç yetmez;
burada 0.05'in altındaki fark gürültüdür.

<div class="presenter-note">
Bu modülde hiçbir şeyi düzeltme. EUR 70 ekrana düşer düşmez biri "overlap ekle" ya da "düzgün bir
splitter kullan" diye bağıracak — ikisini de tahtaya yaz, modül 7'de ölçüldüklerini ve birinin yanlış
olduğunu söyle. Modül 7 tam olarak o tahtaya dönerek açılıyor.
</div>

## Daha derine

BM25'in `idf` terimi asıl anlaşılması gereken kısım, çünkü keyword search'ün hâlâ ölmemiş olmasının
sebebi o. `idf` bir sürpriz ölçüsü: bir dokümanda `KSHEU26` görmek o doküman hakkında güçlü bir kanıt,
`the` görmek hiçbir şey. Dense embedding'in buna karşılık gelen bir düğmesi yok. Koca bir pasajı birkaç
yüz sayılık tek bir vektöre sıkıştırıyor ve nadir bir tanımlayıcı, içinde bulunduğu pasajın anlamının
minicik bir kesri olduğu için ortalamanın içinde eriyip gidiyor. 1.000'den 0.500'e düşüşün mekanizması
bu; yapısal bir şey, `bge-m3`'ün bug'ı değil.

`b = 0.75` uzunluk normalizasyonu da chunking'in BM25'i neden mahvettiğinin diğer yarısı. Her şeyi 280
karaktere kes, her doküman ortalama uzunluğa gelir, `norm` herkes için 1 olur ve kısa bir bülteni uzun
bir anlaşmadan ayıran terim iş yapmayı bırakır. BM25 chunk'lar üzerinde 0.300 aldı, dense tam dokümanlar
üzerinde 0.550 aldı — ama tam dokümanlar üzerinde BM25 exact token'larda dense'i yeniyor. Aynı algoritma,
zıt karar; değişen tek şey neyi index'lediğin. Retrieval kararları birbirinden bağımsız değil; günün
bileşenleri değil kombinasyonları ölçmesinin sebebi bu.

Top-k bir kez verip sonra verdiğini unuttuğun bir karar. Bu sayfadaki her metrik k'ye bağlı bir ifade:
sabit 280 chunking'de recall@5 0.950 iken hit@1 0.650 — yani doğru doküman neredeyse her zaman ilk beşte,
çoğu zaman da birinci değil. Generator yanlış dördü güvenilir şekilde görmezden gelseydi k=5 yapıp işi
bitirirdin. Görmezden gelmiyor: context'teki yanlış doküman, alıntılanmayı bekleyen bir yalandır ve
v3/v4 hatası tam olarak budur.

On milyon dokümanda `DenseRetriever`'daki brute-force döngü ölür. Yaklaşık bir index kurarsın — genelde
cevap HNSW — ve parametreyle kendin seçtiğin bir recall kaybını, doğrusal yerine logaritmik arama
karşılığında kabul edersin. Similarity hesaplamadan önce metadata ile filtrelersin; v3/v4 problemini
gerçekten çözen şey de bu: `status = current` tek satırlık bir filtre ve hiçbir embedding iyileştirmesi
onun yerini tutmaz. Çeyrek, route band ve doküman sürümü zaten elinde olan yapısal alanlar; korpusa düz
metin muamelesi yaparak onları çöpe atıyorsun.

Naive pipeline bir korkuluk değil. Otuz satır, framework yok, vector database yok ve ağı kapalı bir
laptopta gold set'in yarısından fazlasını doğru cevaplıyor. Bundan sonraki her şey 0.550'ye karşı
ölçülüyor ve popüler iyileştirmelerden ikisi bunu geçemeyecek.

## Çıkış cümlesi

> RAG çalışıyor ve retrieval çöp getiriyor. Onu düzeltmeden önce — bu embedding modelini
> ben senin yerine seçtim, ve neden seçtiğimi hiç söylemedim.
