---
title: "10. Agentic RAG — Finale"
description: "Ya tek atış yetmiyorsa?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Ya tek atış yetmiyorsa?**

Gold set'te bugün kurduğumuz her yöntemin patladığı bir soru var. Keyword search'te patladı,
dense retrieval'da patladı, hit@1'i **0.800**'e çıkaran structure-aware chunking'de patladı,
rerank'ten sonra da patladı. İki `multi_hop` sorusu bu kurstaki her tek atışlık yöntemde en iyi
**0.500** alıyor — ve bu 0.500 onları kayırıyor, çünkü doküman seviyesinde skorlama, gold
dokümanlardan *biri* rank 1'e geldiğinde "isabet" diyor. Bu sorunun üç tanesine ihtiyacı var.

Soru, bir acentenin gerçekten yazacağı hâliyle:

> H9 1487 gecikti, CLASSIC K sınıfındaki yolcum CDG'de AU 88 bağlantısını kaçırıyor ve yeni
> uçuşa kadar 5 saat bekleyecek. Bu beklemede kendisine ne vermem gerekiyor, Aurora bacağı
> yeniden fiyatlandırılır mı, bir de yolcu bugünkü seferi hiç istemeyip kendi isteğiyle yarına
> geçmek isterse ondan ne kadar değişiklik ücreti alırım?

Tek paltonun altında üç soru. Beklemedeki bakım `sop_misconnect_v4.md` içinde. Aurora
kuponunun ayakta kalıp kalmadığı `interline_h9_au.md`'nin Clause 4'ünde. Değişiklik ücreti
`fare_classic_shorthaul.md`'de. Hiçbir dosya üçünden ikisini birden tutmuyor ve corpus bilerek
böyle kurulmuş: SOP kendi içinde "establishes no monetary penalty, waiver or refund value of
any kind" diyor, Clause 4.4 ise anlaşmanın hiçbir fişin tutarını belirlemediğini söylüyor.
Dokümanlar birbirini işaret ediyor. Gerçek kural kitapları da böyle yapar.

<div class="presenter-note">
Türkçe soruyu ekrana koy ve bir yolcunun anlattığı gibi, yavaşça yüksek sesle oku. Salona sor:
"bu soru kaç doküman istiyor?" Cevapları al — bir ve iki diyecekler. Ancak ondan sonra üç
olduğunu açıkla ve SOP ile anlaşmanın birbirinin sorusunu cevaplamayı reddettiği iki cümleyi
göster. 4 dakika, laptoplar kapalı, henüz hiçbir şey çalışmıyor.
</div>

## Tek geçiş neden tek doküman getiriyor

Bu soruyu embed et, elinde tek bir vektör olur. Tek nokta — ama sorunun üç ağırlık merkezi var:
duty of care, interline koruması, ücret cezası. Nokta üçünün arasına, cümlenin en çok ağırlık
verdiğine yakın bir yere düşer ve top-5 o tek komşuluktan gelir. Tek retrieval geçişi sana
üçünden birini verir.

Pipeline'daki hiçbir şey bunu düzeltemez, çünkü pipeline'da bozuk bir şey yok. Daha iyi bir
embedder noktayı taşır; ikiye bölmez. Daha iyi bir chunker adayları keskinleştirir; ikinci bir
sorgu eklemez. Rerank, zaten getirilmiş olanı yeniden sıralar — getirilenin içinde diğer iki
doküman hiç yoktu. Bugün çektiğimiz her kol tek bir geçişin üstünde çalışıyor. Sorun tek
geçişin kendisi.

## Döngü

Dört adım, yeni bir makine yok.

**Decompose.** Bir model çağrısı soruyu alt sorulara böler, satır başına bir tane, en fazla
dört: beş saatlik beklemede hangi bakım veriliyor; AU segmenti korunuyor mu ve yeniden
fiyatlandırılıyor mu; CLASSIC K short-haul isteğe bağlı değişiklik cezası ne kadar.

**Her alt soru için ayrı retrieval.** Her biri kendi embedding'ini ve kendi top-k'sını alıyor —
aynı `bge-m3` index'i, modül 7'deki aynı structure-aware chunk'lar. Bir nokta yerine üç nokta.

**Yeterlilik kontrolü.** Bir model çağrısı `YES` diyor ya da eksik olanın adını veriyor. Düz
yazı değil, tek token — pointwise rerank'in listwise'ı **5/5'e 2/5** yenmesiyle aynı sebep:
küçük model dar soruya iyi, geniş soruya kötü cevap verir.

**Citation'lı cevap,** her rakamın arkasındaki dokümanın adıyla.

Döngünün döndürdüğü: **EUR 15** yemek fişi, çünkü bekleme üç saati aşıyor (`sop_misconnect_v4`
3.4); otel yok, çünkü beş saat altıyı aşmıyor (3.5); Aurora segmenti korunuyor ve **yeniden
fiyatlandırılmadan** yerleştiriliyor (`interline_h9_au` 4.1 ve 4.2); yarına isteğe bağlı geçiş
**EUR 70** (`fare_classic_shorthaul`, CLASSIC K, change penalty).

Son rakam günü kendi üstüne kapatıyor. **EUR 70, modül 7'den beri peşinde koştuğumuz yanlış
cevaptı** — cancellation sütunu yerine change sütununun okunması. Burada doğru, çünkü yolcu
gerçekten isteğe bağlı bir değişiklik istiyor; ve döngü iki sütunu ancak structure-aware
chunking header'ı K satırına bağlı tuttuğu için ayırt edebiliyor. Nihai cevap doğru, çünkü üç
modül önce bir string'in nereden kesileceğine dair bir karar verildi.

<div class="presenter-note">
Döngüyü çalıştırmadan önce salona modelin üreteceğini düşündükleri üç alt soruyu yazdır. Sonra
sadece decompose hücresini çalıştır ve karşılaştır. Kimseninkiyle birebir tutmayacak, ders de
bu: plan yazılmıyor, üretiliyor. Ağzında gevelenmemesi gereken cümle: <strong>retrieval artık modelden önce gelen bir
adım değil, modelin çağırdığı bir tool.</strong> Bir kez, net söyle; sonra sus.
</div>

## Aslında ne değişti

Şimdiye kadar şekil sabitti: retrieve, sonra generate. Retrieval modelden önce, bizim
kararımızla ve tam olarak bir kez oluyordu. Şimdi sorguları model açıyor, geleni okuyor ve
tekrar gidip gitmeyeceğine kendisi karar veriyor. Retrieval çağıranı olan bir tool'a dönüştü ve
akış kontrolü bizim kodumuzdan modelin çıktısına taşındı. Agent'ın tanımı tam olarak bu; ortada
kütüphane de yok: bir döngü, bir durma koşulu ve modelin bizden çalıştırmamızı isteyebileceği
bir fonksiyon.

## Fatura

**Daha çok çağrı.** Bir decompose, üç retrieval, bir yeterlilik kontrolü, bir cevap: naive
RAG'in bir çağrı yaptığı yerde altı çağrı. Ölçülen `qwen2.5:3b` üretim maliyeti **0,9 sn**
üzerinden bu bir aritmetik, ölçüm değil — ve tek haneli saniyelere denk geliyor; zaten sorgu
başına 8 çağrıyla **~4 sn** tutan rerank geçişiyle aynı mertebe. Yerine, reasoning token
ürettiği için **11,6 sn** ölçülen `qwen3:4b`'yi koy, aynı döngü bir dakikaya çıkar. Döngüde
çağrı başına gecikme çarpılır.

**Non-determinism.** Plan üretildiği için aynı sorunun iki koşusu farklı bölünebilir, farklı
doküman getirebilir, farklı citation verebilir. Temperature 0, ve her alt soruyu logla — debug
edeceğin artefakt o liste. O log yoksa yanlış cevabın sorumlusu bulunamaz.

**Daha çok geçiş, yanlış şeyi getirmek için daha çok şans.** Corpus'ta yürürlükten kalkmış
`sop_misconnect_v3.md` duruyor: **EUR 10** ve **8 saatlik** otel eşiği, v4'ün EUR 15 ve 6
saatine karşı. Her ek geçiş onu çekmek için bir fırsat daha. Chunk'lar modele ulaşmadan önce
sürüm satırına göre filtrele.

**Bitmeyen döngü.** Yeterlilik kontrolü her seferinde "hâlâ bir şey eksik" diyebiliyorsa,
diyecektir. Üç yerden sınırla: en fazla iki ek tur, en fazla dört alt soru ve son turda modelin
elindekiyle cevap verip bulamadığını açıkça söylemesi kuralı. Dürüst bir kısmi cevap, sonsuz
döngüden de kendinden emin uydurma bir cevaptan da iyidir — ki modül 1'de çıplak modelin
kurgusal bir havayoluna "20-30% ceza" uydurduğunu ölçmüştük.

<div class="presenter-note">
Döngüyü canlı kır: yeterlilik prompt'unu asla YES diyemeyecek şekilde değiştir, tekrar çalıştır,
tur sayacının tavana çarpıp durduğunu izlet. On saniye — ve production'da ihtiyaç duyacakları
kısım tam olarak bu. Laptoplar yavaşsa döngüyü sadece projeksiyonda bir kez çalıştır, onlar
loglanan alt soruları okusun. Toplam 35 dakika: gate 4, tek geçiş hatası 6, döngü 12, kırma 5,
geri sarma 8.
</div>

## Ne çalıştırıyorsun

Notebook: `07_agentic_rag.ipynb`

```bash
ollama serve                                  # modül 0'dan beri zaten ayakta
jupyter lab notebooks/07_agentic_rag.ipynb
```

Benchmark'ın kullandığı fonksiyonların aynısı. Burada yeni bir bağımlılık yok:

```python
from retrieval import DenseRetriever, generate
from chunking import chunk_corpus, to_documents
from metrics import load_gold

gold = load_gold()
q19 = next(q for q in gold if q["id"] == "q19")     # üç dokümanlık soru

ids, texts, parents = chunk_corpus(documents, "structure-aware")
index = DenseRetriever(ids, texts)

subqs = decompose(q19["query"], max_parts=4)        # tek generate() çağrısı
seen = {}
for _ in range(3):                                  # sert tur limiti
    for sq in subqs:
        for cid in index.rank(sq)[:3]:
            seen[cid] = texts[ids.index(cid)]       # chunk id ile dedupe
    missing = check_sufficient(q19["query"], seen)  # "YES" ya da eksik olan
    if missing == "YES":
        break
    subqs = [missing]
print(answer_with_citations(q19["query"], seen))
```

Aynı soruyu bir üstteki hücrede tek geçişli retrieval'dan da geçir ki iki çıktı tek ekranda
yan yana dursun. Bir dokümana karşı üç doküman.

## Sayılar ne dedi

<div class="measured">

| | ölçüm |
|---|---|
| gold set'teki `multi_hop` soru sayısı | 20'de 2 |
| q19'un gerektirdiği gold doküman | 3 (`sop_misconnect_v4`, `interline_h9_au`, `fare_classic_shorthaul`) |
| en iyi `multi_hop` skoru, tek atışlık her yöntemde | **0.500** |
| genelde en iyi tek atış (bge-m3 + structure-aware) | hit@1 **0.800**, MRR **0.846** |
| pointwise vs listwise rerank, probe corpus | 5/5 vs 2/5 (MRR 1.000 vs 0.600) |
| bir rerank geçişinin maliyeti | 8 model çağrısı, sorgu başına ~4 sn |
| `qwen2.5:3b` üretim | 0,9 sn, 3/3 doğru |
| `qwen3:4b` üretim | 11,6 sn (reasoning token) |
| yürürlükten kalkmış `sop_misconnect_v3` | EUR 10 fiş, 8 saatlik otel eşiği |
| güncel `sop_misconnect_v4` | EUR 15 fiş, 6 saatlik otel eşiği |

Döngünün kendisi **benchmark tablosunda yok**. İki multi-hop sorusu bir yöntemi ölçemez;
yalnızca böyle bir yöntemin var olduğunu gösterebilir. Bunu neyin çözeceği belli: elli etiketli
multi-hop sorusu ve gold dokümanlardan birinin değil, *hepsinin* getirilip getirilmediğine göre
skorlama. O gelene kadar bu sayfanın iddiası dar olan: tek geçiş üç dokümandan birini getirdi,
döngü üçünü de getirdi.

</div>

## Daha derine

Decompose, bütçesi olan bir query rewriting'dir. Bugün retrieval kalitesi hakkında ölçtüğümüz
her şey her alt soru için hâlâ geçerli — ama artık metni insan değil model yazıyor. Ortalamada
daha iyi, çünkü modelin ifadesi corpus'un kelime dağarcığına doğru kayıyor; kuyrukta daha kötü,
çünkü kötü yazılmış bir alt soru sessizce başarısız oluyor. "Yanlış şeyi güzel sordun" diye bir
hata sinyali yok.

Yeterlilik kontrolü hem en ilginç hem en zayıf parça. 3B'lik bir modelden kendi kanıtının
eksiksizliğine hüküm vermesini istiyorsun ve modül 9 bu hükmün kaç para ettiğini ölçtü: zaten
kendisinden iyi olan bir sıralamaya dayatıldığında **kendi tavanına düzlüyor**, 0.800'den
0.650'ye. Çözüm hükmü daraltmak: "bu yeterli mi?" değil, "getirilen metin isteğe bağlı
değişiklik için bir euro tutarı söylüyor mu?" Alt sorulardan türetilen bir checklist, bulanık
bir kararı lookup'a çevirir. Pointwise'ın listwise'ı yenmesiyle aynı ders, bir kat yukarıda.

Döngünün neye ihtiyaç duymadığına bak: planner framework yok, tool-calling API yok, agent
class'ı yok. `eval/retrieval.py`'den altı fonksiyon ve içinde break olan bir `for`. Çoğu agent
framework'ü bunun üstüne retry, tracing ve şema demek. Framework'ü davranışı elde etmek için
değil, tracing'e ihtiyacın olduğunda al.

10 milyon dokümanda döngü bedava olmaktan çıkar. Her ek geçiş yeni bir tam ANN araması demek;
o yüzden fan-out'u sınırla ve alt soru seviyesinde cache'le — aynı üç alt soru binlerce
misconnect vakasında tekrar ediyor ve normalize edilmiş alt soruyla anahtarlanmış bir cache her
model değişikliğinden fazla kazandırır. Önüne bir router koy, çünkü soruların çoğu tek adımlık
ve decompose'un bedelini hiç ödememeli. Tool'ları açık ve tipli yap — `search_sop`,
`search_fare_rules`, `search_interline`, her biri kendi filtresiyle — ki model tek bir index'in
her şeyi kapsamasını ummak yerine bir corpus seçsin. v3'ü dışarıda tutan sürüm filtresi de
oraya, sıralamaya dair bir umuda değil tool'un bir özelliğine ait.

Dürüst sınır: hiç yazılmamış bir dokümanı hiçbir geçiş sayısı üretemez. Interline partnerin
kendi SOP'si bizimkiyle çelişiyorsa ve bunu çözen bir paragraf yoksa, döngünün hata biçimi var
olmayan metni aramaya devam etmektir. Sınırla ve bunu söylet.

## Günü geri sarmak

Zinciri tersten, yüksek sesle, tek nefeste oku.

Model bizim verimizi bilmiyordu ve bilmediğini de bilmiyordu — "20-30% ceza" uydurdu. Sonra
training'in ne olduğuna baktık ve weight'lerin donmuş bir fotoğraf olduğunu gördük. Fine-tune
ettik, çalıştı, sonra Q2 Q3 oldu: **EUR 120, EUR 90'a döndü**, weight'ler hâlâ 120 diyordu ve
hiçbir kaynak gösteremiyorlardı. Bütün kural kitabını prompt'a koyduk, sığdı, ve her sorguda 75
KB'ın tamamını ödedik. Seçmeye başladık — retrieval çöp getirdi. Embedder'ı değiştirdik, çünkü
default sadece İngilizceydi ve kimse uyarmamıştı. Chunker'ı değiştirdik ve hatanın satır değil
header olduğunu bulduk. Gerçek olsun diye ChromaDB'ye koyduk. Hybrid ve rerank ekledik ve bir
reranker'ın upgrade değil takas olduğunu ölçtük. Sonra tek bir soru aynı anda üç doküman
istedi ve kurstaki her tek atışlık yöntem o soruda 0.500 aldı.

On gate, hiçbiri tanımla açılmadı. Her biri bir önceki modülün çarptığı duvardı. Bütün gün tek
bir sayı: **EUR 90** — bir dokümandan alıntılanmış ve dokümanın adı verilmiş hâliyle.

<div class="presenter-note">
Geri sarmayı ayakta yap; slayt yok, laptop yok. Günün son sekiz dakikası ve bir meslektaşına
anlatacakları tek bölüm burası. EUR 90 cümlesiyle bitir, sonra çıkış cümlesi, sonra konuşmayı
kes. Arkasına özet slaytı koyma.
</div>

## Çıkış cümlesi

> RAG'in bir agent'a dönüşmesini az önce izledin.
