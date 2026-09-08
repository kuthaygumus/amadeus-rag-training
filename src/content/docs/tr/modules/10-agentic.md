---
title: "10. Agentic RAG — Finale"
description: "Ya tek atış yetmiyorsa?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Ya tek atış yetmiyorsa?**

Gold set'te `multi_hop` tipli iki soru var ve ikisi de bugün hiç kımıldamadı. hit@1'leri whole
documents'ta **0.000**, merdivendeki her chunking stratejisinde **0.500** — sabit boy, sabit boy
+ overlap, recursive, structure-aware; dördü de aynı. Modül 6'da karşılaştırdığımız üç
embedder'ın hepsinde de 0.500; diğer her şeyi 0.800'e çıkaran embedder dâhil. Pointwise rerank
da kımıldatmadı: en güçlü kurulumda q19'un ilk gold dokümanı rank 6'dan rank 5'e geldi, q20'ninki
zaten rank 1'deydi.

Bu 0.500 onları kayırıyor, çünkü doküman seviyesinde skorlama, gold dokümanlardan *biri* rank
1'e geldiğinde "isabet" diyor. Sayının sakladığı şey şu. q19'u elimizdeki en iyi tek geçişten
geçir — structure-aware chunk'lar üstünde `bge-m3` — ve dönen ilk beş doküman şu olur:

```
faq_en_general   macro_tr_rebook   codeshare_h9_au_conditions
sop_denied_boarding   macro_tr_misconnect
```

Sorunun ihtiyaç duyduğu üç dokümandan **üçte sıfırı**. Üstelik hata gibi de durmuyor.
`faq_en_general` ve `macro_tr_rebook` tam olarak bu durumu anlatıyor. İkisi de hiçbir rakam
taşımayı reddedip acenteyi başka yere yolluyor. Rank 1'de tek bir rakam olsaydı, üstünkörü
bir kontrolden geçerdi.

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

### Hangi fare sheet — RULE 7

Yolcunun elindeki tek bilette hem bir Avrupa Helios bacağı hem bir kıtalararası Aurora bacağı
var; yani iki fare sheet de geçerli olabilir. Corpus bunu, iki sayfada birden duran bir kuralla
çözüyor — `fare_classic_shorthaul` RULE 7 ve `fare_classic_longhaul` RULE 7:

> the sheet is chosen by the transaction and not by the document. A voluntary change to a single
> coupon is assessed on the sheet for the band of the Helios sector held [...] A cancellation or
> refund of the journey as a whole is assessed on the LONG-HAUL CLASSIC sheet.

q19 Avrupa bacağında isteğe bağlı bir değişiklik soruyor; o yüzden short-haul sheet geçerli ve
CLASSIC K change penalty **EUR 70**. q20 — Atlantik bandındaki diğer multi-hop soru — yolcu
yolculuktan tamamen vazgeçince ne tahsil edileceğini soruyor; o yüzden long-haul sheet geçerli
ve CLASSIC K cancellation penalty **EUR 195**. Short-haul'un iptal rakamını, EUR 90'ı vermek,
doğru satırı yanlış sayfadan okumak olur.

<div class="presenter-note">
Türkçe soruyu ekrana koy ve bir yolcunun anlattığı gibi, yavaşça yüksek sesle oku. Salona sor:
"bu soru kaç doküman istiyor?" Cevapları al — bir ve iki diyecekler. Ancak ondan sonra üç
olduğunu açıkla ve SOP ile anlaşmanın birbirinin sorusunu cevaplamayı reddettiği iki cümleyi
göster. RULE 7 projeksiyonda otuz saniyeyi hak ediyor, çünkü gold set'teki iki soru buna
dayanıyor. 4 dakika, laptoplar kapalı, henüz hiçbir şey çalışmıyor.
</div>

## Tek geçiş neden tek doküman getiriyor

Bu soruyu embed et, elinde tek bir vektör olur. Tek nokta — ama sorunun üç ağırlık merkezi var:
duty of care, interline koruması, ücret cezası. Nokta üçünün arasına, cümlenin en çok ağırlık
verdiğine yakın bir yere düşer ve top-5 o tek komşuluktan gelir. Tek retrieval geçişi sana tek
bir komşuluk verir ve q19'da o komşuluk yanlıştı.

Pipeline'daki hiçbir şey bunu düzeltemez, çünkü pipeline'da bozuk bir şey yok. Daha iyi bir
embedder noktayı taşır; ikiye bölmez. Daha iyi bir chunker adayları keskinleştirir; ikinci bir
sorgu eklemez. Rerank, zaten getirilmiş olanı yeniden sıralar — getirilenin içinde diğer iki
doküman hiç yoktu. Bugün çektiğimiz her kol tek bir geçişin üstünde çalışıyor. Sorun tek
geçişin kendisi.

## Döngü

Dört adım, yeni bir makine yok.

**Decompose.** Bir model çağrısı soruyu tek başına cevaplanabilir alt sorulara böler, en fazla
dört: beş saatlik beklemede hangi bakım veriliyor; AU segmenti korunuyor mu ve yeniden
fiyatlandırılıyor mu; CLASSIC K short-haul isteğe bağlı değişiklik cezası ne kadar.

**Her alt soru için ayrı retrieval.** Her biri kendi embedding'ini ve kendi top-k'sını alıyor —
aynı `bge-m3` index'i, modül 7'deki aynı structure-aware chunk'lar. Bir nokta yerine üç nokta.

**Yeterlilik kontrolü.** Bir model çağrısı `YES` diyor ya da `NO` deyip eksik olan için tek bir
kısa sorgu yazıyor. Açık uçlu değil, dar bir soru — pointwise rerank'in listwise'ı emekli probe
corpus'unda yenmesiyle aynı sebep (5 soru ve 10 doküman üzerinde 5/5'e 2/5; yön gerçek, bu
büyüklükte magnitude kanıtsız). Küçük model dar soruya iyi, geniş soruya kötü cevap verir.

**Citation'lı cevap,** her rakamın arkasındaki dokümanın adıyla.

## Koşu aslında ne üretiyor

Döngü dokümanlara ulaşıyor. q19'da **üçte sıfır**dan **üçte üç**e çıkıyor. Sonra yazdığı cevabı
oku, çünkü finalin temiz olmaktan çıktığı yer burası.

Kayıtlı koşuda model altı saatlik otel eşiğini on beş euroluk yemek tutarıyla birlikte, sanki
otelin bedeli oymuş gibi verdi; sonra son iki cümlesinde değişiklik ücreti konusunda kendisiyle
çelişti. Senin koşun başka kelimelerle yazacak — decompose üretildiği için günün determinist
olmayan tek parçası burası — ama biçim tekrar eder. Kelimeye değil, biçime bak.

Dokümanların gerçekte ne dediği, her biri geldiği paragrafla:

- **EUR 15** yemek fişi, çünkü bekleme üç saati aşıyor — `sop_misconnect_v4` 3.4
- **otel yok**, çünkü beş saat altıyı aşmıyor — `sop_misconnect_v4` 3.5
- Aurora segmenti korunuyor ve **yeniden fiyatlandırılmadan** yerleştiriliyor —
  `interline_h9_au` 4.1 ve 4.2
- isteğe bağlı değişiklik **EUR 70** — `fare_classic_shorthaul`, RULE 7 ve CLASSIC K satırı

Modelin hangi ikisi arasında kaydığına dikkat et. **EUR 70** change penalty, **EUR 90**
cancellation penalty — aynı K satırında bir sütun arayla duruyorlar; modül 7'nin tam olarak
konusu olan karışıklık. Header'ın o satıra hâlâ bağlı olmasının sebebi structure-aware
chunking; yani sütun okunmak üzere oradaydı. Yine de yanlış okundu. Retrieval düzeldi. Okuma
düzelmedi.

Günün dürüst kapanışı bu ve finali olduğundan derli toplu göstermek yerine bunu yüksek sesle
söylemek gerekiyor. Bu kurstaki her metrik **retrieval**'ı ölçüyor — doğru dokümanın gelip
gelmediğini. Hiçbiri cevabın doğru olup olmadığını ölçmüyor. Bunlar iki farklı hata biçimi olan
iki farklı sistem ve biz sadece birini ölçtük. Bugünden kendi projene tek bir şey taşıyacaksan:
önce retrieval eval'ini kur, çünkü ucuz ve determinist; sonra cevaplar için ikincisini kur,
çünkü birincisi ikincisinin bozuk olduğunu sana asla söylemeyecek.

**Ve döngü her zaman kazanmıyor.** q20'de üç dokümandan ikisine ulaştı — tek geçişin zaten
ulaştığı aynı ikisine. RULE 7'nin yolculuğun tümünün iptali için seçtiği sayfa,
`fare_classic_longhaul`, denediğimiz iki koşunun hiçbirinde gelmedi. İki soru bir yöntemi
ölçemez. Sadece böyle bir yöntemin var olduğunu ve bedavaya gelmediğini gösterebilir.

<div class="presenter-note">
Döngüyü çalıştırmadan önce salona modelin üreteceğini düşündükleri üç alt soruyu yazdır. Sonra
sadece decompose hücresini çalıştır ve karşılaştır. Kimseninkiyle birebir tutmayacak, ders de
bu: plan yazılmıyor, üretiliyor. Sonra cevap hücresini çalıştır ve corpus açıkken yüksek sesle
oku — alkışa yetişmek için kötü cevabın üstünden atlama. Ağzında gevelenmemesi gereken cümle:
<strong>retrieval artık modelden önce gelen bir adım değil, modelin çağırdığı bir tool.</strong>
Bir kez, net söyle; sonra sus.
</div>

## Aslında ne değişti

Şimdiye kadar şekil sabitti: retrieve, sonra generate. Retrieval modelden önce, bizim
kararımızla ve tam olarak bir kez oluyordu. Şimdi sorguları model açıyor, geleni okuyor ve
tekrar gidip gitmeyeceğine kendisi karar veriyor. Retrieval çağıranı olan bir tool'a dönüştü ve
akış kontrolü bizim kodumuzdan modelin çıktısına taşındı. Agent'ın tanımı tam olarak bu; ortada
kütüphane de yok: bir döngü, bir durma koşulu ve modelin bizden çalıştırmamızı isteyebileceği
bir fonksiyon.

## Fatura

**Daha çok çağrı.** Bir decompose, alt soru başına bir retrieval, tur başına bir yeterlilik
kontrolü, bir cevap: naive RAG'in bir çağrı yaptığı yerde altı ilâ on model çağrısı. Bu
geçişlerin üçü generation değil embedding çağrısı, yani ucuz olan kısım; generation'lar değil.
Ölçek için: modül 9'da fiyatladığımız reranker sorgu başına **8 model çağrısı** tutuyor ve en
güçlü kurulumda yirmi sorunun tamamında **65.2 saniye** sürdü — döngü aynı mertebe, yeni bir
kategori değil. Farklı olan şu: döngü çağrı başına gecikmeyi toplamıyor, tur sayısıyla çarpıyor.
Yani yavaş bir generation modeli sana biraz daha pahalıya gelmiyor; tur sayısı katı pahalıya
geliyor.

**Non-determinism.** Plan üretildiği için aynı sorunun iki koşusu farklı bölünebilir, farklı
doküman getirebilir, farklı citation verebilir. Temperature 0, ve her alt soruyu logla — debug
edeceğin artefakt o liste. O log yoksa yanlış cevabın sorumlusu bulunamaz.

**Daha çok geçiş, yanlış şeyi getirmek için daha çok şans.** Corpus'ta yürürlükten kalkmış
`sop_misconnect_v3.md` duruyor: **EUR 10** ve **8 saatlik** otel eşiği, v4'ün EUR 15 ve 6
saatine karşı. Her ek geçiş onu çekmek için bir fırsat daha. Chunk'lar modele ulaşmadan önce
sürüm satırına göre filtrele.

**Bitmeyen döngü.** Yeterlilik kontrolü her seferinde "hâlâ bir şey eksik" diyebiliyorsa,
diyecektir. Notebook bunu üç yerden sınırlıyor: sert bir `MAX_ROUNDS`, en fazla dört alt soru ve
bir tur yeni hiçbir şey getirmediğinde durma. Dördüncü sınırı production'da sen ekle: son turda
model elindekiyle cevap versin ve bulamadığını açıkça söylesin. Dürüst bir kısmi cevap, sonsuz
döngüden de kendinden emin uydurma bir cevaptan da iyidir — ki modül 1'de çıplak modelin
kurgusal bir havayoluna "20-30% ceza" uydurduğunu izlemiştik.

<div class="presenter-note">
Döngüyü canlı kır: yeterlilik prompt'unu asla YES diyemeyecek şekilde değiştir, tekrar çalıştır,
tur sayacının tavana çarpıp durduğunu izlet. On saniye — ve production'da ihtiyaç duyacakları
kısım tam olarak bu. Toplam 28 dakika: gate 4, tek geçiş hatası 5, döngü 10, cevabı okuma 5,
kırma 4. Aşağıdaki geri sarma bu bütçede değil — 14:33'teki 15 dakikalık kapanış bloğuna ait.
</div>

## Ne çalıştırıyorsun

**Bunu izliyorsun.** Notebook'u eğitmen sürüyor; dosya repoda duruyor ve sonrasında kendi
laptopunda, diğer her modülle aynı iki bağımlılıkla koşuyor. Canlı takip etmek istersen kimse
engellemiyor — ama burada salonun tek ekrana bakması daha verimli.

`notebooks/07_agentic_rag.py` dosyasını VS Code'da aç ve blokları `Shift+Enter` ile çalıştır.
`ollama serve` modül 0'dan beri zaten ayakta.

- **ne görmen gerekiyor** — tek geçiş hücresi `of the 3 documents needed, retrieval found 0`
  yazıyor; döngüden sonra `final coverage:` üçünü birden listeliyor; sondaki karşılaştırma
  tablosu q19'u `0 → 3`, q20'yi `2 → 2` olarak basıyor
- **kabaca ne kadar sürüyor** — döngü hücrelerinin her biri saniyeler; sondaki iki soruluk
  karşılaştırma M-serisi bir Mac'te 15 saniyenin altında koştu, sadece CPU'lu bir laptopta
  daha uzun

Benchmark'ın kullandığı fonksiyonların aynısı. Burada yeni bir bağımlılık yok:

```python
import _preflight; _preflight.ready(chat=True, embed=True)   # cwd -> notebooks/, eval/ path'e eklenir
from pathlib import Path
import retrieval as R, chunking as C, metrics

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("../corpus/2026-Q3").glob("*.md"))}
questions = metrics.load_gold("../eval/gold_questions.jsonl")
chunk_ids, chunk_texts, _ = C.chunk_corpus(docs, "structure-aware")
chunks = dict(zip(chunk_ids, chunk_texts))
dense = R.DenseRetriever(chunk_ids, chunk_texts)

task = [q for q in questions if q["type"] == "multi_hop"][0]      # q19

MAX_ROUNDS = 3
gathered: dict[str, str] = {}
for sub in decompose(task["query"]):                              # tek R.generate() çağrısı
    for chunk_id in dense.rank(sub)[:3]:
        gathered.setdefault(chunk_id, sub)                        # chunk id ile dedupe

for round_number in range(MAX_ROUNDS):                            # sert tur limiti
    enough, verdict = sufficient(task["query"], list(gathered))   # "YES" ya da NO + sorgu
    if enough:
        break
    follow_up = verdict.split("\n")[-1].lstrip("NO").strip(" .:,-") or task["query"]
    before = len(gathered)
    for chunk_id in dense.rank(follow_up)[:3]:
        gathered.setdefault(chunk_id, follow_up)
    if len(gathered) == before:
        break                                                     # yeni hiçbir şey gelmedi
```

`decompose` ve `sufficient`, notebook'un içinde yazılmış iki prompt; yani bu blok o iki hücreden
sonra çalışıyor, öncesinde değil. Geri kalan her şey —
`generate`, `DenseRetriever`, `chunk_corpus`, `load_gold` — modül 5'ten beri zaten koşan kod.
Tek geçiş hücresi hemen üstte duruyor, böylece iki çıktı tek ekranda yan yana geliyor: sıfır
dokümana karşı üç.

## Sayılar ne dedi

<div class="measured">

| | ölçüm |
|---|---|
| gold set'teki `multi_hop` soru sayısı | 20'de 2 |
| q19'un gerektirdiği gold doküman | 3 — `sop_misconnect_v4`, `interline_h9_au`, `fare_classic_shorthaul` |
| `multi_hop` hit@1, whole documents | 0.000 |
| `multi_hop` hit@1, merdivendeki dört chunk'lı stratejinin dördü de | 0.500 |
| `multi_hop` hit@1, structure-aware chunk'larda üç embedder'ın hepsi | 0.500 |
| q19, ilk gold dokümanın rank'ı — pointwise rerank öncesi / sonrası | 6 / 5 |
| genelde en iyi tek atış (bge-m3 + structure-aware) | hit@1 **0.800**, MRR **0.844** |
| q19'da ulaşılan gold doküman — tek geçiş / döngü | **3'te 0** / **3'te 3** |
| q20'de ulaşılan gold doküman — tek geçiş / döngü | 3'te 2 / 3'te 2 |
| pointwise vs listwise rerank, 10 doküman ve 5 soruluk emekli probe corpus | 5/5'e 2/5 (MRR 1.000 vs 0.600) — yön gerçek, n=5'te magnitude kanıtsız |
| bir rerank geçişinin maliyeti, en güçlü kurulum | sorgu başına 8 model çağrısı, 20 soruda 65.2 sn |
| yürürlükten kalkmış `sop_misconnect_v3` | EUR 10 fiş, 8 saatlik otel eşiği |
| güncel `sop_misconnect_v4` | EUR 15 fiş, 6 saatlik otel eşiği |

Döngünün kendisi **benchmark tablosunda yok**. İki multi-hop sorusu bir yöntemi ölçemez;
yalnızca böyle bir yöntemin var olduğunu gösterebilir. Bunu neyin çözeceği belli: elli etiketli
multi-hop sorusu, gold dokümanlardan birinin değil *hepsinin* getirilip getirilmediğine göre
skorlama ve retrieval'ı değil cevabı ölçen ikinci bir eval. O gelene kadar bu sayfanın iddiası
dar olan: q19'da tek geçiş üç dokümandan hiçbirini getirmedi, döngü üçünü de getirdi; q20'de
döngü, tek geçişin zaten getirdiğini getirdi.

</div>

## Daha derine

Decompose, bütçesi olan bir query rewriting'dir. Bugün retrieval kalitesi hakkında ölçtüğümüz
her şey her alt soru için hâlâ geçerli — ama artık metni insan değil model yazıyor. Ortalamada
daha iyi, çünkü modelin ifadesi corpus'un kelime dağarcığına doğru kayıyor; kuyrukta daha kötü,
çünkü kötü yazılmış bir alt soru sessizce başarısız oluyor. "Yanlış şeyi güzel sordun" diye bir
hata sinyali yok.

Yeterlilik kontrolü hem en ilginç hem en zayıf parça. 3B'lik bir modelden kendi kanıtının
eksiksizliğine hüküm vermesini istiyorsun ve modül 9 bu hükmün kaç para ettiğini ölçtü: zaten
kendisinden iyi olan bir sıralamaya dayatıldığında **kendi tavanına düzlüyor** — hit@1 0.800'den
0.600'e, MRR 0.844'ten 0.717'ye. Çözüm hükmü daraltmak: "bu yeterli mi?" değil, "getirilen metin
isteğe bağlı değişiklik için bir euro tutarı söylüyor mu?" Alt sorulardan türetilen bir
checklist, bulanık bir kararı lookup'a çevirir. Pointwise'ın listwise'ı yenmesiyle aynı ders,
bir kat yukarıda.

Döngünün neye ihtiyaç duymadığına bak: planner framework yok, tool-calling API yok, agent
class'ı yok. `eval/retrieval.py`'den iki şey — `generate` ve `DenseRetriever` — artı
notebook'ta yazdığın üç prompt ve içinde break olan bir `for`. Çoğu agent framework'ü bunun
üstüne retry, tracing ve şema demek. Framework'ü davranışı elde etmek için değil, tracing'e
ihtiyacın olduğunda al.

10 milyon dokümanda döngü bedava olmaktan çıkar. Her ek geçiş yeni bir tam ANN araması demek;
o yüzden fan-out'u sınırla ve alt soru seviyesinde cache'le — aynı üç alt soru binlerce
misconnect vakasında tekrar ediyor ve normalize edilmiş alt soruyla anahtarlanmış bir cache her
model değişikliğinden fazla kazandırır. Önüne bir router koy, çünkü soruların çoğu tek adımlık
ve decompose'un bedelini hiç ödememeli. Tool'ları açık ve tipli yap — `search_sop`,
`search_fare_rules`, `search_interline`, her biri kendi filtresiyle — ki model tek bir index'in
her şeyi kapsamasını ummak yerine bir corpus seçsin. v3'ü dışarıda tutan sürüm filtresi de
oraya, sıralamaya dair bir umuda değil tool'un bir özelliğine ait. RULE 7'nin ayrımı da öyle:
bir change mi yoksa bir cancellation mı fiyatladığını bilen bir tool, sayfayı retrieval'dan önce
seçebilir — 3B'lik bir modelden iki dokümanın da sonuna gömülü bir kuralı fark etmesini
istemek yerine.

Dürüst sınır: hiç yazılmamış bir dokümanı hiçbir geçiş sayısı üretemez. Interline partnerin
kendi SOP'si bizimkiyle çelişiyorsa ve bunu çözen bir paragraf yoksa, döngünün hata biçimi var
olmayan metni aramaya devam etmektir. Sınırla ve bunu söylet.

## Günü geri sarmak

Burası kapanış bloğu, modül 10'un 28 dakikasının parçası değil. Zinciri tersten, yüksek sesle,
tek nefeste oku.

Model bizim verimizi bilmiyordu ve bilmediğini de bilmiyordu — "20-30% ceza" uydurdu. Sonra
training'in ne olduğuna baktık ve weight'lerin donmuş bir fotoğraf olduğunu gördük. Fine-tune
ettik, çalıştı, sonra Q2 Q3 oldu: **EUR 120, EUR 90'a döndü**, weight'ler hâlâ 120 diyordu ve
hiçbir kaynak gösteremiyorlardı. Bütün kural kitabını prompt'a koyduk, sığdı, ve her sorguda
78,310 karakterin tamamını ödedik. Seçmeye başladık — retrieval çöp getirdi. Embedder'ı
değiştirdik, çünkü default sadece İngilizceydi ve kimse uyarmamıştı. Chunker'ı değiştirdik ve
hatanın satır değil header olduğunu bulduk. Gerçek olsun diye ChromaDB'ye koyduk. Hybrid ve
rerank ekledik ve bir reranker'ın upgrade değil takas olduğunu ölçtük. Sonra iki soru aynı anda
üç doküman istedi ve kurstaki hiçbir tek atışlık yöntem onları **0.500**'ün üstüne çıkaramadı —
q19'da ihtiyaç duyduğu üç dokümandan hiçbiri geri gelmedi bile.

On gate, hiçbiri tanımla açılmadı. Her biri bir önceki modülün çarptığı duvardı. Bütün gün tek
bir sayı: **EUR 90** — bir dokümandan alıntılanmış ve dokümanın adı verilmiş hâliyle. Bir de tek
bir çekince: dokümanın gelip gelmediğini ölçtük, üstüne kurulan cümlenin doğru olup olmadığını
hiç ölçmedik.

<div class="presenter-note">
Geri sarmayı ayakta yap; slayt yok, laptop yok. Günün son bölümü ve bir meslektaşına
anlatacakları tek kısım burası. EUR 90 cümlesiyle bitir, sonra çıkış cümlesi, sonra konuşmayı
kes. Arkasına özet slaytı koyma. Burası 14:33'teki 15 dakikalık kapanış bloğu; repo linki ve
handout da burada veriliyor.
</div>

## Çıkış cümlesi

> Bugünkü her adım, bir öncekinin yetmediği yerde doğdu. Sonuncusu RAG'i bir agent'a
> dönüştürdü — agent günü de tam buradan devralıyor.
