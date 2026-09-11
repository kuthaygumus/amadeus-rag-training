---
title: "10. Agentic RAG — Finale"
description: "Ya tek atış yetmiyorsa?"
---

## Gate sorusu

> **Ya tek atış yetmiyorsa?**

Gold set'te `multi_hop` tipli iki soru var ve ikisi de bugün hiç kımıldamadı. hit@1'leri whole
documents'ta **0.000**, merdivenin her basamağında ise **0.500** — sabit boy, sabit boy + overlap,
recursive, structure-aware; boilerplate temizlenmiş hâliyle de temizlenmemiş hâliyle de —
onu sıfırdan çıkaran tek şey chunklama, ve 0.500 bir başarı değil. Fusion bunu da geri alıyor:
BM25 ve RRF chunk'lar üzerinde yeniden 0.000'a düşüyor, tek dense retrieval 0.500'ü tutuyor.
Structure-aware chunk'larda üç embedder'da satır 0.500, 0.500 ve 0.500 okunuyor — MiniLM,
`nomic-embed-text` ve `bge-m3` aynı şekilde — ki `eval/RESULTS.md` bunu kimsenin yeteneği değil,
dokümanlar chunklandığında iki sorudan birinin erişilebilir hâle gelmesi olarak okumamızı
söylüyor.

Bugün çekilen tek başka kol rerank oldu: en güçlü kurulumda rank 1'in altından tam olarak bir
soruyu yukarı çekti — q14, `multi_hop` çiftinden hiçbiri değil. `UNVERIFIED: rerank'in doküman seviyesinde multi_hop satırına ne
yaptığı — bunu basan bayraksız benchmark koşusu bu yeniden ölçüme dâhil edilmedi, dolayısıyla
satırı 0.000'dan 0.500'e çıkardığı yönündeki eski iddia geri çekildi.`

Doküman seviyesinde skorlama bu ikisini, bir puan alabilselerdi, kayırırdı: üç gold dokümandan
*biri* rank 1'e geldiğinde "isabet" diyor. O kadarına bile ulaşamadılar. q19'u elimizdeki en iyi
tek geçişten geçir — structure-aware chunk'lar üstünde `bge-m3` — ve ihtiyaç duyduğu üç dokümandan
ilk beşte **3'te 0** çıkar.

Üstelik hata gibi de durmuyor. Onun yerine gelenleri q19'un kendi `why` alanı sayıyor:
`macro_tr_misconnect`, `macro_tr_rebook`, `faq_en_general` — üçü de tam olarak bu durumu
anlatıyor, üçü de yeniden fiyatlandırma yapılmadığı kuralını söylüyor ve hiçbiri rakam taşımıyor.
Rank 1'deki makul bir doküman üstünkörü bir kontrolden geçer. Soru doğru görünürken sıfır alıyor.

Sorunun kendisi, `eval/gold_questions.jsonl` içindeki q19, saklandığı hâliyle:

> XX 1487 gecikti, CLASSIC K sınıfındaki yolcum CDG'de YY 88 bağlantısını kaçırıyor ve yeni
> uçuşa kadar 5 saat bekleyecek. Bu beklemede kendisine ne vermem gerekiyor, Wyvern bacağı
> yeniden fiyatlandırılır mı, bir de yolcu bugünkü seferi hiç istemeyip kendi isteğiyle yarına
> geçmek isterse ondan ne kadar değişiklik ücreti alırım?

Tek paltonun altında üç soru. Beklemedeki bakım `sop_misconnect_v4.md` içinde; Wyvern kuponunun
ayakta kalıp kalmadığı `interline_xx_yy.md`'nin Clause 4'ünde; değişiklik ücreti
`fare_classic_shorthaul.md`'de. Hiçbir dosya üçünden ikisini birden tutmuyor ve corpus bilerek
böyle kurulmuş: SOP "establishes no monetary penalty, waiver or refund value of any kind" diyor,
Clause 4.4 ise hiçbir fişin tutarını belirlemiyor. Dokümanlar birbirini işaret ediyor. Gerçek
kural kitapları da böyle yapar.

### Hangi fare sheet — RULE 7

Tek bir bilette iki fare sheet birden geçerli olabilir; corpus bunu iki sayfada birden duran bir
kuralla çözüyor — `fare_classic_shorthaul` RULE 7 ve `fare_classic_longhaul` RULE 7:

> the sheet is chosen by the transaction and not by the document. A voluntary change to a single
> coupon is assessed on the sheet for the band of the Kraken sector held [...] A cancellation or
> refund of the journey as a whole is assessed on the LONG-HAUL CLASSIC sheet.

Yani q19, Avrupa bacağında isteğe bağlı bir değişiklik olduğu için short-haul sheet'e gidiyor:
CLASSIC K değişiklik cezası **EUR 70**. q20, yolculuğun tümünün iptali olduğu için long-haul
sheet'e gidiyor: CLASSIC K iptal cezası **EUR 195**. Short-haul'un iptal rakamını, EUR 90'ı
vermek, doğru satırı yanlış sayfadan okumak olur.

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
verdiğine yakın bir yere düşer ve top-5 o tek komşuluktan gelir. q19'da o komşuluk yanlıştı.

Pipeline'daki hiçbir şey bunu düzeltemez, çünkü pipeline'da bozuk bir şey yok. Daha iyi bir
embedder noktayı taşır; ikiye bölmez. Daha iyi bir chunker adayları keskinleştirir; ikinci bir
sorgu eklemez. Rerank, zaten getirilmiş olanı yeniden sıralar — getirilenin içinde diğer iki
doküman hiç yoktu. Bugün çektiğimiz her kol tek bir geçişin üstünde çalışıyor. Sorun tek
geçişin kendisi.

## Döngü

Dört adım, yeni bir makine yok.

**Decompose.** Bir model çağrısı soruyu tek başına cevaplanabilir alt sorulara böler, en fazla
dört: beş saatlik beklemede hangi bakım veriliyor; YY segmenti korunuyor mu ve yeniden
fiyatlandırılıyor mu; CLASSIC K short-haul isteğe bağlı değişiklik cezası ne kadar.

**Her alt soru için ayrı retrieval.** Her biri kendi embedding'ini ve kendi top-k'sını alıyor —
aynı `bge-m3` index'i, modül 7'deki aynı structure-aware chunk'lar. Bir nokta yerine üç nokta.

**Yeterlilik kontrolü.** Bir model çağrısı `YES` diyor ya da `NO` deyip eksik olan için tek bir
kısa sorgu yazıyor. Açık uçlu değil, dar bir soru — modül 9'da pointwise rerank'in çalışıp
listwise'ın çalışmamasıyla aynı sebep: altı pasaj verilip sıraya dizmesi istendiğinde
`qwen2.5:3b` `1,4,2,5` döndürdü, altı pasaj için dört indeks; teker teker puanlaması istendiğinde
altı çağrıda ve 3.3 saniyede kullanılabilir altı tam sayı verdi. Küçük model dar soruya iyi,
geniş soruya kötü cevap verir.

**Citation'lı cevap,** her rakamın arkasındaki dokümanın adıyla.

Şimdiye kadar şekil sabitti: retrieve, sonra generate — bizim kararımızla ve tam olarak bir kez.
Artık retrieval'ın bir çağıranı var ve ortada kütüphane de yok: bir döngü, bir durma koşulu ve
modelin bizden çalıştırmamızı isteyebileceği bir fonksiyon. Bunun nereye kadar gittiği konusunda
net olalım. Model *ne zaman duracağına* ve *sırada ne soracağına* karar veriyor; onunla ne
yapılacağına bizim kodumuz karar veriyor. Tool *seçimi* hâlâ bizde, çünkü ortada tek bir tool
var — ve açık hâle getirilecek bir sonraki şey de bu.

## Koşu aslında ne üretiyor

Döngü üç dokümandan ikisine ulaşıyor. q19'da **3'te 0**'dan **3'te 2**'ye çıkıyor —
`fare_classic_shorthaul` ve `sop_misconnect_v4` geri geliyor, booking class K'nin fare basis'i
için iki ek arama turu hiçbir yeni şey getirmiyor ve `interline_xx_yy` hiç gelmiyor. Sonra yine
de yazdığı cevabı oku, çünkü finalin temiz olmaktan çıktığı yer burası.

Kayıtlı koşuda model yemek fişini ve değişiklik cezasını doğru veriyor, otel eşiğini ve yeniden
fiyatlandırma sorusunu bulamadığını söylemeden atlıyor ve kimsenin sormadığı, **booking class
M**'ye ait **EUR 90**'lık bir **iptal** cezası ekliyor. Senin koşun başka kelimelerle yazacak —
decompose üretildiği için günün determinist olmayan tek parçası burası — ama biçim tekrar eder:
döngünün bulamadığı doküman cevapta belirtilen bir boşluk değil, sessiz bir boşluk hâline geliyor.
Kelimeye değil, biçime bak.

Dokümanların gerçekte ne dediği, her biri geldiği paragrafla:

- **EUR 15** yemek fişi, çünkü bekleme üç saati aşıyor — `sop_misconnect_v4` 3.4
- **otel yok**, çünkü beş saat altıyı aşmıyor — `sop_misconnect_v4` 3.5
- Wyvern segmenti korunuyor ve **yeniden fiyatlandırılmadan** yerleştiriliyor —
  `interline_xx_yy` 4.1 ve 4.2
- isteğe bağlı değişiklik **EUR 70** — `fare_classic_shorthaul`, RULE 7 ve CLASSIC K satırı

Modelin kimsenin istemediği hâlde eklediğine dikkat et. **EUR 70**, masadaki gerçek soru olan
class K değişiklik cezası; **EUR 90** ise aynı satırda bir sütun öteki short-haul iptal cezası —
modül 7'nin tam olarak konusu olan karışıklık, ama bu sefer yanlış hücreyi kimse eline
tutuşturmadı, kendisi gönüllü oldu. Header'ın, getirdiği satıra hâlâ bağlı kalmasının sebebi
structure-aware chunking; bu yüzden değişiklik cezası doğru çıktı. Retrieval yine de bir doküman
eksik kaldı. Elindekiyle okuma da hâlâ temiz değildi.

Günün dürüst kapanışı bu ve finali olduğundan derli toplu göstermek yerine bunu yüksek sesle
söylemek gerekiyor. Bu kurstaki her metrik **retrieval**'ı ölçüyor — doğru dokümanın gelip
gelmediğini. Hiçbiri cevabın doğru olup olmadığını ölçmüyor. İki sistem, iki hata biçimi ve biz
sadece birini ölçtük. Önce retrieval eval'ini kur, çünkü ucuz ve determinist; sonra cevaplar için
ikincisini kur, çünkü birincisi ikincisinin bozuk olduğunu sana asla söylemeyecek.

**Ve döngü her zaman kazanmıyor.** q20'de üç dokümandan ikisine ulaştı — tek geçişin zaten
ulaştığı aynı ikisine. RULE 7'nin yolculuğun tümünün iptali için seçtiği sayfa,
`fare_classic_longhaul`, hiç gelmedi. İki soru bir yöntemi ölçemez. Sadece böyle bir yöntemin
var olduğunu ve bedavaya gelmediğini gösterebilir.

<div class="presenter-note">
Döngüyü çalıştırmadan önce salona modelin üreteceğini düşündükleri üç alt soruyu yazdır. Sonra
sadece decompose hücresini çalıştır ve karşılaştır. Kimseninkiyle birebir tutmayacak, ders de
bu: plan yazılmıyor, üretiliyor. Sonra cevap hücresini çalıştır ve corpus açıkken yüksek sesle
oku — alkışa yetişmek için kötü cevabın üstünden atlama. Ağzında gevelenmemesi gereken cümle:
<strong>retrieval artık modelden önce gelen bir adım değil, modelin çağırdığı bir tool.</strong>
Bir kez, net söyle; sonra sus.
</div>

## Fatura

**Daha çok çağrı.** Bir decompose, alt soru başına bir retrieval, tur başına bir yeterlilik
kontrolü, bir cevap: naive RAG'in bir çağrı yaptığı yerde altı ile on arası model çağrısı —
bunların üçü ya da dördü ucuz embedding çağrısı, gerisi generation. Ölçek için: modül 9'un
reranker'ı soru başına **8 model çağrısı** tutuyor, 20 soruluk bir geçiş için 160 çağrı, ve en
güçlü kurulumda o geçiş **71 saniye** sürdü; dört kurulumluk taramanın tamamı 640 çağrı ve 440
saniye. Döngü aynı mertebe, yeni bir kategori değil. Farklı olan şu: döngü çağrı başına gecikmeyi
toplamıyor, tur sayısıyla çarpıyor. Yani yavaş bir generation modeli sana biraz daha pahalıya
gelmiyor; tur sayısı katı pahalıya geliyor.

**Determinizm yok.** Plan üretildiği için aynı sorunun iki koşusu farklı bölünebilir, farklı
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
"20-30% ceza" uydurduğunu izlemiştik.

<div class="presenter-note">
Döngüyü canlı kır: yeterlilik prompt'unu asla YES diyemeyecek şekilde değiştir, tekrar çalıştır,
tur sayacının tavana çarpıp durduğunu izlet. On saniye — ve production'da ihtiyaç duyacakları
kısım tam olarak bu. Toplam 28 dakika: gate 4, tek geçiş hatası 5, döngü 10, cevabı okuma 5,
kırma 4. Bu modül "asla kesilmez" listesinde — eğitmenin sürmesinden gelen tasarruf zaten
varsayılan olarak harcanmış durumda, yani burada geri kazanılacak dakika yok. Aşağıdaki geri
sarma bu bütçede değil — 14:33'teki 15 dakikalık kapanış bloğuna ait.
</div>

## Ne çalıştırıyorsun

**Bunu izliyorsun.** Notebook'u eğitmen sürüyor; dosya repoda duruyor ve sonrasında kendi
laptopunda, diğer her modülle aynı iki bağımlılıkla koşuyor. İstersen canlı takip et — ama burada
salonun tek ekrana bakması daha verimli.

**VS Code, açık klasör repo kökü — `notebooks/07_agentic_rag.py`:** dosyayı aç, imleci ilk `# %%`
bloğunun içine koy ve dosya boyunca `Shift+Enter`'a bas. Ollama modül 0'dan beri zaten servis
veriyor — macOS'te uygulama, Windows'ta oturumunun arka plan servisi; elle başlatman gerekmedi.

- **ne görmen gerekiyor** — tek geçiş hücresi `of the 3 documents needed, retrieval found 0`
  yazıyor; döngüden sonra `final coverage:` üçte ikisini listeliyor — `interline_xx_yy` eksik
  kalıyor; sondaki karşılaştırma tablosu q19'u `0 → 2`, q20'yi `2 → 2` olarak basıyor
- **kabaca ne kadar sürüyor** — döngü hücrelerinin her biri saniyeler; kayıtlı koşuda sondaki iki
  soruluk karşılaştırma M serisi bir Mac'te q19 için 4.9 sn, q20 için 7.9 sn sürdü, sadece CPU'lu
  bir laptopta daha uzun

**Bir hücre çalışmazsa.** Sondaki karşılaştırma hücresinin kaydı var:
`notebooks/cached_runs.json` içinde `07-single-shot-vs-agentic` anahtarı — 2026-09-09, M serisi
bir Mac.

**Terminal (repo kökü):**

```bash
USE_CACHED=1 python notebooks/07_agentic_rag.py
```

`notebooks/_cached.py` o zaman o hücreyi modeli çağırmak yerine kayıttan oynatıyor ve tarihi ile
makineyi yazan bir `[CACHED]` bandı basıyor; yani bir replay hiçbir zaman canlı koşu diye
yutturulamıyor. Bu notebook'ta kaydı olan tek hücre bu: üstündeki decompose, yeterlilik ve cevap
hücreleri hâlâ modeli çağırıyor. Yani kayıttan oynatılan bir koşu sana sonucu veriyor — q19
`0 → 2`, q20 `2 → 2` — anlatımı ise bu sayfa veriyor.

<div class="presenter-note">
<strong>Yedek plan ve ne zaman devreye alınacağı.</strong> Burası günün en düşük enerjili slotu,
projeksiyonda yalnızca senin laptopun var ve arkasından gelen geri sarma salonun eve götürdüğü
kısım. Öğretmek yerine debug etmeye başladığın an — Ollama cevap vermiyor, bir model eksik, bir
hücre salonun sabrından uzun sürüyor — dur, <code>USE_CACHED=1</code> ayarla ve yeniden çalıştır.
Kayıtlı koşu kapanış tablosunu <code>[CACHED]</code> bandıyla ekrana koyuyor, sen de döngüyü bu
sayfadan anlatıyorsun. Bunun bir replay olduğunu yüksek sesle söyle; zaten bant ekranda. Finalde
salonun önünde canlı debug yapma.
</div>

Sadece döngünün kendisi — üstündeki corpus, chunking ve retriever hücreleri modül 5'ten beri
zaten koşan hücrelerin aynısı:

**VS Code — `notebooks/07_agentic_rag.py`, iki prompt hücresinden sonraki blok:**

```python
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

`decompose` ve `sufficient`, notebook'un içinde yazılmış iki prompt; `dense` ve `questions` ise
üstteki indeksleme hücresinden geliyor. Yani bu blok onlardan sonra çalışıyor, öncesinde değil.
Kendi koduna taşıman gereken bir kusur: `lstrip("NO")` bir önek değil bir *karakter kümesi*
siliyor; yani "NOT enough on hotels" diye başlayan bir takip sorgusu "T enough on hotels" olarak
geliyor. Notebook'ta da aynı şekilde duruyor. Öneki regex ile temizle.

## Sayılar ne dedi

<div class="measured">

| | ölçüm |
|---|---|
| gold set'teki `multi_hop` soru sayısı | 20'de 2 |
| q19'un gerektirdiği gold doküman | 3 — `sop_misconnect_v4`, `interline_xx_yy`, `fare_classic_shorthaul` |
| `multi_hop` hit@1, tam dokümanlar | 0.000 |
| `multi_hop` hit@1, merdivenin her basamağı | **0.500** |
| `multi_hop` hit@1, structure-aware chunk'larda üç embedder | **0.500** · **0.500** · **0.500** — MiniLM, `nomic-embed-text`, `bge-m3`, aynı şekilde |
| `multi_hop` hit@1, dense / BM25 / RRF | dense chunk'larda **0.500**, tam dokümanlarda 0.000; BM25 ve RRF her iki granülerlikte de 0.000 |
| döngünün üstünde koştuğu kurulum (`bge-m3` + structure-aware) | hit@1 **0.750**, MRR **0.817** |
| merdivenin en iyi basamağı (structure-aware + boilerplate temizlenmiş) | hit@1 **0.800**, MRR **0.841** |
| q19'da ulaşılan gold doküman — tek geçiş / döngü | **3'te 0** / **3'te 2**, 4.9 sn |
| q20'de ulaşılan gold doküman — tek geçiş / döngü | 3'te 2 / 3'te 2, 7.9 sn |
| bir rerank geçişinin maliyeti, en güçlü kurulum | soru başına 8 model çağrısı, kurulum başına 160 çağrı, 20 soruda 71 sn |
| yürürlükten kalkmış `sop_misconnect_v3` | EUR 10 fiş, 8 saatlik otel eşiği |
| güncel `sop_misconnect_v4` | EUR 15 fiş, 6 saatlik otel eşiği |

Merdiven, embedder, fusion ve rerank satırları `eval/RESULTS.md`'den; q19 ve q20 satırları
`notebooks/cached_runs.json` içindeki 2026-09-09 tarihli kayıtlı koşudan.

Döngünün kendisi **benchmark tablosunda yok**. İki multi-hop sorusu bir yöntemi ölçemez;
yalnızca böyle bir yöntemin var olduğunu gösterebilir. Bunu neyin çözeceği belli: elli etiketli
multi-hop sorusu, gold dokümanlardan birinin değil *hepsinin* getirilip getirilmediğine göre
skorlama ve retrieval'ı değil cevabı ölçen ikinci bir eval. O gelene kadar bu sayfanın iddiası
dar olan: q19'da tek geçiş üç dokümandan hiçbirini getirmedi, döngü üçünden ikisini getirdi;
q20'de döngü, tek geçişin zaten getirdiğini getirdi.

</div>

## Daha derine

Decompose, bütçesi olan bir query rewriting'dir; bugün ölçtüğümüz her şey her alt soru için hâlâ
geçerli — ama metni artık insan değil model yazıyor. Ortalamada daha iyi, çünkü modelin ifadesi
corpus'un kelime dağarcığına doğru kayıyor; kuyrukta daha kötü, çünkü "yanlış şeyi güzel sordun"
diye bir hata sinyali yok.

Yeterlilik kontrolü hem en ilginç hem en zayıf parça. 3B'lik bir modelden kendi kanıtının
eksiksizliğine hüküm vermesini istiyorsun ve modül 9'un reranker sonucu burada olduğu gibi
geçerli: 3B'lik bir modelin görüşü kendi tavanına düzlüyor. Çözüm hükmü daraltmak: "bu yeterli
mi?" değil, "getirilen metin isteğe bağlı değişiklik için bir euro tutarı söylüyor mu?" Alt
sorulardan türetilen bir checklist, bulanık bir kararı lookup'a çevirir. Pointwise'ın listwise'ı
yenmesiyle aynı ders, bir kat yukarıda.

Döngünün neye ihtiyaç duymadığına bak: planner framework yok, tool-calling API yok, agent
class'ı yok. `eval/retrieval.py`'den iki şey — `generate` ve `DenseRetriever` — artı
notebook'ta yazdığın üç prompt ve içinde break olan bir `for`. Çoğu agent framework'ü, bunun
üstüne retry, tracing ve şema eklenmiş hâlidir. Framework'ü davranışı elde etmek için değil,
tracing'e ihtiyacın olduğunda al.

10 milyon dokümanda döngü bedava olmaktan çıkar, çünkü her ek geçiş yeni bir tam ANN araması
demek. Fan-out'u sınırla ve alt soru seviyesinde cache'le — aynı üç alt soru binlerce misconnect
vakasında tekrar ediyor ve normalize edilmiş alt soruyla anahtarlanmış bir cache her model
değişikliğinden fazla kazandırır. Önüne bir router koy, çünkü soruların çoğu tek adımlık ve
decompose'un bedelini hiç ödememeli. Tool'ları açık ve tipli yap — `search_sop`,
`search_fare_rules`, `search_interline`, her biri kendi filtresiyle. v3'ü dışarıda tutan sürüm
filtresi oraya ait, sıralamaya dair bir umuda değil tool'un bir özelliğine; RULE 7 de öyle: bir
change mi yoksa bir cancellation mı fiyatladığını bilen bir tool, sayfayı retrieval'dan önce
seçer — 3B'lik bir modelden iki dokümanın da sonuna gömülü bir kuralı fark etmesini istemek
yerine.

Dürüst sınır: hiç yazılmamış bir dokümanı hiçbir geçiş sayısı üretemez. Interline partnerin
kendi SOP'u bizimkiyle çelişiyorsa ve bunu çözen bir paragraf yoksa, döngünün hata biçimi var
olmayan metni aramaya devam etmektir. Sınırla ve bunu söylet.

## Günü geri sarmak

Burası kapanış bloğu, modül 10'un 28 dakikasının parçası değil. Zinciri tersten, yüksek sesle,
tek nefeste oku.

Model bizim verimizi bilmiyordu ve bilmediğini de bilmiyordu — CLASSIC K iptal cezası
sorulduğunda "20-30% ceza" uydurdu. Sonra training'in ne olduğuna baktık ve weight'lerin donmuş
bir fotoğraf olduğunu gördük. Bir fine-tune kurduk — Q2 cevabını ezberleten 695 üretilmiş çift —
sonra Q2 Q3 oldu ve o satır **EUR 90**'a döndü. Yanlış cevap verdiğini izleyemedik:
`UNVERIFIED: fine-tune edilmiş modelin bayat cevabı — kraken-q2 hiç kurulmadı ve ona ait hiçbir
koşu yok.` Modül 3'ün model olmadan gösterdiği şey sorunun biçimi: weight'lerin bir satırın
değiştiğini bilmesinin de, öğrendikleri satırı kaynak göstermesinin de yolu yok. Bütün kural
kitabını prompt'a koyduk, sığdı, ve her sorguda 79 309 karakterin tamamını ödedik. Seçmeye
başladık — retrieval çöp getirdi. Elimize tutuşturulan embedder'ı kontrol ettik ve her
framework'ün varsayılan olarak seçtiği modelin, altı Türkçe soru/İngilizce doküman çiftinin
hepsinde **0.000** aldığını gördük. Chunker'ı değiştirdik ve hatanın satır değil header olduğunu
bulduk. Gerçek olsun diye ChromaDB'ye koyduk — ve iki satır kodun içinde sessiz varsayılan
embedder'ının iki soruyu yanlış cevapladığını izledik. Hybrid ve rerank ekledik ve bir
reranker'ın kendi tavanına düzlediğini ölçtük: upgrade değil, takas. Sonra iki soru aynı anda üç
doküman istedi ve `multi_hop` satırı chunklanmış her basamakta **0.500**, sadece chunklamadan
önce **0.000** okundu — q19'da tek bir geçişte ihtiyaç duyduğu üç dokümandan hiçbiri geri
gelmedi bile.

Dokuz gate ve bir kurulum, hiçbiri tanımla açılmadı. Her biri bir önceki modülün çarptığı
duvardı. Bütün gün tek bir sayı: **EUR 90** — bir dokümandan alıntılanmış ve dokümanın adı
verilmiş hâliyle. Bir de tek bir çekince: dokümanın gelip gelmediğini ölçtük, üstüne kurulan
cümlenin doğru olup olmadığını hiç ölçmedik.

<div class="presenter-note">
Geri sarmayı ayakta yap; slayt yok, laptop yok. Günün son bölümü ve bir meslektaşına
anlatacakları tek kısım burası. Yuvarlamadan, dürüst söylemen gereken iki şey var: fine-tune
cümlesi bir tahmin, bu salonun izlediği bir şey değil — <code>kraken-q2</code> hiç kurulmadı — ve
merdivenin basamakları arasındaki farklar bir soru genişliğinde, yani rakamı değil yönü aktar.
EUR 90 cümlesiyle bitir, sonra çıkış cümlesi, sonra konuşmayı kes. Arkasına özet slaytı koyma.
Burası 14:33'teki 15 dakikalık kapanış bloğu; repo linki ve handout da burada veriliyor.
</div>

## Çıkış cümlesi

> Bugünkü her adım, bir öncekinin yetmediği yerde doğdu. Sonuncusu RAG'i bir agent'a
> dönüştürdü — agent günü de tam buradan devralıyor.
