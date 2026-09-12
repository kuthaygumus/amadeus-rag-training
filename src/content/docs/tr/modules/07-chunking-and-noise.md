---
title: "7. Chunking ve Gürültü"
description: "Retriever yanlış dilimi getirdi. Modeli değil, kesimi düzelt."
---

## Gate sorusu

> **Doğru doküman geldi, cevap yine yanlıştı. Neyi değiştiriyoruz — modeli mi, kesimi mi?**

[Modül 6](/tr/modules/06-chromadb/) chunk'lar ekrandayken bitti. `fare_classic_shorthaul.md`
birinci sıradaydı — isabet — ve K satırını tutan chunk, sütunları isimlendiren satırı
tutmuyordu. Model `EUR 70 | EUR 90 | EUR 180` okudu, hangi sütun ne bilmeden yanlış olandan cevap
verdi. 12 Eylül koşumunda fixed-280 index'i "EUR 13" üretti; senin sayın farklı çıkacak, kendinden
eminliği çıkmayacak.

Splitter'ı salonda kimse yazmadı. Corpus ile prompt arasında duran ve default'ta bırakılmış tek
kod parçası o. **Satır kurtuldu; header onunla birlikte gelmedi.** Bu bir dilimleme hatası ve
dilimleme string kodu — pipeline'da değiştirmesi en ucuz şey.

<div class="presenter-note">
Bruno'ya dokunmadan önce oylama iste: "ne bozuk — model mi, embedder mı, başka bir şey mi?"
"Model küçük" ve "daha iyi embedder" gelecek. İkisini de tahtaya yaz; ikisi de cevap değil ve
önümüzdeki 30 dakikanın esprisi şu: emindiler. Üç stratejiyi henüz açıklama — ilk sürpriz,
1. adımdaki <code>chunks</code> sayısı olsun.
</div>

## Kesmenin üç yolu, en ucuzundan başlayarak

Üçü de `amadeus-rag-lab` içindeki `src/chunking.ts` dosyasında duruyor; `/ingest` birini adıyla
seçiyor.

**`fixed`** — her N karakterde kes, içinde ne olduğuna bakma. Default 280.
[Modül 5](/tr/modules/05-simple-rag/) bununla ingest etti ve kötü adam bu: tablo satırı nadiren
ortadan bölünür, ama header satırı bir önceki chunk'a düşer ve onunla birlikte hiç getirilmez.

**`recursive`** — sığan en doğal sınırdan kes, sığmazsa daha incesine düş: başlıklar, sonra
paragraflar, satırlar, cümleler, boşluklar. Default 600. Düz metin sağ kalır. Pencereden uzun bir
tablo kalmaz — dokuz sütunluk header artı yedi satır 600 karakterden geniş, yani separator
mantığı ne derse desin kesik tablonun içine düşer. Her framework'ün default'u bu ve salonun çoğu
zaten production'da bunu kullanıyor.

**`structure`** — dokümanın kendi başlıklarından kes: `#` satırları, `RULE n.`, `SECTION n`,
`Step n.`. Kural, tanıttığı tabloyla birlikte kalır. Sonra her chunk'ın başına nereden geldiği
yazılır — `[doküman adı > başlık]`; doküman adı, alt çizgileri boşluğa çevrilmiş dosya adı. Çıplak
bir sayı ızgarası etiketiyle gelir:

```
[fare classic shorthaul > RULE 2A. Reading the schedule. Each booking class carries its own fare basis code. The change]
penalty and the cancellation penalty are distinct amounts and must not be substituted for one
another; the no-show penalty is derived from the cancellation penalty under RULE 4 below.

| Booking class | Fare basis | Advance purchase | Minimum stay | Change penalty | Cancellation penalty | ...
```

Prefix, başlık satırının kendisi — `RULE 2A` için bu, paragrafın ilk satırının tamamı, çünkü bu
corpus'ta kuralın başlığı ile ilk cümlesi aynı satırı paylaşıyor. Maliyeti bir string birleştirme.
"Contextual retrieval" diye satılan şeyin tamamı da bu.

**Neden 900 değil 1500.** `RULE 2A` bölümü — düz metin, header, yedi satır, ortada kalan
`Page 3 of 7` — 1 140 karakter. Structure chunk boyutu 900'de bölüm ikiye kesiliyor, tablo
cümlesini geride bırakıyor ve tablo chunk'ı günün sorusu için ilk 10'a girmiyor. 1500'de bölüm
tek chunk ve sıralanıyor. Default'un 1500 olmasının tek sebebi bu; ölçüldü, tahmin edilmedi.

## Overlap: kaybeden güvenli default

Overlap, emin olmadığında yaptığın hamle diye satılır — biraz fazlalık nasıl zarar versin.
fixed-280 üstüne 60 karakter overlap, Q3 corpus'unu 294 chunk'tan 368'e çıkarıyor ve ücret
sayfasında yaptığı tek şey sınırı kaydırmak: header hâlâ K satırından önceki bir chunk'ta. Overlap
bir cümleyi ortadan bölmeye karşı sigortadır. Yüzlerce karakter ötedeki bir referansa karşı hiçbir
şey yapmaz, ve o 368 birbirine benzer parça top-K için birbiriyle yarışır. Trainer tarafındaki
eval'de her metrikte kaybediyor. Sayılar aşağıdaki kutuda.

## Gürültü kozmetik değil

Corpus, gerçek bir export'un taşıdığını bilerek taşıyor. `corpus/2026-Q3/fare_classic_shorthaul.md`
dosyasını aç ve sonunu oku: bir `<div class="legal">` bloğu — aynı LEGAL NOTICE her ücret kuralı
sayfasını kapatıyor — HTML'den kalan bir `&nbsp;`, ve tablo ile `RULE 3` arasında kendi satırında
kalmış bir `Page 3 of 7`. 28 dokümanın altısı aynı legal blokla kapanıyor.

`src/chunking.ts` içindeki `stripBoilerplate` tam olarak bu tür şeyleri chunking'den önce siliyor:
legal blok, `LEGAL NOTICE` ve `DISCLAIMER:` paragrafları, `Page n of n` satırları, artık `<div>`
`<br>` `<span>` `<p>` etiketleri ve `&nbsp;`, bir de `Distribution:` footer'ı. Sonra geride
bıraktığı boş satır dizilerini topluyor.

Neden chunking'den önce: index'e sızan boilerplate, tek içeriği diğer bütün sayfalarla ortak olan
chunk'lar üretir. Altı neredeyse aynı legal kuyruk üç top-K yeri için yarışır ve hiçbiri hiçbir
şeye cevap vermez. Bu corpus'ta structure chunk sayısı yerinden oynamıyor (132 → 132) çünkü legal
bloğun kendi başlığı yok — son kuralın chunk'ının kuyruğunda gidiyor — yani temizlik chunk silmiyor,
kısaltıyor. Vektörler yine de kayıyor: bir `RULE 7` chunk'ı üçte bir oranında legal notice olmaktan
çıkıyor.

## Ne çalıştırıyorsun

Aynı 28 doküman, aynı `bge-m3`, aynı `gemma3:4b`, aynı Chroma. Yalnızca chunk'lar değişiyor.

**Bruno — `05-chunking-and-noise` › `1-reingest-structure`**

```json
{
  "edition": "2026-Q3",
  "strategy": "structure",
  "stripBoilerplate": true
}
```

`chunks` alanını oku: **132**, `02-ingest`'in kurduğu 294'e karşı. `collection` alanını oku:
`kraken-2026-Q3-structure-1500-strip` — her konfigürasyona bir collection, yani geri dönmek
istersen fixed-280 index'i hâlâ yerinde. `stages.embed` alanını oku: modülün tek yavaş isteği bu
ve sebebi de orada. `sample`, ilk üç chunk'ı `[ad > başlık]` prefix'iyle gösteriyor.

**Bruno — `05-chunking-and-noise` › `2-peek-again`**

`GET /chunks?source=fare_classic_shorthaul&limit=20`. `chunks[].text` içinde `| K |` satırını bul.
Bu kez header satırı aynı chunk'ta ve chunk `[fare classic shorthaul > RULE 2A …]` ile başlıyor.
`chars` değerini `02-ingest › peek-chunks`'ın gösterdiğiyle karşılaştır: daha az, daha uzun parça.

**Bruno — `05-chunking-and-noise` › `3-query-again`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
}
```

`answer` alanını oku: EUR 90, `[fare_classic_shorthaul]` kaynağıyla. `sources[]` alanını oku:
tablo chunk'ı artık aralarında — `excerpt` prefix'le başlıyor. Modele header ile satırın bugün ilk
kez birlikte verildiğini görmek istersen `debug.prompt`'a bak.

**Model hakkında hiçbir şey iyileşmedi. Bunu bir model değil, bir splitter kazandı.**

<div class="presenter-note">
Otuz dakika, beşi canlı istek: 1 yavaş olan (132 chunk embed ediyor — <code>stages.embed</code>
değerini yüksek sesle oku), 2 ve 3 saniyelik. 2. adımda <code>| K |</code> chunk'ını göstermeden
önce sor: "modül 6'da satır kurtulmuştu — o zaman cevap neden yanlıştı?" On beş saniye sessizlik,
sonra biri "header" diyecek. Katılımcıdan gelince on kat daha sert oturuyor. Ollama kapalıysa
<code>/ingest</code> embed edemez ve modülün canlı ayağı kalmaz — sayfayı oku, sonra kendi
makinende repo kökünde <code>npm test</code> çalıştır: <code>test/chunking.test.ts</code> model
istemiyor ve tam olarak bunu assert ediyor — fixed testi K satırı chunk'ında "Cancellation penalty"
<em>olmadığını</em>, structure testi oyuncak sayfasındaki
<code>[fare classic shorthaul &gt; ## RULE 3 …]</code> prefix'ini kontrol ediyor.
</div>

### Dene

**Recursive.** 1. adıma `"strategy": "recursive"` koy (default boyut 600, 197 chunk), sonra 2 ve
3'ü tekrar çalıştır. Düz metin chunk'ları temiz görünüyor. K satırı ile header'ı hâlâ ayrı
chunk'larda, çünkü tablo pencereden geniş — ve bunu hiçbir separator listesi düzeltmiyor.

**topK 5.** 3. adıma `"topK": 5` ekle ve `sources[]`'i oku. `fare_classic_longhaul` sızıyor mu?
Long-haul CLASSIC K, EUR 195 — biraz farklı bir sorunun *doğru* cevabı olan bir distractor. Chat
modelinin ekmeğini kazandığı yer burası: iki tablo da context'teyken `gemma3:4b` 90'ı seçiyor. Bu
kursun 12 Eylül'e kadar kullandığı model `qwen2.5:3b`, denediğimiz her prompt varyantında 195'i
seçti; gitme sebebi bu. Retrieval modelin ne göreceğine karar verir; modelin onu yine de okuması
gerekir.

## Sayılar ne dedi

<div class="measured">

Yalnızca trainer tarafı — `amadeus-rag-lab` repo'sunda `npm run eval`; katılımcıların
çalıştırdığı bir şey değil. 20 gold soru, edisyon 2026-Q3, embedder `bge-m3`, **doküman**
seviyesinde skorlanıyor (bir dokümanın en iyi chunk'ı o doküman için isabet sayılır). M-series
Mac'te birer koşum, 12 Eylül 2026; sayılar başka makinelerde kayar.

| collection | chunk | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| fixed-280 | 294 | 0.700 | **0.917** | **0.817** |
| fixed-280 + overlap 60 | 368 | **0.550** | 0.883 | 0.717 |
| recursive-600 | 197 | 0.700 | 0.900 | 0.806 |
| structure-1500 | 132 | 0.700 | 0.833 | 0.781 |
| structure-1500 + strip | 132 | 0.650 | 0.850 | 0.766 |

Dürüst oku. **Structure-aware chunking doğru dokümanı daha sık bulmuyor** — beş satırın üçünde
hit@1 0.700 ve temizlenmiş index bir soru aşağıda. Düzelttiği şey **chunk**: K satırı header'ıyla
birlikte, ki kendinden emin yanlış tutarı EUR 90'a çeviren şey bu. Doküman seviyesindeki bir skor
bunu göremez; 2. adım görür.

Tablonun söylediği iki şey var. Overlap her metrikte aynı anda kaybediyor ve bunun için 74 chunk
fazladan ödüyor. Ve en iyi recall@5, en parçalanmış index'e ait: daha çok küçük parça, gold
dokümana ilk beşte görünmek için daha çok şans verirken birinci sıraya koymayı zorlaştırır.
recall@5'i seçersen kör chunking kazanır; hit@1'i seçersen aralarında fark yok. İkisi de dürüst.

0.05'lik bir adım yirmide bir soru. Rakamı değil, yönü oku.

</div>

## Daha derine

Retriever bir dokümanı hiç görmez. Chunk başına 1024 boyutlu tek bir nokta görür. Bir header'ı,
yedi satırı ve bir sayfa numarasını tek noktada ortalarsan, short-haul ücretleri hakkında her şeye
yakın ama hiçbir şey hakkında spesifik olmayan bir şey elde edersin. Uzun chunk generator'a daha
çok bağlam, retriever'a daha bulanık bir vektör demek; structure-aware bölme kazanıyor çünkü bir
bölüm hem anlamın hem embedding'in doğal birimi.

Başlık prefix'i iki iş yapıyor. Vektör kayıyor, çünkü "fare classic shorthaul" ve "Reading the
schedule", ne işe yaradığını söylemeyen bir ızgaraya katılıyor. Ve generator'ın girdisi iyileşiyor,
çünkü metin nereden geldiğini kendisi söylüyor. Yayımlanmış contextual-retrieval çalışmaları o
cümleyi chunk başına bir LLM ile üretiyor; biz etkinin çoğunu dosyada zaten duran bir başlıktan
alıyoruz. Dokümanlarında ödünç alınacak yapı yoksa üret — taranmış PDF'ler, chat log'ları, ticket
dump'ları. Varsa ödünç al.

Ölçek büyüdüğünde tabloları chunk'lamayı tamamen bırakırsın. Dosyalanmış bir tarife için kalıcı
çözüm farklı bir temsil: tabloyu bir kez parse et, header'ı içine açılmış şekilde satır başına bir
chunk üret — `CLASSIC short-haul, booking class K, fare basis KSHEU26, change penalty EUR 70,
cancellation penalty EUR 90` — ve yanlış cevabı üreten belirsizlik artık oluşamaz. Bu, doküman
tipi başına yapılan bir iş ve chunking'i versiyonlanan bir offline işe çevirir: splitter'ı
değiştirdiğinde corpus'u yeniden embed edersin. Lab bunu zaten kodluyor —
`kraken-2026-Q3-structure-1500-strip` collection adı, içindeki her vektörü hangi splitter'ın
ürettiğini söylüyor.

<div class="presenter-note">
Biri "biz RecursiveCharacterTextSplitter kullanıyoruz, gayet iyi" derse — katıl, sonra recursive
satırını göster: hit@1 0.700, iyi bir default; ve recursive altında 2. adım header'ı K satırından
hâlâ ayırıyor. Düz metin için iyi bir default, tablolar için çözüm değil. Framework tartışmasına
dönmesine izin verme: 2. adımdaki peek benchmark da model de istemiyor, ve bu modülün yirmi soruya
bağlı olmayan kısmı orası.
</div>

## Çıkış cümlesi

> Doğru chunk, doğru cevap — EUR 90, ve tek bir weight değişmedi. Ama modül 3'ün fine-tune edilmiş
> modeli de K sınıfı için bir iptal cezası biliyor. Şimdi RAG'in onu asıl önemli yerde yendiğini
> kanıtla: [veri değişiyor](/tr/modules/08-freshness/).
