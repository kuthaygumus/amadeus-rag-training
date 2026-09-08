---
title: "7. Chunking, Gürültü ve Ölçüm"
description: "Retrieval neden çöp getirdi?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Retrieval neden çöp getirdi?**

Modül 5'ten çalışan bir pipeline ve çalışmayan cevaplarla çıktık. Retriever konu olarak doğru,
işe yaramaz metinler getirdi: bir legal footer, yarım bir revizyon geçmişi, hangi sütunun ne
olduğunu söylemeyen bir euro tutarları tablosu. Embedder'a, modele, prompt'a kimse dokunmamıştı.
Corpus ile retriever arasında duran tek şey, kimsenin tartışmadığı kod parçasıydı: splitter.

<div class="presenter-note">
Hiçbir şeyi çalıştırmadan önce K sınıfı sorusunu tekrar ekrana al ve salondan "sence ne bozuk?"
oylaması iste. "Model küçük" ve "daha iyi embedder lazım" gelecek. İkisini de tahtaya yaz.
İkisi de cevap değil ve önümüzdeki 35 dakikanın esprisi tam olarak bu: emindiler. Merdiven
tablosunu henüz gösterme.
</div>

## Merdiven

Aynı 28 doküman, aynı `bge-m3` embedding'leri, aynı 20 soru, doküman seviyesinde skorlanıyor.
Tek değişken metnin nasıl kesildiği. Bütün dokümanlar: hit@1 **0.550**. Körlemesine 280
karakterlik chunk'lar: 0.700. 60 karakter overlap ekle: 0.700. Herkesin ilk uzandığı recursive
splitter: 0.700. Dokümanın kendi başlıklarından böl: **0.800**, MRR 0.654 → **0.844**.

Merdivenin ortasına bir daha bak. Üç ayrı naive strateji — sabit boyut, overlap'li sabit boyut ve
recursive splitter — tam olarak 0.700'de duruyor. Daha akıllı bir naive splitter aramanın burada
getirisi yok. Bu barajı yalnızca dokümanın kendi yapısından kesmek aşıyor.

Merdivenin altından üstüne hit@1'de %45 göreli kazanç: hiçbir modele dokunmadan, sorgu anında
hiçbir maliyet eklemeden, `eval/chunking.py` içindeki üç splitter ile. Günün en büyük kaldıracı bu
değil — aynı structure-aware chunk'larda modül 6'daki embedder değişimi 0.450 değerindeydi
(`nomic-embed-text` 0.350'ye karşı `bge-m3` 0.800), chunking ise 0.250. Ama insanların default'ta
bıraktığı ayar bu.

## Hata satırda değil, header'da

Herkes sabit boyutlu chunking'in tablo satırını ortadan ikiye böldüğünü sanıyor. Bölmüyor. İşte
`fare_classic_shorthaul.md` dosyasının 280 karakterlik kör bölmedeki 6. chunk'ı — bu sayfadaki
chunk numaraları 0'dan başlıyor, `enumerate`'in yazdırdığı gibi:

```
 | none | none | EUR 65 | EUR 85 | EUR 170 | 1 x 23 kg | Yes |
| K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |
| M | MSHEU26 | none | none | EUR 90 | EUR 120 | EUR 240 | 2 x 23 kg | Yes |

Page 3 of 7

RULE 3. Involuntary cases. Where the cancellation
```

Satır sapasağlam. Cevabın ihtiyaç duyduğu her sayı orada. Eksik olan, sütunları isimlendiren
satır: `| Booking class | Fare basis | ... | Change penalty | Cancellation penalty | ...` — o da
iki chunk öncesine, **4. chunk'a** düşmüş ve hiç getirilmiyor.

Model bir euro tutarları ızgarası okuyup tahmin ediyor. CLASSIC K iptal cezası soruldu — doğrusu
**EUR 90** — cevabı **EUR 65** veriyor. Notebook'un cache'indeki değer bu, üç tekrar çalıştırmada
da aynısı çıktı. EUR 65, bir üstteki satırın değişim cezası; o satır da W sınıfı ve sınıf adıyla
fare basis kodu chunk sınırında kesilmiş durumda. Yanlış sütun, yanlış satır. Çekinmiyor da,
çünkü oturduğu yerden ortada hiçbir belirsizlik yok: bir sayı ızgarası var ve içlerinden biri
cevap.

<div class="presenter-note">
Sınırı göstermeden önce yüksek sesle sor: "satır bozulmamış — o zaman cevap neden yanlış?"
On beş saniye bekle. Biri bulacak ve katılımcıdan gelince on kat daha sert oturuyor. Ağzında
dolanmaması gereken cümle: <strong>satır kurtuldu, header onunla birlikte gelmedi.</strong>
Model EUR 65 dışında bir şey derse üstüne gitme — hangi yanlış sayı gelirse gelsin nokta aynı
noktada duruyor: hiçbir zaman "ayırt edemiyorum" demiyor.
</div>

**Overlap bunu düzeltmiyor.** 60 karakterlik overlap corpus genelinde 74 chunk ekliyor
(294 → 368) ve sınırı kaydırıyor. Ücret sayfasında header satırının tamamı artık 5. chunk'ta, ilk
iki sütun adı 4. chunk'ın kuyruğuna kopyalanmış durumda, K satırı ise 7. chunk'ta. Hâlâ iki chunk
arayla. Overlap bir cümleyi ortadan bölmeye karşı sigortadır; 612 karakter öteye yapılan bir
referansa karşı hiçbir şey yapmaz.

**Recursive splitting de düzeltmiyor.** Rahatsız edici olan bu, çünkü her framework'ün
default'u ve salondaki çoğu kişinin production'da kullandığı şey. Önce paragraflardan, sonra
satırlardan, sonra cümlelerden bölüyor ve düz metni gerçekten güzel koruyor — ama header 4.
chunk'ta, K satırı 5. chunk'ta. Yan yana, yine de ayrı. Sebebi ayarla kapatılamaz: tablo split
boyutundan geniş. 600 karakterlik bir pencere dokuz sütunluk header'ı artı yedi satırı
alamıyor; separator mantığı ne derse desin kesik tablonun içine düşüyor.

**Structure-aware bölme düzeltiyor.** Dokümanın kendi sınırlarından kes — `##` başlıkları,
`RULE n.`, `SECTION n` — tablo kendisini tanıtan kuralla birlikte kalıyor: header da K satırı da
4. chunk'ta. Sonra her chunk'ın başına geldiği bölümün başlığı yazılıyor; böylece çıplak bir sayı
ızgarası olacak parça etiketiyle geliyor:

```
[RULE 2A. Reading the schedule. Each booking class carries its own fare basis code. The change]
| Booking class | Fare basis | Advance purchase | ... | Change penalty | Cancellation penalty | ...
```

Prefix bir yol değil, başlık satırının kendisi: `structure_aware()` doküman adına yalnızca ilk
başlıktan önceki metin için düşüyor, doküman adını her chunk'ta tekrarlamıyor. "Contextual
retrieval" diye satılan şeyin tamamı bu: her parçaya nereden geldiğini söylemek. Maliyeti bir
string birleştirme.

## Tek sayı raporlamamak için iki sebep

**Fixed-280 ortalamayı yükseltirken bir kategoriyi geriletiyor.** Exact-token sorguları — uçuş
kodları, bülten id'leri — bütün dokümanlarda **1.000**, fixed-280'de **0.750**, overlap eklendiğinde
de 0.750'de kalıyor. Recursive-600 ve structure-aware ikisi de 1.000'e geri getiriyor. Manşet
hit@1 0.550 → 0.700 çıkarken altında dört exact-token sorusundan biri ters yöne gitti. Bir soru
kaybetmek küçük bir şey. Asıl mesele şu: manşet sayıda bunun olduğunu söyleyen hiçbir şey yok;
sadece tip kırılımı gösteriyor.

**Recall@5 en kötü stratejide en yüksek.** Fixed-280 **0.950**, structure-aware 0.833. Tesadüf
değil, aritmetik: 294 küçük parça, gold dokümanın ilk beşte bir yerde görünmesi için daha çok
şans verir, ama birinci sıraya koymayı zorlaştırır. recall@5'i seçersen kör chunking kazanır;
hit@1'i seçersen 0.100 farkla kaybeder. İkisi de dürüst. Aralarındaki seçim, ölçüm mü yoksa
reklam mı yaptığına karar verdiğin yerdir.

## Gürültü kozmetik değil, ölçülebilir

Corpus, gerçek bir export'un taşıdığı pisliği bilerek taşıyor. 28 dokümanın **23**'ünde
`Page 3 of 7` türünden bir sayfa numarası artığı var — bunların **18**'i tablo ortasında kendi
satırında duruyor, ki `strip_boilerplate()` tam olarak bu biçimi yakalıyor. **16**'sında artık
HTML var (`<br>`, `&nbsp;`, `<div class="legal">`), ve altı ücret kuralı sayfası aynı 431
karakterlik legal footer'la, bayt bayt aynı şekilde kapanıyor. Kendi içinde bir gövde paragrafını
tekrarlayan tam olarak bir doküman var: `macro_en_refund.md`, üstelik tekrarlanan cümle "iptal
sütununu oku, değişim sütununu değil" talimatının ta kendisi. İki doküman daha bir sayfa
footer'ını tekrarlıyor; onlar yukarıdaki artıklar arasında zaten sayıldı.

`eval/chunking.py` içindeki `strip_boilerplate()` legal blokları, sayfa artıklarını ve artık
markup'ı siliyor; corpus karakterlerinin **%4.2**'sini kaybediyor — 78,310'dan 75,037'ye. Küçük
bir kesinti, ama metriği oynatıyor:

| | chunk | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| structure-aware, ham | 154 | 0.800 | 0.833 | 0.844 |
| structure-aware, temizlenmiş | 153 | **0.850** | 0.833 | **0.869** |

Structure-aware index'ten bir chunk siliniyor. Kör stratejiden on dört tane (294 → 280 — model
gerektirmeyen bir string sayımı), çünkü structure-aware boilerplate'i zaten kendi bölümüne
ayırmıştı, her chunk'a bulaştırmak yerine. Temizlenmiş corpus'u yalnızca structure-aware ile
skorladık, o yüzden bu sayfada temizlenmiş fixed-280 satırı yok.

Bu fark ölçüldü ve tek değişken temizlikti: hit@1'de **+0.050** — yirmi sorudan biri — MRR'de
**+0.025**, recall@5 değişmedi. Yirmi soru 0.050'yi gürültüden ayıramaz; yönü makul, büyüklüğü
kanıtlanmamış olarak oku. Bunu neyin çözeceği belli: 0.05'in anlam taşıyacağı büyüklükte bir gold
set üzerinde, tek değişken temizlik olacak şekilde aynı benchmark.

## Ne çalıştırıyorsun

Bu modülde mekanizma ile ölçüm iki ayrı artefakt.

**Mekanizma.** `notebooks/05_chunking_and_noise.py` dosyasını VS Code'da Microsoft Python
eklentisiyle aç ve blokları Shift+Enter ile çalıştır. İki model çağrısı dışında her şey string
kesme işlemi, yani yavaş bir laptopta takılmaz. En çok işe yarayan hücre hiç model istemiyor: her
strateji için hangi chunk'ta sütun header'ı, hangi chunk'ta K satırı var, yazdır.

```python
import chunking as C

pieces = C.fixed(sheet, size=280, overlap=0)
for n, piece in enumerate(pieces):
    if "| K |" in piece and "EUR 90" in piece:
        print(f"chunk {n}: THE K ROW")
    if "Booking class" in piece and "Cancellation penalty" in piece:
        print(f"chunk {n}: THE COLUMN HEADER")
```

*Görmen gereken:* `chunk 4: THE COLUMN HEADER` ve `chunk 6: THE K ROW`. İki chunk arayla; ve
yalnızca K satırı chunk'ından üretilen cevap EUR 65 çıkıyor, oysa doğru cevap EUR 90.
*Yaklaşık süre:* notebook'un yazıldığı tempoda on iki dakika.

**Ölçüm.** Tek komut, tek tablo:

```bash
python exercises/m7_chunking_ladder.py
```

*Görmen gereken:* beş merdiven satırı, artı temizlenmiş corpus için altıncı bir satır; her birinde
chunk sayısı, hit@1, recall@5, MRR ve beş soru tipi. Sonunda
`Stripping the legal footer removed 4.2% of the characters.` satırı. Durup bakılacak satır şu:
`exact_token went 1.000 -> 0.750 while hit@1 went 0.550 -> 0.700`. *Yaklaşık süre:* iki-üç dakika,
neredeyse tamamı embedding çağrısı.

Corpus genelindeki chunk sayılarını notebook'ta da istersen:

```python
for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    ids, texts, parents = C.chunk_corpus(documents, strategy)
    print(strategy, len(texts))       # 294 · 368 · 197 · 154
```

<div class="presenter-note">
35 dakika: merdiven 6, header demo 12, overlap ve recursive 6, metrik uyuşmazlığı 6, gürültü 5.
Merdiven komutunu header demosuna başlamadan önce çalıştırmaya bırak — iki-üç dakika sürüyor ve
demo o süreyi kapatıyor. Birinin laptopunda patlarsa sayıları bu sayfadan oku ve sınır hücresini
canlı çalıştır: string kesme işlemi, patlaması mümkün değil. Sınır hücresi demo, merdiven kanıt.
Gün sarktığında kesilecek ölçüm bu değil.
</div>

## Sayılar ne dedi

<div class="measured">

| strateji | chunk | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| bütün dokümanlar | 28 | 0.550 | 0.717 | 0.654 |
| fixed 280 | 294 | 0.700 | **0.950** | 0.814 |
| fixed 280 + overlap 60 | 368 | 0.700 | 0.883 | 0.799 |
| recursive 600 | 197 | 0.700 | 0.900 | 0.816 |
| **structure-aware 900** | 154 | **0.800** | 0.833 | **0.844** |

Aynı merdiven, soru tipine göre hit@1:

| strateji | tr_tr | tr_en | en_en | exact_token | multi_hop |
|---|---|---|---|---|---|
| bütün dokümanlar | 1.000 | 0.333 | 0.250 | 1.000 | 0.000 |
| fixed 280 | 0.750 | 0.667 | 0.750 | 0.750 | 0.500 |
| fixed 280 + overlap 60 | 1.000 | 0.667 | 0.500 | 0.750 | 0.500 |
| recursive 600 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 |
| structure-aware 900 | 1.000 | 0.667 | 0.750 | 1.000 | 0.500 |

`fare_classic_shorthaul.md` içinde header nereye düşüyor — 0-tabanlı, model gerekmeden ölçüldü:

| strateji | o dosyadaki chunk | header | K satırı | birlikte mi? |
|---|---|---|---|---|
| bütün dokümanlar | 1 | — | — | evet — hiçbir şey kesilmedi |
| fixed 280 | 15 | 4 | 6 | hayır |
| fixed 280 + overlap 60 | 18 | 5 | 7 | hayır — overlap sınırı kaydırıyor |
| recursive 600 | 10 | 4 | 5 | hayır — tablo split boyutundan geniş |
| structure-aware 900 | 10 | 4 | 4 | evet |

</div>

Yirmi soru bir benchmark değildir. İki tasarım arasında seçim yapmaya yeter, yayınlamaya hiç
yetmez; yaklaşık 0.05'in altındaki fark bu örneklemin gürültüsünün içindedir.

## Daha derine

Retriever bir dokümanı hiç görmez. Chunk başına tek bir vektör görür — bütün bir bölüm için tek
bir nokta. Dokuz sütunluk bir header'ı, yedi ücret satırını ve bir sayfa numarası artığını tek bir
1024 boyutlu noktada ortalarsan, kısa mesafe ücretleri hakkında her şeye yakın ama hiçbir şey
hakkında spesifik olmayan bir şey elde edersin. Yani chunk boyutu, model kalitesinden bağımsız
olarak hassasiyetle takas edilir: uzun chunk generator'a daha çok bağlam, retriever'a daha bulanık
bir vektör verir. Structure-aware bölme kazanıyor çünkü bu iki baskıyı aynı yöne çeviriyor — bir
bölüm hem anlamın hem embedding'in doğal birimi.

Başlık prefix'i iki iş yapıyor. Vektör kayıyor, çünkü başlığın kendi kelimeleri — "Reading the
schedule", "fare basis code", "change" — ne işe yaradığını söylemeyen bir sayı ızgarasına
katılıyor. Ve generator'ın girdisi iyileşiyor, çünkü metin artık nereden geldiğini kendisi
söylüyor. Yayınlanmış contextual-retrieval çalışmaları o bağlam cümlesini chunk başına bir LLM ile
üretiyor; biz etkinin çoğunu dosyada zaten duran bir başlıktan alıyoruz. Dokümanlarında ödünç
alınacak yapı yoksa üret — taranmış PDF'ler, chat log'ları, ticket dump'ları. Varsa ödünç al.

Ölçek büyüdüğünde tabloları chunk'lamayı tamamen bırakırsın. Dosyalanmış bir tarife için kalıcı
çözüm farklı bir temsildir: tabloyu bir kez parse et ve header'ı içine açılmış şekilde chunk başına
bir satır üret — `CLASSIC short-haul, booking class K, fare basis KSHEU26, change penalty EUR 70,
cancellation penalty EUR 90`. Yanlış cevabı üreten belirsizlik artık oluşamaz, çünkü sütun adı
değerin yanında duruyor. Bu, doküman tipi başına yapılan bir iş ve demo ile sistemi ayıran şey de
bu.

10 milyon dokümanda kavramsal olarak hiçbir şey, operasyonel olarak her şey değişir. Chunking
versiyonlanan bir offline batch işine dönüşür: splitter'ı değiştirdiğinde corpus'u yeniden embed
edersin, yani hangi stratejinin hangi vektörü ürettiğini bilmen gerekir. Her chunk'ta parent id'yi
tut ve sorgu anında komşuları getir — chunk'ı retrieve et, bölümü servis et — bu, küçük chunk'ların
kaybettiğinin çoğunu bulanıklığı ödemeden geri kazandırır. Ve 20 soruya güvenmeyi bırak: log'lardan
gerçek sorgu örnekle, birkaç yüz tanesini etiketle, sonucu sorgu tipine göre kırılımlı raporla —
çünkü az önce ölçtüğümüz exact-token düşüşü herhangi bir ortalamanın içinde görünmez.

Kendi sayılarımız için bir uyarı. Yirmi soru iki tasarım arasında karar verdirir; yayınlamaz.
Recursive-600 ile overlap'li fixed-280 arasındaki fark hit@1'de 0.000, MRR'de 0.017 — tek bir
sorunun kayması, anlamsız. Bütün dokümanlarla structure-aware arasındaki fark 0.250, beş soru,
gerçek. Sıralamadan önce farkın büyüklüğüne bak.

<div class="presenter-note">
Biri "biz RecursiveCharacterTextSplitter kullanıyoruz, gayet iyi" derse — katıl, sonra
recursive-600 satırını göster: 0.700, bütün dokümanlardan iyi, structure-aware'den 0.100 kötü ve
geçmesi gereken iki stratejiyle berabere. Header'ı K satırından hâlâ ayırıyor. İyi bir default,
tablolar için çözüm değil. Bunun bir framework tartışmasına dönüşmesine izin verme; argüman ölçümün
kendisi.
</div>

## Çıkış cümlesi

> hit@1 0.800, boilerplate temizlenince 0.850 — bu corpus'un gün boyunca ulaştığı en iyi değer; ve
> exact-token soruları yalnızca körlemesine bölmeyi bıraktığımız için ayakta kaldı. Bu sonuç, tek
> bir notebook process'inin içindeki bir Python listesinde duran 154 vektör. O process gittiğinde
> bu vektörler nerede yaşıyor?
