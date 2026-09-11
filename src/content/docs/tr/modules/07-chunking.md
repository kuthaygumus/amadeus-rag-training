---
title: "7. Chunking, Gürültü ve Ölçüm"
description: "Retrieval neden çöp getirdi?"
---

## Gate sorusu

> **Retrieval neden çöp getirdi?**

Embedder konusu kapandı ve sayı yerinden kıpırdamadı: 28 tam doküman üzerinde `bge-m3` hâlâ hit@1
**0.600**. Yani sorun embedder değildi. Modül 5'ten çalışan bir pipeline ve çalışmayan cevaplarla
çıkmıştık — retrieval `fare_classic_shorthaul` dosyasını birinci sıraya koydu, yani isabet aldı;
model yine de yanlış booking class için cevap verdi, çünkü prompt'a 3 923 karakterlik dokümanın ilk
1 500 karakteri girmişti ve cevabı taşıyan satır o kesiğin ötesinde kalıyordu. Bir kez, körlemesine
kestik ve cevabı kaybettik. Corpus ile prompt arasında duran tek şey, kimsenin tartışmadığı kod
parçası: kesik.

<div class="presenter-note">
Hiçbir şeyi çalıştırmadan önce K sınıfı sorusunu tekrar ekrana al ve salondan "sence ne bozuk?"
oylaması iste. "Model küçük" ve "daha iyi embedder lazım" gelecek. İkisini de tahtaya yaz.
İkisi de cevap değil ve önümüzdeki 35 dakikanın esprisi tam olarak bu: emindiler. Merdiven
tablosunu henüz gösterme.
</div>

## Merdiven

Aynı 28 doküman, aynı `bge-m3` embedding'leri, aynı 20 soru, doküman seviyesinde skorlanıyor.
Tek değişken metnin nasıl kesildiği. Tam dokümanlar: hit@1 **0.600**. Körlemesine 280 karakterlik
chunk'lar: 0.700. 60 karakter overlap ekle: **0.550**. Herkesin ilk uzandığı recursive splitter:
0.700. Dokümanın kendi başlıklarından böl: **0.750**. Önce boilerplate'i temizle, aynı bölme
**0.800**'e çıkıyor.

Üçüncü basamağa bir daha bak. Overlap, bu merdivende geriye giden tek hamle, ve her şeyde geriye
gidiyor: hit@1 0.700 → 0.550, MRR 0.817 → 0.720, recall@5 de düşüyor, 0.917 → 0.883. Üstüne 74
chunk daha ve onları embed etme süresi. hit@1'i, hiç chunk'lamadığın basamağın bile altında.
Overlap güvenli default diye satılır — emin olmadığında yaptığın hamle, çünkü biraz fazlalık nasıl
zarar versin. Bu corpus'ta ise her eksende aynı anda kaybettiren tek değişiklik o.

Buradaki hiçbir farkı sonuç diye okumadan önce: 20 soruda hit@1 yalnızca 0.05'lik adımlarla
hareket eder, çünkü 0.05 *bir sorudur*. Bu merdivendeki her fark bir soru genişliğinde; tek
istisna, üç soruluk overlap düşüşü. Yani ders yön, rakam değil — bu, sonuca sonradan iliştirilmiş
bir çekince değil, sonucu üreten aletin çözünürlüğü.

Merdivenin altından üstüne yirmi sorunun üçü: hiçbir modele dokunmadan, sorgu anında hiçbir maliyet
eklemeden, `eval/chunking.py` içindeki splitter'larla. Günün en büyük kaldıracı bu değil — aynı
structure-aware chunk'larda modül 6'daki embedder seçimi 0.400 değerindeydi (`nomic-embed-text`
0.350'ye karşı `bge-m3` 0.750). Ama insanların default'ta bıraktığı ayar bu.

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
**EUR 90** — kayda geçen çalıştırmada cevap **EUR 65** geldi. EUR 65, bir üstteki satırın değişim
cezası; o satır da W sınıfı ve sınıf adıyla fare basis kodu chunk sınırında kesilmiş durumda.
Yanlış sütun, yanlış satır. Generation seed'li değil, yani hangi yanlış hücrenin geleceği
çalıştırmadan çalıştırmaya değişebilir; kendinden emin şekilde yanlış olması ve hiçbir zaman
çekince koymaması değişmiyor.

<div class="presenter-note">
Sınırı göstermeden önce yüksek sesle sor: "satır bozulmamış — o zaman cevap neden yanlış?"
On beş saniye bekle. Biri bulacak ve katılımcıdan gelince on kat daha sert oturuyor. Ağzında
gevelenmemesi gereken cümle: <strong>satır kurtuldu, header onunla birlikte gelmedi.</strong>
Model EUR 65 dışında bir şey derse üstüne gitme — hangi yanlış sayı gelirse gelsin nokta aynı
noktada duruyor: hiçbir zaman "ayırt edemiyorum" demiyor.
</div>

**Overlap bunu düzeltmiyor ve denemenin bedelini kesiyor.** 60 karakterlik overlap corpus genelinde
74 chunk ekliyor (294 → 368) ve sınırı kaydırıyor. Ücret sayfasında header satırının tamamı artık
5. chunk'ta, ilk iki sütun adı 4. chunk'ın kuyruğuna kopyalanmış durumda, K satırı ise 7. chunk'ta.
Hâlâ iki chunk arayla. Overlap bir cümleyi ortadan bölmeye karşı sigortadır; 612 karakter öteye
yapılan bir referansa karşı hiçbir şey yapmaz — ve o 368 birbirine benzer parça, hit@1'i merdivenin
en kötü sıralaması olan 0.550'ye indiren şeydir.

**Recursive splitting de düzeltmiyor.** Rahatsız edici olan bu, çünkü her framework'ün default'u ve
salondaki çoğu kişinin production'da kullandığı şey. Önce paragraflardan, sonra satırlardan, sonra
cümlelerden bölüyor ve düz metni bozmadan koruyor — ama header 4. chunk'ta, K satırı 5. chunk'ta.
Yan yana, yine de ayrı. Sebebi ayarla kapatılamaz: tablo split boyutundan geniş. 600 karakterlik
bir pencere dokuz sütunluk header'ı artı yedi satırı alamıyor; separator mantığı ne derse desin
kesik tablonun içine düşüyor. Bu basamağı aklında tut: recall@5'i 0.900, yalnızca düz fixed-280'in 0.917'sinin gerisinde.

**Structure-aware bölme düzeltiyor.** Dokümanın kendi sınırlarından kes — `##` başlıkları,
`RULE n.`, `SECTION n` — tablo kendisini tanıtan kuralla birlikte kalıyor: header da K satırı da
4. chunk'ta. Sonra her chunk'ın başına geldiği bölümün başlığı yazılıyor; böylece çıplak bir sayı
ızgarası olacak parça etiketiyle geliyor:

```
[RULE 2A. Reading the schedule. Each booking class carries its own fare basis code. The change]
| Booking class | Fare basis | Advance purchase | ... | Change penalty | Cancellation penalty | ...
```

Prefix bir yol değil, başlık satırının kendisi: `structure_aware()` doküman adına yalnızca ilk
başlıktan önceki metin için düşüyor. "Contextual retrieval" diye satılan şeyin tamamı bu — her
parçaya nereden geldiğini söylemek — ve maliyeti bir string birleştirme. İki sıralama sütununun
ikisinde de önde gidiyor: hit@1 0.750, MRR 0.817; üstelik chunk'lanan stratejilerin en düşük
recall@5'iyle, 0.833.

## Tek sayı raporlamamak için iki sebep

**Yükselen bir ortalama, geriye giden bir kategoriyi saklayabilir.** Dört `exact_token` sorusu —
literal bir uçuş numarası ya da bülten id'si, artı bir ücret tablosu sorgusu — tam dokümanlarda
**1.000** alıyor. Fixed-280 bunu 0.750'ye düşürüyor, overlap eklendiğinde **0.500**'e iniyor: dördün
ikisi gitti. Recursive-600 ile structure-aware ikisi de 1.000'e dönüyor. Yani fixed-280 manşet
ortalamayı yükseltti — 0.600 → 0.700 — ve bunun bedelini tam da salonun ilk soracağı kategoriden
ödedi; overlap basamağı ise iki kez ödedi, hem ortalamada hem kategoride. Manşet sayıda bunların
olduğunu söyleyen hiçbir şey yok. Sadece tip kırılımı gösteriyor.

**En iyi recall@5, en iyi sıralayıcıya ait değil.** Fixed-280 o sütunun tepesinde, **0.917**,
ama hit@1'i 0.700 — ve üç naive kesim de recall@5'te structure-aware'i geçiyor: 0.917, 0.883 ve
0.900'e karşı 0.833. Tesadüf değil, aritmetik: daha çok küçük parça, gold dokümana ilk beşte
görünmek için daha çok şans verir, birinci sıraya koymayı ise zorlaştırır. recall@5'i seçersen kör
chunking kazanır; hit@1'i seçersen kaybeder. İkisi de dürüst. Aralarındaki seçim, ölçüm mü yoksa
reklam mı yaptığına karar verdiğin yerdir.

## Gürültü kozmetik değil, ölçülebilir

Corpus, gerçek bir export'un taşıdığı dağınıklığı bilerek taşıyor ve notebook bunu 28 doküman
üzerinde sayıyor: **6** doküman aynı legal footer'la, bayt bayt aynı şekilde kapanıyor; metinde
`Page 3 of 7` türünden **25** sayfa numarası artığı var — bir kısmı tablo ortasında kendi satırında
duruyor, ki `strip_boilerplate()` tam olarak bu biçimi yakalıyor; ve **42** parça artık HTML
(`<br>`, `&nbsp;`, `<div class="legal">`). Bir doküman kendi içinde bir gövde paragrafını
tekrarlıyor: `macro_en_refund.md`; tekrarlanan cümle tam da "the **cancellation** column, not the
change column" talimatının kendisi.

`eval/chunking.py` içindeki `strip_boilerplate()` legal blokları, sayfa artıklarını ve artık
markup'ı siliyor ve corpus karakterlerinin **%4.2**'si gidiyor — 78 310'dan 75 037'ye. Küçük bir
kesinti, ama metriği oynatıyor:

| | chunk | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| structure-aware, ham | 154 | 0.750 | 0.833 | 0.817 |
| structure-aware, temizlenmiş | 153 | **0.800** | 0.833 | **0.841** |

Structure-aware index'ten yalnızca bir chunk siliniyor, çünkü structure-aware bölme boilerplate'i
zaten kendi bölümlerine ayırmıştı, her chunk'a bulaştırmak yerine. Temizlenmiş corpus'u yalnızca
structure-aware ile skorladık, o yüzden temizlenmiş fixed-280 satırı yok. Fark ölçüldü ve tek
değişken temizlikti: hit@1'de **+0.050**, MRR'de **+0.024**, recall@5 değişmedi — yirmi sorudan
biri, yani büyüklüğü değil yönü oku. Kazancın tamamı da tek bir yere düşüyor: `en_en` 0.500 →
0.750 çıkıyor, başka hiçbir şey kıpırdamıyor.

## Ne çalıştırıyorsun

Bu modülde mekanizma ile ölçüm iki ayrı artefakt.

**VS Code — `notebooks/05_chunking_and_noise.py`, ilk markdown hücresinden sonraki `# %%` bloğu:**
blokları Shift+Enter ile çalıştır. İki model çağrısı dışında her şey string kesme işlemi, yani yavaş
bir laptopta takılmaz. En çok işe yarayan hücre hiç model istemiyor: her strateji için hangi
chunk'ta sütun header'ı, hangi chunk'ta K satırı var, yazdır. `sheet` ve `C` notebook'un ilk
bloğundan geliyor — bu, dosyanın içindeki bir hücre, yazman gereken bir şey değil.

```python
pieces = C.fixed(sheet, size=280, overlap=0)
print(f"{len(pieces)} chunks\n")
for n, piece in enumerate(pieces):
    has_row = "| K |" in piece and "EUR 90" in piece
    has_header = "Booking class" in piece and "Cancellation penalty" in piece
    if has_row or has_header:
        print(f"chunk {n}: {'THE K ROW' if has_row else ''}{'THE COLUMN HEADER' if has_header else ''}")
```

**Ne görmen gerekiyor.** `chunk 4: THE COLUMN HEADER` ve `chunk 6: THE K ROW`. İki chunk arayla; ve
yalnızca K satırı chunk'ından üretilen cevap EUR 65 çıkıyor, oysa doğru cevap EUR 90.
**Kabaca ne kadar sürüyor.** Notebook'un yazıldığı tempoda on iki dakika.

**Terminal (repo kökü):** tek komut, tek tablo.

```bash
python exercises/m7_chunking_ladder.py   # notebook'ta çalışırken bunu arkada çalışır bırak
```

**Ne görmen gerekiyor.** Beş merdiven satırı, artı temizlenmiş corpus için altıncı bir satır; her
birinde chunk sayısı, hit@1, recall@5, MRR ve beş soru tipi. Sonunda
`Stripping the legal footer removed 4.2% of the characters.` satırı. Durup bakılacak satır şu:
`The column to argue about: cutting into fixed 280-character pieces moved hit@1 0.600 -> 0.700 while exact_token went 1.000 -> 0.750.`
**Kabaca ne kadar sürüyor.** İki-üç dakika, neredeyse tamamı embedding çağrısı.

**VS Code — `05_chunking_and_noise.py` dosyasının sonuna yeni bir `# %%` bloğu,** corpus genelindeki
chunk sayılarını orada da istersen:

```python
for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    ids, texts, parents = C.chunk_corpus(docs, strategy)
    print(strategy, len(texts))       # 294 · 368 · 197 · 154
```

<div class="presenter-note">
35 dakika: merdiven 6, header demo 12, overlap ve recursive 6, metrik uyuşmazlığı 6, gürültü 5.
Merdiven komutunu notebook'u açmadan önce repo kökündeki terminalde çalıştırmaya bırak — iki-üç
dakika sürüyor ve demo o süreyi kapatıyor. Birinin laptopunda patlarsa sayıları bu sayfadan oku ve
sınır hücresini canlı çalıştır: string kesme işlemi, patlaması mümkün değil. Script'in kendi kapanış
paragraflarını az önce yaptığı koşumdan türetiyor, o yüzden onları hazırlamak yerine ekrandan oku:
overlap basamağını kendisiyle çelişen basamak diye adlandıracak, ve fixed 280'i yalnızca 0.700
sıralayan recall@5 galibi olarak gösterecek. Bu modülün üstünde döndüğü iki argüman da bunlar ve
senin ağzından değil script'in ağzından çıkıyorlar — asıl mesele de bu. Sınır hücresi demo,
merdiven kanıt. Gün sarktığında kesilecek ölçüm bu değil.
</div>

## Sayılar ne dedi

<div class="measured">

| strateji | chunk | hit@1 | recall@5 | MRR |
|---|---|---|---|---|
| tam dokümanlar | 28 | 0.600 | 0.717 | 0.677 |
| fixed 280 | 294 | 0.700 | **0.917** | 0.817 |
| fixed 280 + overlap 60 | 368 | **0.550** | 0.883 | 0.720 |
| recursive 600 | 197 | 0.700 | 0.900 | 0.806 |
| structure-aware 900 | 154 | 0.750 | 0.833 | 0.817 |
| **structure-aware + boilerplate temizlenmiş** | 153 | **0.800** | 0.833 | **0.841** |

Aynı merdiven, soru tipine göre hit@1:

| strateji | tr_tr | tr_en | en_en | exact_token | multi_hop |
|---|---|---|---|---|---|
| tam dokümanlar | 1.000 | 0.333 | 0.500 | 1.000 | **0.000** |
| fixed 280 | 1.000 | 0.500 | 0.750 | 0.750 | 0.500 |
| fixed 280 + overlap 60 | 0.750 | 0.500 | 0.500 | **0.500** | 0.500 |
| recursive 600 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 |
| structure-aware 900 | 1.000 | **0.667** | 0.500 | 1.000 | 0.500 |
| + boilerplate temizlenmiş | 1.000 | **0.667** | **0.750** | 1.000 | 0.500 |

`multi_hop` sütunu tek başına bir cümleyi hak ediyor. Tam dokümanlarda 0.000 — iki multi-hop
sorusunun ikisinde de doğru doküman birinci sıraya gelmiyor. Chunk'lanan her strateji 0.500'e
çıkıyor ve hiçbiri onu geçemiyor. Chunking sana ikisinden birini kazandırıyor, ikincisini hiçbir
kesim kazandırmıyor — modül 10'un tırmanmak için kurulduğu duvar bu.

`fare_classic_shorthaul.md` içinde header nereye düşüyor — 0-tabanlı, model gerekmeden ölçüldü:

| strateji | o dosyadaki chunk | header | K satırı | birlikte mi? |
|---|---|---|---|---|
| tam dokümanlar | 1 | — | — | evet — hiçbir şey kesilmedi |
| fixed 280 | 15 | 4 | 6 | hayır |
| fixed 280 + overlap 60 | 18 | 5 | 7 | hayır — overlap sınırı kaydırıyor |
| recursive 600 | 10 | 4 | 5 | hayır — tablo split boyutundan geniş |
| structure-aware 900 | 10 | 4 | 4 | evet |

</div>

`multi_hop` sütunu her satırda 0.000 — bu sayfadaki her değişiklikten önce de sonra da. Metni hiçbir
kesme biçimi oraya ulaştıramıyor; modül 10 tam olarak bu başarısızlığın üstüne kuruluyor.

## Daha derine

Retriever bir dokümanı hiç görmez. Chunk başına tek bir vektör görür — bütün bir bölüm için tek
bir nokta. Dokuz sütunluk bir header'ı, yedi ücret satırını ve bir sayfa numarası artığını tek bir
1024 boyutlu noktada ortalarsan, short-haul ücretleri hakkında her şeye yakın ama hiçbir şey
hakkında spesifik olmayan bir şey elde edersin. Yani chunk boyutu hassasiyetle takas edilir: uzun
chunk generator'a daha çok bağlam, retriever'a daha bulanık bir vektör verir. Structure-aware bölme kazanıyor çünkü bu iki baskıyı aynı yöne çeviriyor — bir
bölüm hem anlamın hem embedding'in doğal birimi.

Başlık prefix'i iki iş yapıyor. Vektör kayıyor, çünkü başlığın kendi kelimeleri — "Reading the
schedule", "fare basis code", "change" — ne işe yaradığını söylemeyen bir sayı ızgarasına
katılıyor. Ve generator'ın girdisi iyileşiyor, çünkü metin artık nereden geldiğini kendisi
söylüyor. Yayımlanmış contextual-retrieval çalışmaları o bağlam cümlesini chunk başına bir LLM ile
üretiyor; biz etkinin çoğunu dosyada zaten duran bir başlıktan alıyoruz. Dokümanlarında ödünç
alınacak yapı yoksa üret — taranmış PDF'ler, chat log'ları, ticket dump'ları. Varsa ödünç al.

Ölçek büyüdüğünde tabloları chunk'lamayı tamamen bırakırsın. Dosyalanmış bir tarife için kalıcı
çözüm farklı bir temsildir: tabloyu bir kez parse et ve header'ı içine açılmış şekilde chunk başına
bir satır üret — `CLASSIC short-haul, booking class K, fare basis KSHEU26, change penalty EUR 70,
cancellation penalty EUR 90` — böylece yanlış cevabı üreten belirsizlik artık oluşamaz. Bu, doküman
tipi başına yapılan bir iştir ve chunking'i versiyonlanan bir offline batch işine dönüştürür:
splitter'ı değiştirdiğinde corpus'u yeniden embed edersin, yani hangi stratejinin hangi vektörü
ürettiğini bilmen gerekir.

<div class="presenter-note">
Biri "biz RecursiveCharacterTextSplitter kullanıyoruz, gayet iyi" derse — katıl, sonra
recursive-600 satırını göster: hit@1 0.700, recall@5 0.900 ve header'ı K satırından hâlâ
ayırıyor. İyi bir default, tablolar için çözüm değil. "Hangisi daha iyi" sorusunun dürüst cevabı
şu: structure-aware'den onu ayıran şey tek bir soru; asıl kanıt, model de benchmark da istemeyen
header demosu — yirmi soruya bağlı olmayan kısım orası. Bunun bir framework tartışmasına
dönüşmesine izin verme; argüman ölçümün kendisi.
</div>

## Çıkış cümlesi

> hit@1 0.750, boilerplate temizlenince 0.800 — arada tek bir soru var, o yüzden günün geri kalanı
> 0.750 ile tartışıyor; ve bu, corpus'un gün boyunca ulaştığı en iyi değer. Bunu bir model değil,
> bir splitter kazandırdı. Aynı zamanda bu sonuç, tek bir notebook process'inin içindeki bir Python
> listesinde duran 154 vektör. O process gittiğinde bu vektörler nerede yaşıyor?
