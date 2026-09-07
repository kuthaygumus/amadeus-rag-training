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
İkisi de cevap değil ve önümüzdeki 40 dakikanın esprisi tam olarak bu: emindiler. Merdiven
tablosunu henüz gösterme.
</div>

## Merdiven

Aynı 28 doküman, aynı `bge-m3` embedding'leri, aynı 20 soru, doküman seviyesinde skorlanıyor.
Tek değişken metnin nasıl kesildiği. Bütün dokümanlar: hit@1 **0.550**. Körlemesine 280
karakterlik chunk'lar: 0.650. 60 karakter overlap ekle: 0.700. Herkesin ilk uzandığı recursive
splitter: 0.700. Dokümanın kendi başlıklarından böl: **0.800**, MRR 0.655 → **0.846**.

Hiçbir modele dokunmadan, sorgu anında hiçbir maliyet eklemeden, otuz satır Python ile hit@1'de
%45 göreli kazanç. Chunking gün boyunca ölçtüğümüz en büyük kaldıraç ve insanların default'ta
bıraktığı tek ayar.

## Hata satırda değil, header'da

Herkes sabit boyutlu chunking'in tablo satırını ortadan ikiye böldüğünü sanıyor. Bölmüyor. İşte
`fare_classic_shorthaul.md` dosyasının 280 karakterlik kör bölmedeki 6. chunk'ı:

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

Model `EUR 70 | EUR 90 | EUR 180` okuyup tahmin ediyor. Cevabı **EUR 70** veriyor, yani değişim
cezası; oysa iptal cezası **EUR 90**. Çekinmiyor da. Çünkü oturduğu yerden ortada hiçbir
belirsizlik yok: bir tane ilk sayı var ve ilk sayı genelde cevaptır.

<div class="presenter-note">
Sınırı göstermeden önce yüksek sesle sor: "satır bozulmamış — o zaman cevap neden yanlış?"
On beş saniye bekle. Biri bulacak ve katılımcıdan gelince on kat daha sert oturuyor. Ağzında
dolanmaması gereken cümle: <strong>satır kurtuldu, header onunla birlikte gelmedi.</strong>
</div>

**Overlap bunu düzeltmiyor.** 60 karakterlik overlap 75 chunk ekliyor (288 → 363) ve sınırı
kaydırıyor: header 5. chunk'ta, K satırı 7. chunk'ta. Hâlâ iki chunk arayla. Overlap bir cümleyi
ortadan bölmeye karşı sigortadır; 900 karakter öteye yapılan bir referansa karşı hiçbir şey
yapmaz.

**Recursive splitting de düzeltmiyor.** Rahatsız edici olan bu, çünkü her framework'ün
default'u ve salondaki çoğu kişinin production'da kullandığı şey. Önce paragraflardan, sonra
satırlardan, sonra cümlelerden bölüyor ve düz metni gerçekten güzel koruyor — ama header 4.
chunk'ta, K satırı 5. chunk'ta. Yan yana, yine de ayrı. Sebebi ayarla kapatılamaz: tablo split
boyutundan geniş. 600 karakterlik bir pencere dokuz sütunluk header'ı artı yedi satırı
alamıyor; separator mantığı ne derse desin kesik tablonun içine düşüyor.

**Structure-aware bölme düzeltiyor.** Dokümanın kendi sınırlarından kes — `##` başlıkları,
`RULE n.`, `SECTION n` — tablo kendisini tanıtan kuralla birlikte kalıyor: header da K satırı da
4. chunk'ta. Sonra her chunk'ın başına doküman adını ve geldiği bölüm başlığını yaz; böylece
çıplak bir sayı ızgarası olacak parça `[fare classic shorthaul > ## SECTION 4 — VOLUNTARY
CHANGES AND CANCELLATIONS]` etiketiyle geliyor. "Contextual retrieval" diye satılan şeyin tamamı
bu: her parçaya nereden geldiğini söylemek. Maliyeti bir string birleştirme.

## Tek sayı raporlamamak için iki sebep

**Fixed-280 ortalamayı yükseltirken bir kategoriyi yarıya indiriyor.** Exact-token sorguları —
uçuş kodları, bülten id'leri — bütün dokümanlarda **1.000**, fixed-280'de **0.500**; ancak
structure-aware ile 1.000'e dönüyor. Manşet hit@1 0.550 → 0.650 çıkarken altında bir soru sınıfı
iki kat kötüleşti.

**Recall@5 en kötü stratejide en yüksek.** Fixed-280 **0.950**, structure-aware 0.850. Tesadüf
değil, aritmetik: 288 küçük parça, gold dokümanın ilk beşte bir yerde görünmesi için daha çok
şans verir, ama birinci sıraya koymayı zorlaştırır. recall@5'i seçersen kör chunking kazanır;
hit@1'i seçersen 0.150 farkla kaybeder. İkisi de dürüst. Aralarındaki seçim, ölçüm mü yoksa
reklam mı yaptığına karar verdiğin yerdir.

## Gürültü kozmetik değil, ölçülebilir

Corpus, gerçek bir export'un taşıdığı pisliği bilerek taşıyor. 28 dokümandan **23**'ünde tablo
ortasında kalmış bir `Page 3 of 7` var, **16**'sında artık HTML (`<br>`, `&nbsp;`,
`<div class="legal">`), **6**'sında birebir tekrarlanan bir paragraf — bunlardan biri
`macro_en_refund.md`, üstelik tekrarlanan cümle "iptal sütununu oku, değişim sütununu değil"
talimatının ta kendisi.

Sayfa artıklarını, HTML'i, boilerplate footer'ları ve tekrar eden paragrafları temizlediğinde
corpus karakterlerinin **%10.8**'ini kaybediyor. Fixed-280 chunk sayısı **288'den 260'a**
düşüyor: on chunk'tan yaklaşık biri ağırlıklı olarak boilerplate'ti ve gerçek içerikle ilk beş
için yarışıyordu. Structure-aware ise sadece **152'den 149'a** düşüyor, çünkü o boilerplate
zaten kendi bölümlerine ayrılmıştı ve hiçbir kural chunk'ını sulandırmıyordu. Kör chunking aynı
pislikten iki kez ceza yiyor.

Sadece temizlemenin retrieval'a etkisini ölçmedik; dürüst sebebi şu: 20 soruda o fark gürültü
tabanının içinde kalır. Bunu kesinleştirecek olan şey, 0.05'in bir anlam taşıyacağı büyüklükte
bir gold set üzerinde, tek değişken temizlik olacak şekilde aynı benchmark'ı koşmak.

## Ne çalıştırıyorsun

Notebook: `05_chunking_and_noise.ipynb`

```bash
jupyter lab notebooks/05_chunking_and_noise.ipynb
python eval/run_benchmark.py     # aynı karşılaştırma, komut satırından
```

Notebook, benchmark'ın çağırdığı fonksiyonların aynısını `eval/chunking.py` içinden çağırıyor:

```python
from chunking import fixed, recursive, structure_aware, chunk_corpus, to_documents
from metrics import load_gold, evaluate, compare

for strategy in ["fixed-280", "fixed-280+overlap60", "recursive-600", "structure-aware"]:
    ids, texts, parents = chunk_corpus(documents, strategy)
    print(strategy, len(texts))
```

En kritik hücre hiçbir model gerektirmiyor: her strateji için sütun header'ının hangi chunk'ta,
K satırının hangi chunk'ta olduğunu yazdırıyor.

<div class="presenter-note">
40 dakika: merdiven 8, header demosu 12, overlap ve recursive 8, metriklerin çelişmesi 6,
gürültü 6. Benchmark salondaki laptop'larda yavaş kalırsa retrieval sayılarını cache'ten oku ve
chunk sınırı hücresini canlı çalıştır — o sadece string dilimleme, patlaması mümkün değil. Demo
o hücre; benchmark ise kanıt.
</div>

## Sayılar ne dedi

<div class="measured">

| strateji | hit@1 | recall@5 | MRR | chunk |
|---|---|---|---|---|
| bütün dokümanlar | 0.550 | 0.717 | 0.655 | 28 |
| fixed 280 | 0.650 | **0.950** | 0.789 | 288 |
| fixed 280 + overlap 60 | 0.700 | 0.867 | 0.795 | 363 |
| recursive 600 | 0.700 | 0.900 | 0.816 | 195 |
| **structure-aware 900** | **0.800** | 0.850 | **0.846** | 152 |

| strateji | header K satırıyla kalıyor mu? | exact-token hit@1 |
|---|---|---|
| bütün dokümanlar | konu dışı — hiçbir şey kesilmedi | 1.000 |
| fixed 280 | hayır (header chunk 4, K satırı chunk 6) | 0.500 |
| fixed 280 + overlap 60 | hayır (chunk 5 / chunk 7) — overlap sınırı sadece kaydırıyor | — |
| recursive 600 | hayır (chunk 4 / chunk 5) — tablo split boyutundan geniş | — |
| structure-aware 900 | evet (ikisi de chunk 4) | 1.000 |

</div>

## Daha derine

Retriever dokümanı hiç görmez. Chunk başına tek bir vektör görür — tüm parça için tek bir nokta.
Dokuz sütunluk bir header'ı, yedi ücret satırını ve bir sayfa numarası artığını tek bir 1024
boyutlu noktada ortalarsan, kısa menzil ücretlerine dair her şeye yakın ama hiçbir şeyde spesifik
olmayan bir şey elde edersin. Chunk boyutu bu yüzden model kalitesinden bağımsız olarak
precision'la takas edilir: uzun chunk generator'a daha çok bağlam, retriever'a daha bulanık bir
vektör verir. Structure-aware bölme kazanıyor, çünkü bu iki baskıyı aynı yöne çeviriyor — bir
bölüm hem doğal anlam birimi hem de doğal embedding birimidir.

Başlık prefix'i iki iş birden yapıyor. Vektör kayıyor, çünkü "VOLUNTARY CHANGES AND
CANCELLATIONS" chunk'ı normalde kaçıracağı iptal sorgularına doğru çekiyor. Bir de generator'ın
girdisi düzeliyor, çünkü metin artık nereden geldiğini kendisi söylüyor. Yayınlanmış contextual
retrieval çalışmaları o bağlam cümlesini chunk başına bir LLM'e ürettiriyor; biz aynı etkinin
büyük kısmını dosyada zaten duran başlıktan alıyoruz. Dokümanların ödünç alınacak yapısı yoksa
üret — taranmış PDF, chat log, ticket dökümü. Yapısı varsa ödünç al.

Ölçek büyüdüğünde tabloları chunk'lamayı tamamen bırakırsın. Filed tariff için kalıcı çözüm daha
zeki bir splitter değil, farklı bir gösterim: tabloyu bir kez parse et ve her satırı, header'ı
içine açılmış halde tek bir chunk olarak yaz — `CLASSIC short-haul, booking class K, fare basis
KSHEU26, change penalty EUR 70, cancellation penalty EUR 90`. Yanlış cevabı üreten belirsizlik
artık oluşamaz, çünkü sütun adı değerin yanında duruyor. Bu, doküman tipi başına iş demek; demoyu
sistemden ayıran da bu iş.

10 milyon dokümanda kavramsal olarak hiçbir şey, operasyonel olarak her şey değişir. Chunking
versiyonlanmış bir offline batch job'a dönüşür: splitter'ı değiştirdiğin gün corpus'u yeniden
embed edersin, dolayısıyla hangi vektörü hangi stratejinin ürettiğini bilmek zorundasın. Her
chunk'ta parent doküman id'sini tut ve sorgu anında komşuları çek — chunk'ı bul, bölümü ver —
böylece küçük chunk'ların kaybettiğinin çoğunu bulanıklığı ödemeden geri alırsın. Ve 20 soruya
güvenmeyi bırak: gerçek sorguları log'lardan örnekle, birkaç yüz tane etiketle, raporu soru
tipine göre ayrıştır; çünkü az önce ölçtüğümüz exact-token çöküşü hiçbir ortalamada görünmüyor.

Kendi sayılarımız için bir uyarı. Yirmi soru iki tasarım arasında karar verdirir, yayınlatmaz.
Recursive-600 ile fixed-280+overlap arasındaki fark hit@1'de 0.000, MRR'de 0.021 — tek bir
sorunun yer değiştirmesi, yani anlamsız. Bütün dokümanlar ile structure-aware arasındaki fark
0.250, yani beş soru, yani gerçek. Sıralamayı okumadan önce farkın büyüklüğünü oku.

<div class="presenter-note">
Arkadan "biz RecursiveCharacterTextSplitter kullanıyoruz, gayet iyi" gelirse önce katıl, sonra
recursive-600 satırını göster: 0.700, kör bölmeden iyi, structure-aware'den 0.100 kötü ve hâlâ
header'ı K satırından ayırıyor. İyi bir default, tablolar için çözüm değil. Bunu framework
tartışmasına çevirtme; argüman ölçümün kendisi.
</div>

## Çıkış cümlesi

> hit@1 artık 0.800. Ama fixed-280 exact-token retrieval'ı yarıya indirmişti — ortalama başka neyi saklıyor?
