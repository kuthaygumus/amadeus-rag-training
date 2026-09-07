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
kaydırıyor: overlap header'ı kopyaladığı için artık 4. ve 5. chunk'ta, K satırı ise 7.
chunk'ta. Hâlâ iki chunk arayla. Overlap bir cümleyi
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
`<div class="legal">`), ve altı ücret kuralı sayfası aynı legal footer'la kelimesi kelimesine
kapanıyor. Kendi içinde bir paragrafı tekrarlayan tam olarak bir doküman var: `macro_en_refund.md`,
üstelik tekrarlanan cümle "iptal sütununu oku, değişim sütununu değil" talimatının ta kendisi.

`eval/chunking.py` içindeki `strip_boilerplate()` legal blokları, sayfa artıklarını ve artık
markup'ı siliyor; corpus karakterlerinin **%4.2**'sini kaybediyor — 77.114'ten 73.841'e. Küçük
bir kesinti, ama iki stratejide de metriği oynatıyor:

| | chunk | hit@1 | MRR |
|---|---|---|---|
| fixed-280, ham | 288 | 0.650 | 0.789 |
| fixed-280, temizlenmiş | 275 | **0.700** | **0.802** |
| structure-aware, ham | 152 | 0.800 | 0.846 |
| structure-aware, temizlenmiş | 151 | **0.850** | **0.871** |

Kör stratejiden on üç chunk siliniyor, structure-aware'den bir tane — çünkü structure-aware
boilerplate'i zaten kendi bölümüne ayırmıştı, her chunk'a bulaştırmak yerine. İkisi de aynı beş
puanlık hit@1 kazanıyor: temizlik ne yaparsan yap kazandırıyor, ve MRR 0.871 ile 0.850, bu
corpus'un gün boyunca ulaştığı en iyi değer.
