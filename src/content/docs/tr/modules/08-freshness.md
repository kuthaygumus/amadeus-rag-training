---
title: "8. Güncellik: RAG Neden Var"
description: "Aynı soru, geçen çeyreğin kural kitabına ve bu çeyreğinkine: 120, retrain değil bir re-ingest ile 90 oluyor. Modül 3'ün açtığı döngüyü kapatan modül bu."
---

## Gate sorusu

> **Modül 3'ün fine-tune edilmiş modeli Q2'de donmuş. Modül 5–7'de kurduğumuz her şey fine-tuning'den neden daha iyi?**

Günün nerede durduğu konusunda dürüst ol. Modül 3, kural kitabını ağırlıklarında taşıyan bir model gösterdi — `kraken-q2`, `corpus/2026-Q2/` üzerinde bir LoRA fine-tune, `UNVERIFIED: henüz kurulmadı, trainer demo'su olarak gösteriliyor`. Modül 5–7 ise çok daha hantal bir şey kurdu: bir container, bir embedding modeli, K satırını bulabilmesi için iki kez düzeltilmesi gereken bir chunker. Salondaki adil bir şüphecinin sorması gereken soru şu: hantal olan neden kazanıyor?

Tek bir eksende kazanıyor ve o eksen zaman. **Fine-tune edilmiş bir model, dokümanların onu eğittiğin gün ne dediğini bilir. Bir RAG pipeline'ı şu an ne dediklerini bilir.** Bu modül o cümleyi dört request'e ve iki sayıya çeviriyor.

<div class="presenter-note">
On dört dakika; dört request yaklaşık dördü. Request 1'den önce el kaldırt: "Hâlâ fine-tuning daha basit tasarım olurdu diyen var mı?" — birkaç el kalkar ve "daha basit" konusunda haksız değiller; onları sondaki tablo için sakla. Ollama kapalıysa: bu sayfayı oku, iki sayı da üzerinde; DELTA diff'i de hiçbir model gerekmeden projektörde açabileceğin bir dosya.
</div>

## İki edition, tek satır

`corpus/2026-Q2/` geçen çeyreğin kural kitabı: 21 doküman. `corpus/2026-Q3/` bu çeyreğinki: 28. Q2 edition'ı `scripts/make_q2.py` ile Q3'ten üretiliyor ve `corpus/DELTA.md` yaptığı her değişikliği listeliyor — yani bu eğitimde bir kez olsun, iki edition arasında *tam olarak* neyin değiştiğini biliyorsun.

Manşet, `fare_classic_shorthaul.md` içinde tek bir tablo hücresi; RULE 2A, K satırı:

```diff
- (Q2) | K | KSHEU26 | none | none | EUR 70 | EUR 120 | EUR 240 | 2 x 23 kg | Yes |
+ (Q3) | K | KSHEU26 | none | none | EUR 70 | EUR 90 | EUR 180 | 2 x 23 kg | Yes |
```

Sadece günün konusu olan sayıyı değil, satırın tamamını oku. İptal cezası EUR 120'den EUR 90'a indi; yanındaki no-show cezası da 240'tan 180'e — çünkü RULE 4 no-show'u iptal cezasından türetiyor. Tek bir iş kararı, iki hücre. Doküman id'si de bir adım ilerledi, `FR-CL-SH-2026Q2-013` → `FR-CL-SH-2026Q3-014`, ve yeni sayfanın `Supersedes:` satırı eskisini adıyla gösteriyor.

Başka ne değişti — `DELTA.md` nasıl söylüyorsa öyle:

- Yedi doküman Q2'de hiç yoktu: `sop_misconnect_v4.md` ve altı kış sezonu tarife bülteni `bulletin_scb_2026_09xx.md`.
- `sop_misconnect_v3.md` Q2'de **Current**, Q3'te **Superseded** — aynı dosya, aralarında tek kelimelik metadata farkı.
- Üç policy rutin yeniden yayım gördü: `policy_corporate_travel.md` 6.1 → 6.2, `policy_travel_approval.md` 4.7 → 4.8, `policy_expense_reimbursement.md` 5.3 → 5.4.
- Her edition damgası ve her doküman id'si bir çeyrek geriye yazıldı; başka dokümanlardaki çapraz referanslar dahil. Bunlar havayolu hakkında gerçekler değil; Q2 klasörünü bir Q2 edition'ı yapan şeyler.

Geri kalan her şey byte'ı byte'ına aynı. Gerçek bir kural kitabının çeyreklik yeniden yayımı tam böyle görünür: her şeyin yeniden etiketlenmesinin içine gömülü bir avuç gerçek değişiklik.

## Bir retrain'in bedeli ne olurdu

Modül 3'ün pipeline'ını, az sonra çalıştıracağının yanına koy.

`kraken-q2`'yi Q3'e taşımak için dataset'i yeniden üretirdin (`scripts/make_finetune_dataset.py`, Q2 için 695 soru–cevap çifti — EUR 120 öğreten 18 tanesinin 90 olması gerekirdi ve önce onları bulman gerekirdi), LoRA notebook'unu bir GPU'da yeniden koştururdun, GGUF'a çevirirdin, `ollama create` ile bir `kraken-q3` yapardın, değerlendirirdin ve her laptopa dağıtırdın. Bunun bir sahibi olur. Bir takvimi olur. Ve bütün bunun sonunda model 90'ı *hangi dokümandan* aldığını hâlâ söyleyemez — sayı ağırlıklara yayılmıştır, bir satırda durmaz.

RAG pipeline'ını Q3'e taşımak için farklı bir `edition` ile `POST /ingest` atarsın. Cevap, ne kadar sürdüğünü milisaniye cinsinden, aşama aşama söyler.

<div class="presenter-note">
Ağzında gevelenmemesi gereken cümle: "Model hiç değişmedi. Dokümanlar değişti ve cevap onları takip etti." Request 4'ten sonra, `sources[0].source` ekrandayken söyle. `stages.total` içindeki saniyeleri fazla satma — gerçek bir corpus on binlerce doküman ve onu embed etmek saniye değil dakikalar, saatler sürer. Mesele hızlı olması değil; mesele bunun training koşusu, GPU, değerlendirme döngüsü ve içinde insan olmayan bir batch job olması.
</div>

## Ne çalıştırıyorsun

Dört request, sırayla, hepsi tek bir Bruno klasöründe. Her `/ingest` kendi Chroma collection'ını yazıyor ve adını konfigürasyonundan alıyor — `kraken-2026-Q2-structure-1500-strip` ve `kraken-2026-Q3-structure-1500-strip` — yani Q3'ü ingest etmek Q2'nin üzerine yazmıyor. En son ingest, `/query` için varsayılan hedef oluyor.

**Bruno — `06-freshness` › `1-ingest-q2`**

```json
{
  "edition": "2026-Q2",
  "strategy": "structure",
  "stripBoilerplate": true
}
```

Modül 7'nin bitirdiği strateji, geçen çeyreğin kitabına uygulanmış hâli. `edition` (`2026-Q2`), `documents` (21), `collection` ve `stages` alanlarını oku — `read`, `chunk`, `embed`, `store`, `total`, hepsi ms.

**Bruno — `06-freshness` › `2-query-q2`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
}
```

`answer` alanını oku — **EUR 120**. Q2'deki kural buydu ve Q2 üzerinde fine-tune edilmiş bir model sonsuza kadar bunu söylerdi. `sources[0].source` alanına da bak: `fare_classic_shorthaul`, Q2 kopyası. `abstained` `false` olmalı.

**Bruno — `06-freshness` › `3-ingest-q3`**

```json
{
  "edition": "2026-Q3",
  "strategy": "structure",
  "stripBoilerplate": true
}
```

"Retrain" bu. `stages.total` alanını oku. O sayı, senin laptopunda, sistemi geçen çeyreğin kurallarından bu çeyreğinkine taşımanın toplam bedeli. `documents` artık 28; `collection` `2026-Q3-structure-1500-strip` ile bitiyor.

**Bruno — `06-freshness` › `4-query-q3`**

Request 2 ile aynı body. `answer` alanını oku — **EUR 90**, kaynağı `[fare_classic_shorthaul]`; `sources[0].excerpt` içinde RULE 2A başlığı ile K satırı birlikte olmalı — modül 7 tam olarak bunun içindi.

**120 → 90, retrain değil bir re-ingest ile. Model hiç değişmedi; dokümanlar değişti ve cevap onları takip etti — açıp bakabileceğin bir kaynakla.**

İki collection da hâlâ Chroma'da duruyor. İki cevabı yan yana tutmak için collection'ı body'ye koy:

**Bruno — `06-freshness` › `4-query-q3`, body düzenlenmiş**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?",
  "collection": "kraken-2026-Q2-structure-1500-strip"
}
```

Aynı model, aynı soru, aynı prompt şablonu, aynı embedding — yine 120. `collection`'ı Q3 adına geri al, 90. İkisi arasında zekâya dair hiçbir şey değişmedi; sadece okuduğu raf değişti.

<div class="presenter-note">
Zaman kalırsa projektörde bir şey daha: `corpus/2026-Q2/fare_classic_shorthaul.md` ile `corpus/2026-Q3/fare_classic_shorthaul.md`'yi yan yana aç ve `sources[0].excerpt`'in birebir o metin olduğunu göster. Burada "denetlenebilir" demek bu — bir destek temsilcisi tıklayıp satıra kadar inebilir. Fine-tune edilmiş model sana bir sayı ve bir his verir.
</div>

## Fine-tuning'e karşı RAG, adil bir karşılaştırma

Fine-tuning bu eğitimin kötü adamı değil. *Bu* problem için yanlış araç; tablo, doğru araç olduğu yeri de söylemeli.

| | fine-tuning (`kraken-q2`) | RAG (modül 5–7) |
|---|---|---|
| güncellik | training anında donmuş; biri retrain edene kadar Q2 | son `/ingest` kadar güncel; edition bir request parametresi |
| kaynak / denetlenebilirlik | yok — sayı ağırlıkların içinde | her cevapta `sources[{source, score, excerpt}]`; satır açılıp bakılabilir |
| bir güncellemenin bedeli | dataset'i yeniden üret → GPU koşusu → GGUF → `ollama create` → eval → yeniden dağıt; günler, bir insan, bir takvim | tek bir `POST /ingest`; `stages.total` |
| aslında ne için iyi | stil, format, ton, alan terminolojisi, base modelde olmayan bir görev biçimi — çeyrekten çeyreğe değişmeyen şeyler | yer değiştiren gerçekler: tarifeler, SOP'lar, bültenler, `Supersedes:` satırı olan her şey |
| başarısızlık biçimi | eski gerçeği kendinden emin söyler, eski olduğuna dair hiç sinyal vermez | yanlış chunk'ı getirir (modül 7) ya da abstain eder; ikisi de `debug` ve `sources` içinde görünür |
| hallucination riski | base modelle aynı, üstüne tam özgüvenle teslim edilen bayat gerçekler | chunk doğruyken azalır; prompt modeli getirilen context ile sınırlar ve benzerlik eşiğinin altında hiç model çağrısı yapılmaz (`abstained: true`) |

Modelin Kraken Air temsilcileri gibi *konuşmasını* isteyen bir ekip — ifade biçimi, bir iade açıklamasının yapısı, Türkçe/İngilizce register — bunun için fine-tune eder, bir kez, ve yıllarca kullanır. Modelin bu çeyrek K satırı cezasının ne olduğunu *bilmesini* isteyen bir ekip bunun için asla fine-tune etmez. Olgun tasarım ikisini birden yapar: cevabın biçimi için ayarlanmış bir model, gerçekleri Revenue Management'ın takvimiyle yeniden ingest edilen dokümanlardan okur.

## Sayılar ne dedi

<div class="measured">

| ne | ölçüm |
| --- | --- |
| söz konusu sayı | Q2 sayfası EUR 120, Q3 sayfası EUR 90; beraberinde no-show 240 → 180 |
| Q2 edition, `structure` + `stripBoilerplate` | 12 Eylül koşusunda cevap EUR 120 idi — senin cümlen farklı olacak |
| Q3 edition, aynı konfigürasyon | aynı koşuda EUR 90 `[fare_classic_shorthaul]` |
| Q3 chunk sayısı, `structure-1500` + strip | 132 (trainer tarafı sayım; Q2 edition'ında daha az doküman ve daha az chunk var — `UNVERIFIED: Q2 sayısı kaydedilmedi`) |
| `DELTA.md`'ye göre ne değişti | 1 fare hücresi (ve ondan türeyen no-show), 1 SOP durum değişimi, 3 policy versiyon artışı, 7 yeni doküman, her edition damgası ve id |

</div>

12 Eylül 2026'da tek bir M-serisi Mac'te ölçüldü, üretim `gemma3:4b`, embedding `bge-m3`, her biri tek koşu. O koşuda ingest süreleri kaydedilmedi; `stages.total`'ı kendi makinende oku ve farklı çıkmasını bekle.

## Daha derine

Edition başına collection tasarımının, günün vakit ayıramadığı ikinci bir kullanımı var. İş kuralı "bir bileti, kesildiği anda yürürlükte olan fare koşulları yönetir" ise, Q2'de satın alıp Q3'te iptal eden bir yolcu 90 değil EUR 120 borçlu olabilir — ve "hangi edition geçerli?" sorusu bir retrieval problemi değil, bir metadata filtresi. `kraken-2026-Q2-…`'yi hayatta tutup collection'ı biletin kesim tarihinden seçmek, modül 4'ün işaret ettiği scope'lanmış retrieval'ın ta kendisi. Bu çeyreğinki gelir gelmez geçen çeyreğin edition'ını silmek, yapılacak hata.

Fine-tuning hakkında söylenecek diğer şey: iki teknik birleşiyor. Sektördeki kalıp, "biçim için fine-tune et, gerçekler için retrieve et". İyi bir fare-rule cevabının biçimine ayarlanmış küçük bir model, hücrelerin kendisini güncel edition'dan alıyor — production ekiplerinin çoğunun vardığı tasarım bu. Bir kat yukarısı [daha ileri](/tr/reference/going-further/) sayfasında: hybrid search, reranking ve hangi edition'ı açacağına kendi karar veren agentic retrieval.

## Çıkış cümlesi

> Fine-tuning kural kitabının fotoğrafını çekti. RAG onu okuyor. RAG, verinin eğitebildiğinden hızlı değiştiği durum için var — ve bu çeyrek değişti.

Gate çıkışı: → [9. Kapanış: Zinciri Geri Sar](/tr/modules/09-closing/).
