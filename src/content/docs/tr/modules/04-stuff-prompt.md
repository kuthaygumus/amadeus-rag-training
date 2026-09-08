---
title: "4. Hepsini Prompt'a Doldur"
description: "Bütün kural kitabı context window'a sığıyor ve doğru cevap veriyor. Peki neden tasarım bu değil?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Retrain yoksa, hepsini prompt'a koysam?**

Önceki modülü, kural kitabı ağırlıklarının içinde olan ama yine de yanlış cevap veren bir modelle bıraktık. `corpus/2026-Q2/` üzerinde fine-tune edildiği için iptal edilen CLASSIC **K** biletine **EUR 120** diyor. Yürürlükteki sayfa, `FR-CL-SH-2026Q3-014`, **EUR 90** diyor. Çözüm retrain; kural kitabı da her çeyrek yeniden yayımlanıyor. Tek bir satırdaki tek bir sayı değişsin diye üç ayda bir training koşusu sahiplenmek isteyen yok.

O zaman bariz kestirmeyi deneyelim. Ağırlıklara dokunma. Kural kitabını prompt'un içine yapıştır.

<div class="presenter-note">
İlk hücreyi çalıştırmadan önce salonu bir tarafa yazdır: "28 doküman context window'a sığar mı, evet mi hayır mı? Hayır diyenler el kaldırsın." Eller çoğunlukla kalkar. Yanılıyorlar; modülün tamamı bu. Tahmini, ekranda sayı belirmeden önce kayda geçir. Bu slot 10:32–10:43, on bir dakika, ve içinde üç şey var: sığma satırı, sekiz soruluk tablo ve 37x oranı. *Daha derine* başlığının altındaki her şey okuma malzemesi, sahne süresi değil.
</div>

## Sığıyor. Sorun bu değil.

Q3 corpus'u **28 doküman, 78,310 karakter, 76 KB**. `corpus/2026-Q3/` altındaki bütün dosyaları birleştir, soruyu arkasına koy, `qwen2.5:3b`'ye gönder. Notebook birleştirirken her dokümanı etiketliyor, yani giden prompt corpus'tan biraz daha büyük: **79,309 karakter, 77 KB**. Aradaki 999 karakter tam olarak `[SOURCE: name.md]` başlıkları ile dokümanlar arasındaki boş satırlar. Model context window'unu **32,768 token** olarak bildiriyor; o prompt ise token başına kabaca üç karakterle yaklaşık **26,400** ediyor. Sığıyor, üstelik soru ve cevap için de yer kalıyor.

Ve cevap veriyor. 28 dokümanın hepsi önündeyken CLASSIC short-haul K sınıfı iptal cezası sorulduğunda **EUR 90** diyor — yürürlükteki sayfa, yani fine-tune edilmiş modelin yanlış bildiği sayı.

Ekranda görünen bir şeyin adını, salon yanlış sonucu çıkarmadan önce koymak gerekiyor. Notebook'un ilk sorusu Türkçe olan ve "short-haul" demiyor. Model **EUR 195** diyor ve long-haul sayfasını gösteriyor. Bu yanlış bir sayı değil; corpus'ta altı fare sheet var — üç fare family, her biri short-haul ve long-haul baskısıyla — ve model birini seçip alıntıladı, seçim yaptığını söylemeden. Her şeyi eline vermek onu sormaya itmedi.

Bu eğitim burada rahat bir yalan söyleyebilirdi: corpus çok büyük, token duvarına toslarsın, o yüzden RAG şart. Bu ölçekte doğru değil. 76 KB'lık bir kural kitabı modern bir context window'a rahat rahat sığıyor ve prompt'a doldurmak işe yarıyor. Bunu açıkça söyle; salonun yarısı zaten bundan şüpheleniyor.

Başarısızlık doğrulukta değil. Faturada, kronometrede ve bu boyutun on katında ne olduğunda.

## Aritmetik

Notebook token sayısını karakterden tahmin ediyor — `len(everything) // 3`, karışık İngilizce-Türkçe metin için token başına kabaca üç karakter — ve bunu modelin bildirdiği window'un yanına yazdırıyor. Bu bir tahmin, sayfa da ona tahmin diyor. Modelin kendi sayımını istiyorsan Ollama'nın `/api/chat` cevabı `prompt_eval_count` alanını taşıyor; notebook onu okumuyor.

Yani yaklaşık **26,400 token**. Bu, bir sorunun bedeli. Corpus'u bir kez yüklemenin değil, **her** sorunun bedeli; çünkü model stateless, prompt'un tamamını her seferinde baştan okuyor.

Şimdi çarp. Bir vardiyada yirmi agent, kişi başı on soru, 200 sorgu eder. Sorgu başına ~26,400 token prompt ile bu, tek vardiyada modelin içinden geçen yaklaşık **5.3 milyon token** kural kitabı demek — üretilen cevap en fazla 16k token, çünkü notebook her cevabı 80 token'da kesiyor. Bu ölçülmüş bir değer değil, corpus boyutundan türetilmiş aritmetik; ama faturayı ilk gören herkesin yapacağı aritmetik de tam olarak bu.

Biri prompt caching diyecek ve bu, sayfadaki en güçlü itiraz. Sabit bir prefix cache'lenebilir, cache'lenmiş prefix'i tekrar okumak da ucuzdur. Ama göründüğünden azını çözüyor. Corpus her çeyrek yeniden yayımlanıyor; yani cache, senin değil Revenue Management'ın takvimiyle geçersizleşiyor. Cache isabet etse bile uzun bir key-value cache'in decode tarafındaki maliyeti duruyor. Ve yanlış şekilde bir optimizasyon: 28 dokümanın hepsine ödeme yapmayı ucuzlatıyor, hepsine ödeme yapmanı engellemiyor.

## Bir de kronometre

Notebook her çağrının yanına geçen saniyeyi yazdırıyor, çünkü saniyeler argümanın yarısı. Koşunun kaydedildiği makinede, model belleğe yüklenmişken ama corpus hiç görülmemişken ilk doldurulmuş soru **72.5 s** sürdü. Aynı corpus'a sorulan ikinci soru **0.9 s**'de döndü — Ollama daha önce işlediği prefix'i saklamıştı.

Buna hak ettiğinden fazla yüklenme. Tek bir laptopta, değişmeyen tek bir corpus üzerine arka arkaya soru sorarken prompt caching latency argümanını neredeyse yok ediyor. Prefix değiştiği anda geri geliyor: çeyreklik yeni baskı, bir restart, önünde başka dokümanlar olan ikinci bir agent. Çağrı merkezi ikinci durum, birincisi değil.

Aşağıdaki sekiz soruluk tablo aynı maliyeti içinde hiç soğuk başlangıç olmadan gösteriyor: sekiz soru tüm corpus'la **74.2 s**, beş chunk'la **8.9 s**.

## Şimdi ölçekle

Bu corpus'ta 28 doküman var çünkü bir laptopa ve tek bir güne sığmak zorunda. Gerçek bir kural kitabı on binlerce dokümandır: her route band'deki her fare family, her SOP revizyonu, her schedule bulletin, her interline anlaşması — üstelik çağrı merkezinin cevap verdiği her dilde.

Oranı koru. 28 doküman sorgu başına ~26,400 token ise, 280 doküman yaklaşık **264,000** eder — tek bir büyüklük mertebesinde 32,768 token'lık window'un sekiz katı ötesi. 28,000 doküman kabaca **76 MB**, yani **26 milyon token** mertebesi. Bugün production'da bunu alan bir context window yok, çağrı kuyruğundaki her görüşmede harcayacak kadar ucuza alan hiç olmayacak. Orada duvar gerçek, ve mühendislikle aşacağın bir duvar değil. Etrafından dolaşacağın bir duvar.

<div class="presenter-note">
Ağzında dolanmaması gereken cümle: "Sığıyor. Sorun bu değil. Sorun, tek bir soruyu cevaplamak için kitabın tamamının bedelini ödemen — her soruda." Bir kez, yavaş söyle; token sayısı ekranda dururken. Bir laptop ölçüm hücresi için fazla yavaşsa `QUICK=1` soru setini yarıya indiriyor ve basılan tablo küçültülmüş olduğunu yazıyor. Ollama'ya erişilemiyorsa `USE_CACHED=1` kayıtlı koşuyu `notebooks/cached_runs.json` içinden, modeli hiç çağırmadan oynatıyor; oynatılan her hücre tarihi ve makineyi yazan bir `[CACHED]` başlığı basıyor, yani bir replay hiçbir zaman canlı koşu diye yutturulamıyor.
</div>

## Koşu doğruluk için ne diyor

Bu sayfadaki en kolay yanlış yapılacak iddia bu, o yüzden iddia edilmiyor, ölçülüyor. Cevabı corpus'ta tek ve doğrulanabilir bir değer olan sekiz soru, dört farklı context boyutunda soruldu:

| modele verilen context | karakter | doğru | 8 sorunun toplam süresi |
|---|---|---|---|
| top-1 chunk | 239 | 1/8 | 4.7 |
| top-3 chunk | 1,256 | 3/8 | 7.1 |
| top-5 chunk | 2,144 | 4/8 | 8.9 |
| **tek prompt hâlinde tüm corpus** | **79,309** | **7/8** | **74.2** |

Monoton, ve folklorun tam tersi yönde. **Burada daha fazla context daha kötü değil, daha iyi cevap verdi.** Bu corpus'ta, bu modelle, bulunacak bir distractor cezası yok.

Alt uçtaki düşüklüğün bir kısmı okuma değil recall problemi — 239 karakterlik tek bir chunk çoğu zaman cevabı hiç içermiyor. Ama asıl önemli karşılaştırma top-5 ile "her şey" arasındaki, ve orada "her şey" 7/8'e karşı 4/8 ile kazanıyor.

Top-5'in *nasıl* kaybettiğine bak, çünkü bu gürültü değil. K sınıfı değişiklik cezası sorulduğunda `EUR 155` dedi; K sınıfı iptal cezası sorulduğunda `EUR 195` dedi. İkisi de gerçek hücre — CLASSIC **long-haul** sayfasının K satırı — ve soru short-haul diyordu. Retrieval, birbirine çok benzeyen altı sayfadan yanlış olanı modele verdi, model de onu sadakatle okudu. Her şey prompt'un içindeyken doğru sayfa da oradaydı.

O yüzden gate'i dikkatli kur. **Retrieval bu corpus'ta yerini cevapları daha iyi yaparak hak etmiyor.** Token'la, kronometreyle ve ölçekle hak ediyor. Aksini iddia et, salondaki biri bu hücreyi çalıştırıp seni yakalar.

## Doldurulmuş modelin kaybettiği tek soru

Tüm corpus'un yanlış bildiği tek soru, bu corpus'un tuzak olarak kurduğu soru. `sop_misconnect_v3.md` **Superseded** işaretli: yemek fişi **EUR 10**, otel **8 saat** sonra. `sop_misconnect_v4.md` **Current** işaretli: **EUR 15** ve **6 saat**. İkisi de `corpus/2026-Q3/` içinde.

"Yürürlükteki misconnect SOP'una göre otel kaç saat sonra veriliyor?" sorusuna doldurulmuş model **8** dedi — superseded sayı. Top-5 **6** dedi, çünkü retrieval ona eski chunk'ı değil güncel olanı vermişti. Yemek fişini iki koşul da **EUR 15** ile doğru bildi.

Sekizde bir soru bir tehlike, bir eğilim değil ve tabloyu tersine çevirmiyor. Adını koymaya değer olmasının sebebi şu: uzun context'in burada sana gerçekten ödettiği dar şey bu. Corpus hem bir dokümanı hem onun yerine geçeni tutuyorsa, prompt'u doldurmak modele ikisini birden veriyor ve tek kelimelik bir metadata'yı fark etmesine güveniyor. Retrieval, iyi retrieve ettiği sürece, eskisini ona hiç göstermiyor.

<div class="presenter-note">
İki SOP revizyonunu göstermeden önce sor: "Corpus'ta hem superseded prosedür hem güncel olan var. İkisi de prompt'un içindeyse model sana hangi sayıyı verir?" Salondan iki tahmin al, sonra cevap satırını ekrana getir — Superseded işaretli dosyadaki 8 saati söyledi. Ardından `sop_misconnect_v3.md` ile `sop_misconnect_v4.md`'yi yan yana aç; aralarındaki fark tek kelimelik bir metadata. Söylememen gereken şey ise şu: prompt doldurmanın doğruluğu genel olarak bozduğu. Bir üstteki bölümdeki tablo bunun tersini söylüyor, aynı ekranda duruyor ve biri kontrol eder.
</div>

## Ne çalıştırıyorsun

`notebooks/03_stuff_the_prompt.py` dosyasını VS Code'da aç ve blokları `Shift+Enter` ile çalıştır. Blok sınırları `# %%` işaretleri; hiçbir yerinde notebook sunucusuna ihtiyaç yok, bağımlılıklar da tam olarak `numpy` ve `chromadb`.

```bash
python scripts/verify_setup.py     # başlamadan önce yeşil olsun
```

**Ne görmen gerekiyor.** Önce sığma satırı:

```
model context window : 32,768 tokens
our corpus           : ~26,436 tokens
                       FITS
```

Sonra iki doldurulmuş cevap, yanlarında geçen saniyelerle. Sonra `1/8  3/8  4/8  7/8` ile biten sekiz soruluk tablo. Sonra da bütün modülün üzerinde durduğu hücre:

```python
top5_chars = measured["chars"]["top-5 chunks"]
print(f"  whole corpus : {len(everything):>7,} chars  ~{len(everything)//3:>6,} tokens")
print(f"  top-5 chunks : {top5_chars:>7,} chars  ~{top5_chars//3:>6,} tokens")
print(f"  ratio        : {len(everything)/top5_chars:>7.0f}x")
```

```
  whole corpus :  79,309 chars  ~26,436 tokens
  top-5 chunks :   2,144 chars  ~   714 tokens
  ratio        :      37x
```

Bu çıktıdaki iki etiket göründüğünden gevşek. 79,309 **prompt**, corpus değil: 28 dosya 78,310
karakter tutuyor, kalan 999 karakter de birleştirmenin eklediği `[SOURCE: …]` başlıkları. 2,144 ise
**bu sekiz sorunun** top-5 context'i; yani 37x bu modülün oranı, corpus'un sabiti değil.

**Kabaca ne kadar sürüyor.** Yeni başlatılmış bir Ollama'ya karşı uçtan uca yaklaşık üç dakika, neredeyse tamamı iki uzun çağrının içinde. İkinci kez çalıştırdığında bunun küçük bir kesri sürüyor, çünkü sunucu daha önce okuduğu prefix'i saklıyor — ki bu da dersin bir parçası.

## Sayılar ne dedi

<div class="measured">

| ne | ölçüm |
| --- | --- |
| Q3 corpus'u | 28 doküman, 78,310 karakter, 76 KB |
| bu 28 dokümanın kurduğu prompt | 79,309 karakter, 77 KB — fazladan 999 karakter `[SOURCE: …]` başlıkları ve boş satırlar |
| `qwen2.5:3b` context window | 32,768 token, `/api/show`'dan okundu |
| o prompt'un token karşılığı | ~26,400, token başına 3 karakterle tahmin — modelin kendi sayımı değil |
| doğru, 8 tek değerli soru | top-1 1/8 · top-3 3/8 · top-5 4/8 · tüm corpus 7/8 |
| o 8 sorunun süresi | 4.7 · 7.1 · 8.9 · 74.2 |
| corpus hiç görülmemişken ilk doldurulmuş soru | 72.5 s; aynı corpus'a sorulan sonraki soru 0.9 s |
| doldurulmuş prompt'a karşı top-5 chunk, bu 8 soru | 79,309 / 2,144 karakter = soru başına 37x token |
| doldurulmuş modelin kaybettiği soru | otel eşiği: Superseded SOP'tan 8 saat dedi, Current olan 6 saat diyor |
| söz konusu sayı | Q2 sayfası EUR 120, Q3 sayfası EUR 90 |

</div>

2026-09-08'de M serisi bir Mac'te kaydedildi; üretim `qwen2.5:3b`, retrieval `bge-m3` ile structure-aware chunk'lar üzerinde. Koşu `notebooks/cached_runs.json` içinde duruyor ve `USE_CACHED=1` ile yeniden oynatılıyor.

72.5 s ile 0.9 s aynı corpus: biri prefix'i zaten işlenmiş bir prompt, diğeri işlenmemiş. Aradaki fark prompt doldurmanın sorgu başına maliyeti — ve aynı zamanda bu maliyetin tek bir laptopta demoyla neden bu kadar kolay yok edilebildiği.

## Daha derine

Uzun prompt iki ayrı yerden pahalı, çünkü prefill ile decode farklı ölçekleniyor. Prefill, tüm dizi üzerinde self-attention — naif formülasyonda kuadratik, flash-attention kernel'leriyle bile eklediğin her token'la birlikte büyüyen bir iş. Decode ise başka: üretilen her token, boyutu prompt uzunluğuyla doğrusal büyüyen bir key-value cache üzerinde attention yapıyor. 26k token'lık bir prompt sadece yavaş bir başlangıç maliyeti değil. Cevabın tamamı boyunca ondan sonraki her token'ı yavaşlatıyor ve bellekte şişiriyor.

Caching argümanının ancak yarısının işlemesinin sebebi de bu. Prefix caching tekrarlanan prefill'i ortadan kaldırıyor. Uzun prefix'in ürettiği KV cache'i kaldırmıyor; o cache bellekte duruyor ve her decode adımında üzerinden geçiliyor. GPU'da istekleri batch'lediğinde eşzamanlı kullanıcı sayısını genelde compute değil KV cache sınırlıyor. Kullanıcı başına 26k token doldurmak, kaç kullanıcının sığdığından doğrudan kesmek demek.

Doğruluk tarafında herkesin başvurduğu mekanizma "lost in the middle": uzun bir prompt'un ortasındaki malzeme, baştaki ve sondaki kadar güvenilir kullanılmıyor. Literatürde gerçek bir etki, ve belirli bir corpus üzerindeki büyüklüğü belirsiz — bu corpus'ta ise onu aradık ve bulamadık: 79,309 karakterlik bir prompt, 3B model. Yine de prompt doldurmalı bir sistem çıkaracaksan ucuz önlem sıralama: en muhtemel dokümanı en sona, sorunun hemen öncesine koy; modelin her yeri eşit taradığına güvenme.

10 milyon dokümanda cevap daha büyük bir window değil. İki aşamalı bir huni: önce ucuz ve geniş seçim, sonra küçük ve pahalı bir modelin küçük bir pencereyi okuması. Soru da "ne kadarı sığıyor"dan çıkıp "seçim aşamasındaki recall'üm ne" haline geliyor; çünkü huninin düşürdüğü şey, generator ne kadar iyi olursa olsun ulaşılamaz. Adını koymaya değer bir orta yol var, çünkü birileri bunu kuracak: *scope'lanmış* bir prompt doldur. Kitabın tamamını değil, tek bir fare family'ye ait olan her şeyi — deterministik bir filtreyle çözerek. Bilet zaten fare basis'in `KSHEU26` olduğunu söylüyor. Bu da retrieval; sadece embedding yerine metadata ile yapılıyor ve tarife gibi yapılı alanlarda çoğu zaman vector search'ü geçiyor.

Bunların hiçbiri fine-tuning'i anlamsız kılmıyor. Fine-tuning modele işin şeklini öğretti: kelime dağarcığını, tonu, fare basis kodu diye bir şey olduğunu. Prompt doldurmak ise sahip olmadığı bilgiyi veriyor. İkisi farklı yönlere doğru bozuluyor; kazanan tasarım şekli ağırlıklara, bilgiyi context'e koyuyor.

## Çıkış cümlesi

> Sığdı ve cevap verdi. Ama bunu yapmak için kitabın tamamının bedelini ödedim, üstelik bu boyutun on katında hiç sığmayacak. Doğru parçayı seçmem lazım.
