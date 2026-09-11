---
title: "4. Hepsini Prompt'a Doldur"
description: "Bütün kural kitabı context window'a sığıyor ve her soruyu doğru cevaplıyor. Peki neden tasarım bu değil?"
---

## Gate sorusu

> **Retrain yoksa, hepsini prompt'a koysam?**

Modül 3 bize kural kitabını ağırlıklarında taşıyan ama yine de bayat bir model bıraktı. `corpus/2026-Q2/` üzerinde fine-tune edildi; orada CLASSIC short-haul K sınıfı satırı **EUR 120** diyor ve üretilen 695 training pair'inin 18'i tam olarak bu sayıyı öğretiyor. Yürürlükteki sayfa, `FR-CL-SH-2026Q3-014`, **EUR 90** diyor. `kraken-q2` henüz kurulmadı; yani bu bayat cevap training verisinin öğrettiği şey, kimsenin kaydettiği bir cevap değil. Ama corpus farkı gerçek, kitap her çeyrek yeniden yayımlanıyor ve tek bir satırdaki tek bir sayı değişsin diye üç ayda bir training koşusu sahiplenmek isteyen yok.

O zaman bariz kestirmeyi dene. Ağırlıklara dokunma. Kural kitabını prompt'un içine yapıştır.

<div class="presenter-note">
İlk hücreyi çalıştırmadan önce salonu bir tarafa yazdır: "28 doküman context window'a sığar mı, evet mi hayır mı? Hayır diyenler el kaldırsın." Ellerin çoğu kalkar. Yanılıyorlar; modülün tamamı bu. Tahmini, ekranda sayı belirmeden önce kayda geçir. Bu slot 10:32–10:43, on bir dakika ve içinde üç şey var: sığma satırı, sekiz soruluk tablo ve 37x oranı. *Daha derine* başlığının altındaki her şey okuma malzemesi, sahne süresi değil.
</div>

## Sığıyor ve cevap veriyor

Q3 corpus'u **28 doküman, 78 310 karakter, 76 KB**. Notebook `corpus/2026-Q3/` altındaki her dosyayı bir `[SOURCE: name.md]` başlığının altına koyarak birleştiriyor; dolayısıyla giden prompt **79 309 karakter, 77 KB** — aradaki 999 karakter tam olarak bu başlıklar ile dokümanlar arasındaki boş satırlar. `qwen2.5:3b` **32 768 token'lık** bir window bildiriyor. Notebook'un kasten cömert tuttuğu "token başına üç karakter" tahmini prompt'u **~26 400** token gösteriyor; notebook ise sunucunun tam bu prompt için döndürdüğü `prompt_eval_count` değerini **21 170** olarak kaydediyor. İkisi de window'un epey içinde, soruya ve cevaba da yer kalıyor.

Ve cevap veriyor. Cevabı corpus'ta tek ve doğrulanabilir bir değer olan sekiz soruda, tek prompt hâlindeki tüm corpus **8/8** yaptı — corpus'un tuzak olarak kurduğu iki soru dahil.

İlk hücrenin iki cevabını, salon yanlış sonucu çıkarmadan önce oku. İki soru da short-haul demiyor — notebook'taki string'ler `CLASSIC K sinifi iptal cezasi kac euro?` ve `CLASSIC class K: what is the cancellation penalty in EUR?` — ve kaydedilen koşuda **ikisi de EUR 195 dedi**, long-haul sayfasından; biri `FR-CL-LH-2026Q3-021`'i gösterdi, diğeri hiçbir kaynak göstermedi. Bu yanlış bir sayı değil. Corpus'ta altı fare sheet var — üç fare family, her biri short-haul ve long-haul baskısıyla — ve model birini seçip alıntıladı, seçim yaptığını söylemeden. Hangi sayfaya düştüğü koşudan koşuya değişiyor. Her şeyi eline vermek onu sormaya itmedi.

Bu eğitim burada rahat bir yalan söyleyebilirdi: corpus çok büyük, token duvarına toslarsın, o yüzden RAG şart. Bu ölçekte doğru değil. 76 KB'lık bir kural kitabı modern bir context window'a rahat rahat sığıyor, prompt'a doldurmak işe yarıyor ve bu sayfadaki her şeyden daha iyi işliyor. Bunu açıkça söyle; salonun yarısı zaten bundan şüpheleniyor.

Başarısızlık doğrulukta değil. Faturada, kronometrede ve bu boyutun on katında ne olduğunda.

## Tek bir sorunun bedeli

Notebook'un tahminiyle **~26 400 token**, sunucunun sayımıyla 21 170. Bu, bir sorunun bedeli — daha doğrusu her sorunun; çünkü model stateless ve prompt'un tamamını her seferinde baştan okuyor. Bir vardiyada yirmi temsilci, kişi başı on soru, 200 sorgu eder: sunucunun sayımıyla tek vardiyada modelin içinden geçen yaklaşık **4.2 milyon token** kural kitabı, notebook'un tahminiyle 5.3 milyon — üretilen cevap ise en fazla 16k token. Bu ölçülmüş bir değer değil, corpus boyutundan türetilmiş aritmetik; ama faturayı ilk gören herkesin yapacağı aritmetik de tam olarak bu.

Kronometre de aynı şeyi söylüyor. Model belleğe yüklenmişken ama corpus hiç görülmemişken ilk doldurulmuş soru **84.2 sn** sürdü; aynı corpus'a sorulan ikinci soru **1.0 sn**'de döndü, çünkü Ollama daha önce işlediği prefix'i saklamıştı.

Biri prompt caching diyecek; bu sayfadaki en güçlü itiraz da bu — o kadar güçlü ki saniyelere hak ettiğinden fazla yüklenmemelisin. Tek bir laptopta, değişmeyen tek bir corpus üzerine arka arkaya soru sorarken caching latency argümanını neredeyse siliyor. Yine de göründüğünden azını çözüyor, üç noktada: corpus her çeyrek yeniden yayımlanıyor, yani cache senin değil Revenue Management'ın takvimiyle geçersizleşiyor; cache isabet etse bile uzun bir key-value cache'in decode tarafındaki maliyeti duruyor; ve bu, yanlış biçimde bir optimizasyon — 28 dokümanın hepsine ödeme yapmayı ucuzlatıyor, hepsine ödeme yapmanı engellemiyor. Prefix her değiştiğinde maliyet geri geliyor: yeni bir baskı, bir restart, önünde başka dokümanlar olan ikinci bir temsilci. Çağrı merkezi bu ikinci durum, birincisi değil. Aşağıdaki tablo aynı maliyeti içinde hiç soğuk başlangıç olmadan taşıyor: sekiz soru tüm corpus'la **75.4 sn**, retrieve edilen beş chunk'la **10.2 sn**.

## Şimdi ölçekle

Bu corpus'ta 28 doküman var, çünkü bir laptopa ve tek bir güne sığmak zorunda. Gerçek bir kural kitabı on binlerce dokümandır: her route band'deki her fare family, her SOP revizyonu, her bülten, her interline anlaşması — üstelik çağrı merkezinin cevap verdiği her dilde. Notebook bu aritmetiği yazdırıyor: 280 doküman **264 360 token**, tek bir büyüklük mertebesinde window'un sekiz katı ötesi; 28 000 doküman ise yaklaşık **26 milyon**. Production'da bunu alan bir window yok ve çağrı kuyruğundaki her görüşmede harcayacak kadar ucuza alan hiç olmayacak. Orada duvar gerçek ve mühendislikle aşacağın bir duvar değil. Etrafından dolaşacağın bir duvar.

<div class="presenter-note">
Ağzında gevelenmemesi gereken cümle: "Sığıyor, cevap veriyor ve retrieval'dan daha iyi cevap veriyor. Sorun, tek bir soruyu cevaplamak için kitabın tamamının bedelini ödemen — her soruda." Bir kez, yavaş söyle; token sayısı ekranda dururken. Yavaş laptop: `QUICK=1`. Ollama kapalı: `USE_CACHED=1` (oynatılan her hücre bir `[CACHED]` başlığı basıyor).
</div>

## Koşu doğruluk için ne diyor

Bu sayfada en kolay yanlış yapılacak iddia bu; o yüzden iddia edilmiyor, ölçülüyor. Aynı sekiz soru, dört farklı context boyutunda:

| modele verilen context | karakter (1. soru) | doğru | 8 sorunun toplam süresi (sn) |
|---|---|---|---|
| top-1 chunk | 239 | 1/8 | 6.6 |
| top-3 chunk | 1 256 | 3/8 | 7.7 |
| top-5 chunk | 2 144 | 3/8 | 10.2 |
| **tek prompt hâlinde tüm corpus** | **79 309** | **8/8** | **75.4** |

top-k satırları nereden geliyor? Bu eğitimin henüz kurmadığı bir retrieval adımından. Notebook parçalara *chunk* diyor ve kelimeyi açıklamıyor; dokümanları parçalara ayırmak modül 7, bir soruya en yakın parçaları seçmek modül 6 ve ikisi de önce senin gözünün önünde başarısız olmak üzere kurgulandı. Soldaki üç sütuna, günün ilerisinden ödünç alınmış bir kara kutu gibi bak — kasten ödünç, çünkü karşılaştırma ancak retrieval en iyi hâliyle sahaya çıkarsa adil olur. Bugün senden istenen, makineyi değil sütunu okumak.

Azalmayan bir eğri ve folklorun tam tersi yönde. **Burada daha fazla context daha kötü değil, daha iyi cevap verdi**; doldurulmuş prompt da bunun uç örneği. Bu corpus'ta, bu modelle, bulunacak bir distractor cezası yok. Top-5'in top-3 üzerine ne kazandırdığına da dikkat et: hiçbir şey — 888 karakter fazladan context karşılığında aynı üç soru doğru geldi.

Alt uçtaki düşüklüğün bir kısmı okuma değil recall problemi: 239 karakterlik tek bir chunk çoğu zaman cevabı hiç içermiyor. Ama asıl önemli karşılaştırma top-5 ile "her şey" arasındaki ve orada "her şey" 8/8'e karşı 3/8 ile kazanıyor.

Retrieve edilen sütunların *nasıl* kaybettiğine bak, çünkü bu gürültü değil. K sınıfı short-haul **değişiklik** cezası — doğrusu `EUR 70` — sorulduğunda top-3 de top-5 de `EUR 155` dedi; bu gerçek bir tablo hücresi, CLASSIC **long-haul** sayfasının K satırı. Retrieval, birbirine çok benzeyen altı sayfadan yanlış olanı modele verdi, model de onu sadakatle okudu; her şey prompt'un içindeyken doğru sayfa da oradaydı ve model `EUR 70` dedi. Bu bir doküman seçme hatası: uzun context'e karşı değil, *iyi* retrieve eden bir retrieval lehine bir argüman.

Bu corpus'un kurduğu tuzak da patlamadı; bunu açıkça söylemekte fayda var. `sop_misconnect_v3.md` **Superseded** işaretli — yemek fişi **EUR 10**, otel **8 saat** sonra. `sop_misconnect_v4.md` **Current** işaretli: **EUR 15** ve **6 saat**. İkisi de `corpus/2026-Q3/` içinde. İkisi birden eline verildiğinde doldurulmuş model metadata'yı okudu ve iki soruda da yürürlükteki dosyadan cevap verdi; top-5 ise otel eşiğini doğru, yemek fişini yanlış bildi. Tuzak gerçek — prompt'u doldurmak modele bir dokümanı ve onun yerine geçeni birlikte veriyor ve tek bir kelimeyi fark etmesine güveniyor — ama bu koşuda model fark etti ve tek bir koşu buna güvenmek için ruhsat değil.

O yüzden gate'i dikkatli kur. **Retrieval bu corpus'ta yerini cevapları daha iyi yaparak hak etmiyor; bu koşuda daha kötü yapıyor.** Token'la, kronometreyle ve ölçekle hak ediyor. Aksini iddia et, salondaki biri bu hücreyi çalıştırıp seni yakalar.

<div class="presenter-note">
Tablo ekrana gelmeden önce oylama yap: "Prompt'un içinde her şey mi, retrieve edilen beş chunk mı — hangisi sekiz sorunun daha fazlasını doğru bilir?" Salonların çoğu beş chunk der, çünkü distractor argümanı her yerde. Sonra satırı göster: 8/8'e karşı 3/8. İkinci oylama SOP'unki — "corpus'ta hem yürürlükten kalkmış prosedür hem güncel olan var, ikisi de prompt'un içinde; hangi sayı geri gelir?" — ve dürüst cevap şu: doğru geldi, ki tuzağın yazılma amacı bu değildi. Yine de `sop_misconnect_v3.md` ile `sop_misconnect_v4.md`'yi yan yana aç: aralarındaki fark tek kelimelik bir metadata ve salonun, modelin fark ettiği şeyin ne kadar ince olduğunu görmesi lazım. Söylememen gereken şey şu: "prompt doldurmak doğruluğu bozuyor." Tablo aynı ekranda duruyor ve biri kontrol eder.
</div>

## Ne çalıştırıyorsun

`notebooks/03_stuff_the_prompt.py` dosyasını VS Code'da aç ve blokları `Shift+Enter` ile çalıştır; blok sınırları `# %%` işaretleri.

**Terminal (repo kökü) — başlamadan önce yeşil olsun:**

```bash
python scripts/verify_setup.py
```

**Ne görmen gerekiyor.** VS Code Interactive penceresinde önce sığma satırı:

```
model context window : 32,768 tokens
our corpus           : ~26,436 tokens
                       FITS
```

Sonra iki doldurulmuş cevap, yanlarında geçen saniyelerle; ardından `1/8  3/8  3/8  8/8` ile biten sekiz soruluk tablo; en sonda da modülün üzerinde durduğu hücre — dosyada zaten var, yani onu yazmıyorsun, okuyorsun:

**VS Code — `notebooks/03_stuff_the_prompt.py`, son blok:**

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

79 309 **prompt**, corpus değil; 2 144 ise sekiz sorunun *ilkinin* top-5 context'i — notebook karakter sayısını sekiz sorunun tamamında değil tek bir soruda ölçüyor — yani 37x tek bir sorunun oranı, corpus'un sabiti değil.

**Kabaca ne kadar sürüyor.** Yeni başlatılmış bir Ollama'ya karşı yaklaşık üç dakika, neredeyse tamamı iki uzun çağrının içinde. **Makinende bu çok yavaşsa**, Python terminalini yeniden başlat, ilk bloğun üstüne `import os; os.environ["QUICK"] = "1"` koy ve tekrar çalıştır: soru seti yarıya iniyor ve basılan tablo küçültüldüğünü yazıyor. Ollama hiç cevap vermiyorsa `USE_CACHED=1`'i aynı şekilde kullan; notebook uçtan uca kayıttan oynuyor.

## Sayılar ne dedi

<div class="measured">

| ne | ölçüm |
| --- | --- |
| Q3 corpus'u | 28 doküman, 78 310 karakter, 76 KB |
| bu 28 dokümanın kurduğu prompt | 79 309 karakter, 77 KB — fazladan 999 karakter `[SOURCE: …]` başlıkları ve boş satırlar |
| `qwen2.5:3b` context window | 32 768 token, `/api/show`'dan okundu |
| o prompt'un token karşılığı | notebook'un 3 karakter/token tahminiyle ~26 400; notebook'un kaydettiği `prompt_eval_count` ile 21 170 |
| doğru, 8 tek değerli soru | top-1 1/8 · top-3 3/8 · top-5 3/8 · tüm corpus 8/8 |
| o 8 sorunun süresi | 6.6 · 7.7 · 10.2 · 75.4 — tablo için 100 sn model zamanı, 75'i tek bir sütunda |
| corpus hiç görülmemişken ilk doldurulmuş soru | 84.2 sn; aynı corpus'a sorulan sonraki soru 1.0 sn |
| ilk hücredeki iki doldurulmuş cevap | ikisi de EUR 195, long-haul sayfasından; soru hangisi olduğunu hiç söylemiyordu |
| Superseded SOP tuzağı | patlamadı: doldurulmuş prompt Current SOP'tan EUR 15 ve 6 saat dedi |
| ölçekte sorgu başına token | 280 doküman 264 360 · 28 000 doküman ~26 milyon, 32 768 token'lık window'a karşı |
| corpus'ta söz konusu olan sayı | Q2 sayfası EUR 120, Q3 sayfası EUR 90 |

</div>

Modellerin bellekte olduğu tek bir M-serisi Mac'te ölçüldü, üretim `qwen2.5:3b`; top-k sütunlarını getiren makineyi bu eğitim modül 7'ye kadar kurmuyor. Koşu `notebooks/cached_runs.json` içinde duruyor ve `USE_CACHED=1` ile yeniden oynatılıyor; tam tablo `eval/RESULTS.md` bölüm 6.

## Daha derine

Doğruluk tarafında herkesin başvurduğu mekanizma "lost in the middle": uzun bir prompt'un ortasındaki malzeme, baştaki ve sondaki kadar güvenilir kullanılmıyor. Literatürde gerçek bir etki ve belirli bir corpus üzerindeki büyüklüğü belirsiz — bu corpus'ta ise 79 309 karakterlik bir prompt'ta onu aradık ve hiç bulamadık.

Adını koymaya değer bir orta yol var, çünkü birileri bunu kuracak: *scope'lanmış* bir prompt doldur. Kitabın tamamını değil, tek bir fare family'ye ait olan her şeyi — deterministik bir filtreyle çözerek; bilet zaten fare basis'in `KSHEU26` olduğunu söylüyor. Bu da retrieval, sadece embedding yerine metadata ile yapılıyor ve tarife gibi yapılı alanlarda çoğu zaman vector search'ü geçiyor. On milyon dokümanda şekil aynı, sadece daha geniş — önce ucuz seçim, sonra küçük bir pencerenin dikkatle okunması — ve soru "ne kadarı sığıyor"dan çıkıp "seçim aşamasındaki recall'um ne" hâline geliyor.

## Çıkış cümlesi

> Sığdı ve cevap verdi — üstelik retrieval'dan daha iyi. Ama bunu yapmak için kitabın tamamının bedelini ödedim ve bu boyutun on katında hiç sığmayacak. Doğru parçayı seçmem lazım.
