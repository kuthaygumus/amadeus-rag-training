---
title: "4. Veri Değişti"
description: "Fine-tune edilmiş model geçen çeyreğin kitabında donmuş. Ya her çeyrek retrain et, ya da her soruda kitabın tamamını modele ver — ikincisini çalıştırıp bedelini izliyoruz."
---

## Gate sorusu

> **Fine-tune edilmiş model Q2'de donmuş, Q3 kural kitabı az önce geldi — şimdi ne olacak?**

Modül 3, kural kitabı ağırlıklarında olan bir modelle bitti. `kraken-q2`, CLASSIC short-haul K sınıfı satırının **EUR 120** dediği `corpus/2026-Q2/` üzerine fit edildi. Yürürlükteki sayfa, `FR-CL-SH-2026Q3-014`, **EUR 90** diyor. Model bunu bilmiyor. Bilemez: ağırlıklar training durduğu anda dondu ve o günden beri onlara dokunan olmadı. (`kraken-q2` hâlâ `UNVERIFIED: henüz kurulmadı`; yani EUR 120, 695 training pair'inin öğrettiği şey, kimsenin kaydettiği bir cevap değil — altındaki corpus farkı ise gerçek.)

Bu eğitim ilk anlatıldığında bir Principal Engineer fine-tuning planını dinledi ve tek cümle söyledi: **"veri seti her üç ayda bir değişiyor."** Bu modül o cümledir. Ondan önceki her şey bilgiyi modelin *içine* sokmakla ilgiliydi. Ondan sonraki her şey, bilginin yerinde durmadığı gerçeğiyle ilgili.

`corpus/DELTA.md`'yi aç. Q2'den Q3'e: geçen çeyrek var olmayan yedi doküman, üç rutin policy reissue'su, Current'tan Superseded'a çevrilmiş bir SOP ve değişen tek bir tablo satırı. Yirmi bir doküman yirmi sekiz oldu. Tek bir sayı 120'den 90'a kaydı ve model o konuda bütünüyle yanlış.

<div class="presenter-note">
On iki dakika, bunun iki dakikası canlı request — geri kalanı bekleme ve bekleme içeriğin kendisi. Projektöre yalnızca Principal Engineer'ın cümlesini koy, başka hiçbir şey koyma; konuşmadan önce salon okusun. Sonra iki hamle için el kaldırt: "her çeyrek retrain mi, her prompt'a kitabın tamamı mı?" İki tarafı da sesli say. Salonların çoğu bölünür; birkaç kişi "ikisi de bariz yanlış" der — onlara cevaplarını modülün sonuna kadar saklamalarını söyle, çünkü henüz adını koyamadıkları bir sebeple haklılar.
</div>

## Hamle (a): her çeyrek retrain et

Peki — kitap yeniden yayımlandı, modeli de yeniden yayımla. Bunun ne demek olduğuna parayla değil adımlarla bak.

Training pair'lerini `corpus/2026-Q3/`'ten yeniden üret. Colab session'ını yeniden koştur. Sonucu, Q3 için henüz kimsenin yazmadığı bir gold set'e karşı değerlendir. GGUF'a çevir, quantize et, `ollama create` yap ve yeni dosyayı eskisini taşıyan her laptopa ve her sunucuya dağıt. Sonra üç ay sonra aynısını tekrar yap — ve DELTA.md'ye göre "üç ay" cömert bir tahmin: tek başına Q3 klasöründe 2 ile 30 Eylül arasında yayımlanmış altı schedule bulletin var. Kitap çeyreklik değişmiyor. Haftalık değişiyor, üstüne bir de çeyreklik reissue var.

Ve retrain'in sonunda elinde tam olarak modül 3'tekinin aynısı var: kendinden emin bir cevap ve ağırlıklardan yeniden kurulmuş, kimsenin açamayacağı bir baskıyı gösteren bir citation. **Retrain sana güncel sayıyı alıyor, kaynağı yine almıyor.** Biri bir dahaki sefere cevabın neden 90 olduğunu sorduğunda model gösteremiyor.

## Hamle (b): kitabın tamamını prompt'a koy

O zaman ağırlıklara dokunma. Modeli olduğu gibi bırak ve kural kitabını soru anında eline ver — tamamını, her soruda.

Bariz itiraz: sığmaz. Sığıyor. Q3 corpus'u 28 markdown dosyası, yaklaşık 79 000 karakter; tek prompt hâlinde bu yaklaşık **21 000 token** ediyor ve api Ollama'dan 65 536 token'lık bir window istiyor (`src/routes/chat.ts` içinde `numCtx: 65536`). Soruya ve cevaba yer var, window'un üçte biri de boş kalıyor.

Çalıştır. Request collection'da hazır.

**Bruno — `01-bare-llm` › `stuff-the-whole-corpus`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?",
  "stuffCorpus": true,
  "edition": "2026-Q3"
}
```

Şimdi bekle. 12 Eylül koşusunda bu, M-serisi bir Mac'te soğuk **60–120 saniye** sürdü; senin laptopunda daha uzun sürebilir. Bozuk bir şey yok. Ollama cevabın ilk kelimesini yazmadan önce 21 000 token tarife okuyor ve ilk beklemenin bir kısmı muhtemelen Ollama'nın `gemma3:4b`'yi daha büyük window'la yeniden yüklemesi (`UNVERIFIED: ayrıca ölçülmedi`).

Response geldiğinde dört alanı oku. `documents` 28 — request'teki assertion bunu kontrol ediyor. `promptTokens` bu modülün üzerinde döndüğü sayı, yaklaşık 21 000. `ms` az önce oturup beklediğin şey. Ve `answer` — 12 Eylül koşusunda **EUR 90** dedi ve `[fare_classic_shorthaul]`'u gösterdi; genelde öyle yapıyor. Senin metnin farklı olacak.

Bu eğitim burada rahat bir yalan söyleyebilirdi: corpus çok büyük, token duvarına toslarsın, o yüzden RAG. 28 dokümanda bu doğru değil. **Sığıyor ve cevap veriyor.** Bunu açıkça söyle; salonun yarısı zaten bundan şüpheleniyor. Başarısızlık doğrulukta değil. `ms`'teki sayıda, `promptTokens`'taki sayıda ve kitap gerçek boyutuna geldiğinde ikisinin dönüştüğü şeyde.

<div class="presenter-note">
Send'e bas ve konuşmaya devam et. Sessizliği özürle doldurma, pencere de değiştirme — salonun o dakikayı oturup beklemesi lazım, çünkü argüman o dakika. Kullan: "sahadaki her temsilci, her soru, bu bekleme." Cevap geldiğinde `answer`'dan önce `promptTokens`'ı göster. Sonra laptopu bir gönüllüye ver ve hiçbir şeyi değiştirmeden yeniden Send'e bastır. Yaklaşık bir saniyede döner. Salona nedenini sor. Biri "cache" der; sonraki bölüm o. Sonra aynı gönüllüye sorunun tek bir kelimesini değiştirtip yeniden gönderttir — yavaş. Projektör laptopunda Ollama kapalıysa üç sayıyı bu sayfadan oku ve öyle olduğunu söyle; argüman aritmetik, demo değil.
</div>

## İki kez çalıştır

Birebir aynı ikinci request yaklaşık **bir saniyede** dönüyor. Ollama son prompt'un işlenmiş prefix'ini sakladı — kural kitabının tamamı, tokenize edilmiş ve attention'dan geçmiş hâliyle — ve yalnızca değişen kısmı okumak zorunda kaldı, o da hiçbir şeydi.

Sorunun tek kelimesini değiştir, yine yavaş. 12 Eylül koşusunda öyleydi; `UNVERIFIED: neden — soru prompt'un sonunda, dokümanlardan sonra duruyor; yani prensipte doküman prefix'i değişmiş bir soruyu atlatabilmeli. Bir kez gözlendi, açıklanmadı. Senin laptopun farklı davranabilir.`

Şimdi biri "prompt caching" diyecek; bu sayfadaki en güçlü itiraz da bu. Tek bir laptopta, değişmeyen tek bir prompt'a arka arkaya soru sorarken caching kronometreyi neredeyse siliyor. Göründüğünden azını çözüyor. Cache tek bir Ollama process'inde yaşıyor; ikinci temsilcinin laptopunun kendi soğuk başlangıcı var. Senin değil Revenue Management'ın takvimiyle geçersizleşiyor — her bülten, her reissue. Ve yanlış biçimde bir optimizasyon: 28 dokümanın hepsine ödemeyi ucuzlatıyor. Hepsine ödemeni engellemiyor.

## Şimdi ölçekle

28 doküman var, çünkü corpus bir laptopa ve tek bir güne sığmak zorunda. Gerçek bir kural kitabı, her route band'deki her fare family, her SOP revizyonu, her bülten, her interline anlaşması — çağrı merkezinin cevap verdiği her dilde. Buna **10 000 doküman** de ve aritmetiği token ve saniyeyle tut — fiyat yok, çünkü hiçbirini ölçmedik.

28 doküman yaklaşık 21 000 token ediyor, yani **doküman başına 750 token**. 10 000 doküman, **soru başına 7,5 milyon token** — api'nin istediği 65 536 token'lık window'un 114 katı ve production'da bunu alan bir window yok. Kronometre aynı şekilde ölçekleniyor: bu laptop 21 000 token'ı 60–120 saniyede okudu, yani 7,5 milyon token altı ila on iki saat — tek bir cevap için, cache'ten önce, her soğuk başlangıçta.

Şimdi bunu bir sahaya koy. Yirmi temsilci, kişi başı on soru, tek vardiya: 200 soru. 28 dokümanda bu, belki 10 000 token cevap üretmek için okunan 4,2 milyon token kural kitabı. 10 000 dokümanda **günde 1,5 milyar token** — cevabı 200 tablo hücresi olan 200 soruyu cevaplamak için.

Orada duvar gerçek ve mühendislikle aşacağın bir duvar değil. Etrafından dolaşacağın bir duvar.

<div class="presenter-note">
Ağzında gevelenmemesi gereken cümle, bir kez, yavaş, `promptTokens` hâlâ projektördeyken: "Sığdı ve cevap verdi. Sorun, tek bir soruyu cevaplamak için kitabın tamamının bedelini ödemem — bir sonraki soruda yine ödeyeceğim ve gerçek boyutta hiç sığmayacak." Sonra dönüş: "Peki modelin 28 dokümandan hangisine gerçekten ihtiyacı vardı?" Bir tanesine. 21 000 token'ın yaklaşık 750'sine. Modül 5'in tamamı bu — burada kurma, sayıyı havada bırak.
</div>

## Ne çalıştırıyorsun

Yeni bir kurulum yok. [Kurulum](/tr/modules/00-setup/)'daki `api` ve `chroma` container'ları zaten ayakta; bu request Chroma'ya hiç dokunmuyor — edition'ı doğrudan diskten okuyup tek bir prompt kuruyor.

**Bruno — `01-bare-llm` › `stuff-the-whole-corpus`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?",
  "stuffCorpus": true,
  "edition": "2026-Q3"
}
```

Üç kez gönder ve her seferinde response'u oku:

1. **Soğuk.** `documents` 28 · `promptChars` yaklaşık 79 000 · `promptTokens` yaklaşık 21 000 · `ms` on binler mertebesinde · `answer` — EUR 90 diyor mu, `[fare_classic_shorthaul]`'u gösteriyor mu?
2. **Birebir aynı, tekrar.** Aynı `promptTokens`; `ms` yaklaşık bine düşüyor. Bu Ollama'nın prefix cache'i.
3. **`question`'da tek kelime değişik.** `ms`'in yeniden yükselmesini izle.

Sonra bayat kitabı görmek isteyen salon için bir tane daha: `"edition": "2026-Q2"` yap ve gönder. **EUR 120** bekle, yine soğuk, çünkü prefix değişti. Aynı model, aynı soru, farklı kitap — cevap veriyle birlikte hareket etti; `kraken-q2`'nin yapamadığı tek şey bu.

**Ollama cevap vermiyorsa** request, api'den gelen bir connection hatasıyla hızla düşer. Onun yerine bir sonraki bölümdeki sayıları oku; bu modülün argümanı aritmetik ve demo olmadan da ayakta kalıyor.

## Sayılar ne dedi

<div class="measured">

| ne | ölçüm |
| --- | --- |
| Q3 corpus'u | 28 doküman, yaklaşık 79 000 karakter |
| o corpus tek prompt hâlinde | Ollama'nın kendi `prompt_eval_count`'una göre yaklaşık 21 000 prompt token |
| api'nin istediği window | 65 536 token (`src/routes/chat.ts` içinde `numCtx: 65536`) |
| ilk doldurulmuş soru, soğuk | M-serisi Mac'te 60–120 sn, `gemma3:4b` |
| birebir aynı request, tekrar | yaklaşık 1 sn — Ollama'nın prefix cache'i |
| sorunun tek kelimesi değişik | yine yavaş |
| cevap | genelde EUR 90, `[fare_classic_shorthaul]` kaynağıyla |
| söz konusu sayı | Q2 sayfası EUR 120, Q3 sayfası EUR 90 (`corpus/DELTA.md`) |
| aritmetikle ölçeklendi, ölçülmedi | doküman başına 750 token → 10 000 doküman ≈ soru başına 7,5 milyon token; günde 200 soru ≈ 1,5 milyar |

</div>

12 Eylül 2026'da, her biri tek koşu, modeller önceden çekilmiş hâlde eğitmenin M-serisi Mac'inde ölçüldü. Süreler başka makinelerde kayar; token sayıları kaymaz.

## Daha derine

Modül 5'in katı prompt'u — "YALNIZCA context'ten cevapla, yoksa bilmiyorum de" — burada kullanılmıyor. 80 KB context verilip emin olmadıkça çekinmesi söylenen 4B bir model her şeyde çekiniyor. `/chat` stuff modu onun yerine düz bir talimat kullanıyor (`chat.ts` içinde `STUFF_SYSTEM`). Bu kendi başına küçük bir ders: küçük bir modele ne kadar çok verirsen ondan o kadar az isteyebiliyorsun.

Adını koymaya değer bir orta yol var, çünkü birileri bunu kuracak: *scope'lanmış* bir prompt doldur. Kitabın tamamını değil, tek bir fare family'ye ait olan her şeyi — deterministik bir filtreyle seçerek; bilet zaten fare basis'in `KSHEU26` olduğunu söylüyor. Bu da retrieval, sadece embedding yerine metadata ile yapılıyor ve tarife gibi yapılı alanlarda çoğu zaman vector search'ü geçiyor. On milyon dokümanda şekil aynı, sadece daha geniş: önce ucuz seçim, sonra küçük bir pencerenin dikkatle okunması. Soru "ne kadarı sığıyor"dan çıkıp "seçim aşamasındaki recall'um ne" oluyor — ki bu modül 5'in sorusu, erken sorulmuş hâli.

Long-context modeller duvarı öteliyor, kaldırmıyor. Bir milyon token'lık window bizim dokümanlarımızdan 10 000'ini değil 1 300'ünü okur ve onları her soruda yeniden okur. Yukarıdaki cache argümanı olduğu gibi geçerli.

## Çıkış cümlesi

> Her çeyrek retrain edemem, her soruda kitabın tamamının bedelini de ödeyemem. Modelin 28 dokümandan birine ihtiyacı vardı. Ona yalnızca doğru parçayı ver — [modül 5](/tr/modules/05-simple-rag/).
