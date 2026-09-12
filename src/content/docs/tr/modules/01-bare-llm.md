---
title: "1. Çıplak LLM Duvarı"
description: "Tek başına modele tek bir istek: Kraken Air ceza sorusuna kendinden emin, spesifik, kaynaksız — ve yanlış — cevap veriyor. Sonra soruyu değiştiriyorsun, yine cevap veriyor."
---

## Gate sorusu

> **Model *benim* verimi biliyor mu?**

<div class="presenter-note">
Kimse tıklamadan önce soruyu ekrana yansıt ve salonu bağla: 4B'lik bir model Kraken Air'in iptal cezası sorusuna cevap verir mi, yoksa reddeder mi? El kaldırt, sesli say, iki rakamı tahtaya yaz. Çoğu salon ikiye bölünür. İstek meseleyi tek tıkla kapatıyor — cevap veriyor, üstelik bir rakamla — ve modülü taşıyan şey "reddeder" kampının şaşkınlığı. Bunun için iki dakika, fazlası değil.
</div>

Laptopta bir model. Retrieval yok, doküman yok, sadece ağırlıklar. Bütün günün üzerine kurulu olduğu tek soruyu soruyoruz — bir Kraken Air çağrı merkezi temsilcisinin vardiyada on kez aldığı soruyu:

> Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?

Gerçek rakam `corpus/2026-Q3/fare_classic_shorthaul.md` içinde, RULE 2A tablosu, K satırında: **EUR 90**, yolcu başına sabit bir tutar; yüzde değil. Geçen çeyrekte aynı satırda EUR 120 yazıyordu. Kraken Air kurgusal — modelin okumuş olabileceği bir web sitesi, bir forum başlığı, bir basın bülteni yok. Bu rakamı hiç görmedi.

İsteği çalıştır ve `answer` alanını oku. Bu sayfada kayıtlı bir döküm yok, bilerek: model `gemma3:4b`, `temperature: 0` ile çalışıyor ve yine de senin makinende eğitmeninkinden farklı bir şey söyleyecek. **Değişmeyen şey cevabın biçimi — kendinden emin, spesifik, kaynaksız ve yanlış.** 12 Eylül çalıştırmasında "€50" türünden bir tutar söyledi; olgu gibi, tek temiz cümleyle. Sendeki rakam farklı olacak. Farklı olması asıl mesele: bilgiye gerçekten sahip bir model her seferinde, her laptopta EUR 90 derdi.

Cevapta *olmayana* bak. "Emin değilim" yok. "Kraken Air ücret kurallarına göre" yok. Doküman adı yok, tarih yok, çeyrek yok. Cevabını gerçekten bildiği bir soruya vereceği cevapla tıpatıp aynı okunuyor.

Şimdi soruyu değiştir. İki havayolu hakkında herhangi bir şey:

- XX 1487 uçuşu saat kaçta kalkıyor
- Wyvern Overseas'te FLEX ücrette checked baggage hakkı ne
- Kraken Air'in İstanbul'daki lounge'unun adı ne
- aynı K biletinde no-show cezası ne kadar

Hepsine cevap verecek, aynı tonda, ve hiçbiri okuduğun cevabın dışında hiçbir yerde yok. Canını yakacak olan şu: no-show cezası aynı tabloda *var*, iptal cezasının iki sütun sağında — EUR 180 — ve modelin o hücreye erişimi diğer hücrelere olduğundan fazla değil.

<div class="presenter-note">
Birisi aynı soruyu daha büyük bir modele yazıyor olacak, "bakın o daha iyi" demek için. Bırak yapsın — frontier bir model çekince koymaya daha yatkındır, ama bu çeyreğin rakamının 120 değil 90 olduğunu yine bilemez, çünkü hiçbir model eğitildiği sırada var olmayan bir dokümanı okumuş değil. Tartışmayı oraya taşı, 4B modeli savunma.
</div>

## Neden başka türlü yapamaz

Bir dil modeli, öğrenilmiş ağırlıklar üzerinden bir sonraki token'ı tahmin eder: okuduğu her şeyin istatistiksel şeklini sıkıştıran milyarlarca parametre. İçeride lookup tablosu yok, kaynak doküman yok, hiçbir bilginin üstünde tarih yok. Soru geldiğinde ağırlıklar en olası devamı üretir — ve "K booking class için iptal cezası ne kadar" sorusunun en olası devamı *bir tutardır*. Ücret sorularına rakamla cevap veren metin, eğitim verisinde milyonlarca kez geçiyor. "Kraken Air'in ücret kurallarını okumadım" diye cevap veren metin geçmiyor.

Bazen bir model reddeder. O da bir bilgi kontrolü değil. Reddetme, sorunun *biçimine* oturtulmuş eğitilmiş bir davranış — instruction tuning sırasında, özel veya çabuk değişen bir bilginin sorgulanması biçimindeki sorularda "bilmiyorum" ödüllendirildi. Aynı soruyu farklı kur, ret kaybolur; çünkü iki seferde de içeride hiçbir şeye danışılmadı. *Daha derine* bunu sürdürüyor.

**Tehlike modelin yanılması değil. Tehlike, yanılırken doğru bildiği zamankiyle tıpatıp aynı sesi kullanması.** Aynı akıcılık, aynı kesinlik, aynı kaynaksızlık. Çıktının hiçbir yerinde ikisinden hangisini elinde tuttuğunu söyleyen bir işaret yok. Cevabı okuyan temsilci ayırt edemez, müşteri ayırt edemez, 200 ve boş olmayan bir string'e bakan bir test de ayırt edemez — Bruno'nun bu istekteki assertion'ı yanlış cevapta da yeşil yanıyor.

Duvar bu. Bundan sonrası bu duvarı aşma denemesi ve gün onları sırayla deniyor:

1. **Modeli kendi kurallarımızla eğit** — modül 3. Çalışıyor, veri değişene kadar.
2. **Bütün kural kitabını prompt'a koy** — modül 4 veriyi tek seferde hepten veriyor ve bunun soru başına neye mal olduğunu gösteriyor.
3. **Cevap anında doğru parçayı eline ver** — modül 5 ve sonrası. Corpus önümüzdeki çeyrekte yeniden yayımlandığında ayakta kalan tek seçenek.

## Ne çalıştırıyorsun

Tek istek. Container'lar [kurulum](/tr/modules/00-setup/)dan beri ayakta ve `00-health` dört `"ok"` döndürdü.

**Bruno — `01-bare-llm` › `ask-about-kraken`**

```json
{
  "question": "Passenger wants to cancel a short-haul Europe ticket, CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
}
```

Gönder. Cevap, ısınmış modelde birkaç saniye; günün ilk çağrısında Ollama ağırlıkları yüklerken daha uzun. Şu alanları oku:

- `mode` — `"bare"`. Hiçbir şey retrieve edilmedi; api sorunu tek satırlık bir system prompt ile Ollama'ya geçirdi, başka hiçbir şey yok.
- `model` — `"gemma3:4b"`.
- `answer` — modülün konusu olan cümle. Bir tutar var mı? EUR 90 mı? Kaynak var mı?
- `promptTokens` — küçük bir sayı: tek satırlık system prompt artı senin sorun. Aklında tut; modül 4 bunu ~21 000 yapıyor.
- `ms` — modelin ne kadar sürdüğü.

*Tests* sekmesindeki assertion'lar geçiyor. Status'a ve mode'a bakıyorlar, cevaba değil — response'ta bir testin cevabı karşılaştırabileceği hiçbir şey yok.

**Sonra body'yi düzenle ve yeniden gönder.** `question` alanını Kraken Air veya Wyvern Overseas hakkında herhangi bir şeyle değiştir: bir kalkış saati, bir bagaj kuralı, bir lounge, FLEX ücrette iade süresi. Üç dört tane gönder. Kaç kez reddettiğini say.

**Ollama cevap vermiyorsa.** `00-health` eksik parçayı söylüyor — Ollama'ya ulaşılamıyor ya da bir model pull edilmemiş. Şimdi debug etme: bu sayfayı oku, eğitmenin projektörünü izle ve modül 2'de aramıza katıl; o modülün modele hiç ihtiyacı yok. Arada düzelt; modül 5'ten itibaren lab buna ihtiyaç duyuyor.

<div class="presenter-note">
On iki dakika, canlı istek bunun iki dakikası. Bir katılımcının Ollama'sı kapalıysa geri düşülecek yer senin projektörün: isteği kendi makinende çalıştır, cevabı yüksek sesle oku, sonra salonun önünde soruyu iki kez değiştir. Kendi makinen de kapalıysa sayfayı oku ve "bu sabah benim makinemde böyle çıktı; modül 4'te aynı istekle kendiniz üreteceksiniz" de. Farklı rakam bildirenler olacak — tek cümlelik cevabı var: rakam değişir, kalıp değişmez. Kimsenin reddettirecek bir ifade peşine düşmesine izin verme; o bir prior'ın ayarlanması, bilginin kontrol edilmesi değil, üstelik saati yiyor.
</div>

## Daha derine

Ağırlıkları depolama değil sıkıştırma olarak düşün. Eğitim, bir corpus'u sabit bir parametre bütçesine sıkıştırır; sık geçen ve genel olan yüksek doğrulukla hayatta kalır, spesifik ve nadir olan hatırlanmaz, yeniden üretilir. Hallucination, eğitim verisinin hiç kısıtlamadığı bir bölgeden çekilmiş kendinden emin bir tahmindir. "İptal cezaları bir tutar ya da bir yüzdedir" geneldir, hayatta kalır. "2026-Q3 kısa menzil CLASSIC sayfasında K booking class için EUR 90" ise bir bilginin olabileceği kadar spesifiktir ve ölçeği ne kadar büyütürsen büyüt bu çizginin öbür tarafına geçmez — daha büyük bir model sorunun herkese açık kısmındaki olasılıkları değiştirir, özel kısmına hiç dokunmaz.

Reddetme genelde hak ettiğinden az ilgi görüyor. Instruction tuning, özel veya çabuk değişen bir bilginin sorgulanmasına *benzeyen* sorularda "bilmiyorum" demeyi ödüllendiriyor; yani sinyal ifadeyi takip ediyor, modelin o bilgiye sahip olup olmadığını değil. Çizgiyi bir system prompt ile oynatabilirsin — "yalnızca eminsen cevap ver" — ama yaptığın şey bir prior'ı ayarlamak, bir bilgi kontrolü kurmak değil; bunun bedelini lab ileride gösteriyor: katı bir abstain talimatı, doğru doküman context'inde dururken bile küçük bir modeli reddettiriyor. Alternatif, cevabı bulunmuş bir dokümana dayamak ve kaynağı cevapla birlikte geri vermek. Bu, rakam oynuyor mu diye modele beş kez sormaktan ucuz ve denetlenebilir — bir havayolunun ihtiyacı olan da denetlenebilirlik.

Çıplak isteğin gösteremediği tek şey: Kraken Air rakamının doğru cevap verilebilmesi için modelin *neresinde* durması gerektiği. Bir sonraki modül o.

## Çıkış cümlesi

> Bilmiyor — ve bilmediğini bilmiyor.

<div class="presenter-note">
Ağzında gevelememen gereken tek cümle bu. Yavaş söyle, üstüne bir şey ekleme, açıklama — açıklaması zaten bir sonraki modül. Sonra doğrudan modül 2'ye geç: "Peki bilgi bu şeyin neresinde, gerçekten? EUR 90'ın nerede depolanmış olması gerekirdi?"
</div>
