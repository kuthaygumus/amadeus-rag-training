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
İlk hücreyi çalıştırmadan önce salonu bir tarafa yazdır: "28 doküman context window'a sığar mı, evet mi hayır mı? Hayır diyenler el kaldırsın." Eller çoğunlukla kalkar. Yanılıyorlar; modülün tamamı bu. Tahmini ekranda sayı belirmeden önce kayda geçir. Bu sayfa için 25 dakika.
</div>

## Sığıyor. Sorun bu değil.

Q3 corpus'u **28 doküman, 75 KB**. `corpus/2026-Q3/` altındaki bütün dosyaları birleştir, soruyu önüne koy, `qwen2.5:3b`'ye gönder. **EUR 90** diyor. Yemek fişi sorusunu da doğru cevaplıyor. İki doküman gerektiren multi-hop soruları da biliyor, çünkü iki doküman da orada duruyor.

Bu eğitim burada rahat bir yalan söyleyebilirdi: corpus çok büyük, token duvarına toslarsın, o yüzden RAG şart. Bu ölçekte doğru değil. 75 KB'lık bir kural kitabı modern bir context window'a rahat rahat sığıyor ve prompt'a doldurmak işe yarıyor. Bunu açıkça söyle; salonun yarısı zaten bundan şüpheleniyor.

Başarısızlık doğrulukta değil. Faturada.

## Aritmetik

Token sayısını tokenizer'dan değil, modelin kendisinden oku. Ollama'nın `/api/chat` cevabı `prompt_eval_count` (gerçekten okuduğu token sayısı) ve `prompt_eval_duration` (okumak için harcadığı süre) alanlarını taşıyor. Bunun pratik bir sebebi var: kurumsal ağda HuggingFace model ağırlıkları policy ile bloklu, yani token saymak için öylece bir tokenizer `pip install` edemiyorsun. Model zaten senin için saymış.

75 KB'lık karışık İngilizce-Türkçe metin kabaca **20-25k token** ediyor. Bu, bir sorunun bedeli. Corpus'u bir kez yüklemenin değil, **her** sorunun bedeli; çünkü model stateless, prompt'un tamamını her seferinde baştan okuyor.

Şimdi çarp. Bir vardiyada yirmi agent, kişi başı on soru, 200 sorgu eder. Sorgu başına yaklaşık 22k token corpus ile bu, tek vardiyada modelin içinden geçen yaklaşık 4,4 milyon token kural kitabı demek — belki 20k token'lık gerçek cevap üretmek için. Bu ölçülmüş bir değer değil, corpus boyutundan türetilmiş aritmetik; ama faturayı ilk gören herkesin yapacağı aritmetik de tam olarak bu.

Biri prompt caching diyecek ve bu, sayfadaki en güçlü itiraz. Sabit bir prefix cache'lenebilir, cache'lenmiş prefix'i tekrar okumak da ucuzdur. Ama göründüğünden azını çözüyor. Corpus her çeyrek yeniden yayımlanıyor; yani cache, senin değil Revenue Management'ın takvimiyle geçersizleşiyor. Cache isabet etse bile uzun bir key-value cache'in decode tarafındaki maliyeti duruyor. Ve yanlış şekilde bir optimizasyon: 28 dokümanın hepsine ödeme yapmayı ucuzlatıyor, hepsine ödeme yapmanı engellemiyor.

## Şimdi ölçekle

Bu corpus'ta 28 doküman var çünkü bir laptopa ve tek bir güne sığmak zorunda. Gerçek bir kural kitabı on binlerce dokümandır: her route band'deki her fare family, her SOP revizyonu, her schedule bulletin, her interline anlaşması — üstelik çağrı merkezinin cevap verdiği her dilde.

Oranı koru. 28 doküman 75 KB ise, 28.000 doküman kabaca **75 MB** eder; yani **20-25 milyon token** mertebesi. Bugün production'da bunu alan bir context window yok, çağrı kuyruğundaki her görüşmede harcayacak kadar ucuza alan hiç olmayacak. Orada duvar gerçek, ve mühendislikle aşacağın bir duvar değil. Etrafından dolaşacağın bir duvar.

<div class="presenter-note">
Ağzında dolanmaması gereken cümle: "Sığıyor. Sorun bu değil. Sorun, tek bir soruyu cevaplamak için kitabın tamamının bedelini ödemen — her soruda." Bir kez, yavaş söyle; token sayısı ekranda dururken. Notebook cevap yerine hata verirse başka hiçbir şeyi debug etmeden önce `num_ctx`'e bak: Ollama'nın varsayılan context window'u bu corpus'tan küçük ve taşan prompt baştan sessizce kırpılıyor. Notebook tam bu yüzden değeri açıkça set ediyor. Ollama yine de inat ederse, notebook kayıtlı çıktı hücreleriyle geliyor: token sayısını ve süreyi kayıtlı koşudan oku ve devam et.
</div>

<div class="presenter-note">
İki SOP revizyonunu göstermeden önce sor: "Corpus'ta hem superseded prosedür hem güncel olan var. İkisi de prompt'un içindeyse model sana hangi sayıyı verir?" Salondan iki tahmin al, sonra `sop_misconnect_v3.md` ile `sop_misconnect_v4.md`'yi ekranda yan yana aç — EUR 10 ve EUR 15, aralarındaki fark tek kelimelik bir metadata. Prompt'u doldurmanın doğruluğu bozduğunu ölçtüğümüzü iddia etme. Ölçmedik, ve biri bunu kontrol eder.
</div>

## İkinci maliyet, dürüstçe

Prompt doldurmaya karşı daha ince bir argüman var ve fazla iddialı söylemek kolay. 25k token'lık büyük ölçüde alakasız metin verilen bir model, doğru 500 token verilen aynı modelden genelde daha kötü cevap veriyor: daha çok metin, dikkat için yarışan daha çok makul görünen yanlış malzeme demek.

Bu corpus'ta tuzak hazır kurulu. `sop_misconnect_v3.md` **Superseded** işaretli ve yemek fişini **EUR 10** diyor. `sop_misconnect_v4.md` **Current** işaretli ve **EUR 15** diyor. İkisi de `corpus/2026-Q3/` içinde. Hepsini doldurduğunda modele iki sayıyı birden veriyor, sonra tek kelimelik bir metadata'yı fark etmesine güveniyorsun.

Cevap tam önündeyken bile modelin yanlış okuyabildiğine dair ölçülmüş bir ipucumuz var: `llama3.2:3b`, CLASSIC K iptal cezasına **EUR 90** yerine **EUR 70** dedi — üstelik *sütun başlığı context'in içindeyken*. Change penalty sütununu okudu. Bu model bu yüzden eğitimde yasaklı.

Bu kanıtın sınırını da net söyle. **Bu sayfada, bu corpus üzerinde distractor etkisinin ölçümü yok.** Gold set hiçbir zaman "tüm corpus prompt'ta" ile "retrieval'dan gelen context" karşılaştırmasıyla koşulmadı; dolayısıyla "uzun context burada doğruluğu düşürür" cümlesi bizim sayılarımızdan değil, literatürden ödünç. Bunu neyin çözeceği belli: aynı 20 gold soruyu sabit generation ayarlarıyla iki kez koş — bir kez tüm corpus prompt'ta, bir kez top-5 structure-aware chunk ile — ve cevap doğruluğunu karşılaştır. O yapılana kadar maliyet argümanı kendi ayakları üstünde duruyor, doğruluk argümanı ise bir hipotez.

## Ne çalıştırıyorsun

Notebook: `03_stuff_the_prompt.ipynb`.

```bash
cd ~/amadeus-rag-training
python scripts/verify_setup.py          # başlamadan önce yeşil olsun
jupyter lab notebooks/03_stuff_the_prompt.ipynb
```

Notebook'un içinde:

```python
from pathlib import Path
from eval.retrieval import generate

corpus = "\n\n".join(
    p.read_text() for p in sorted(Path("corpus/2026-Q3").glob("*.md"))
)
print(len(corpus), "characters across 28 documents")

question = "Helios CLASSIC K short-haul: how much is the cancellation penalty?"
answer = generate(f"{corpus}\n\nQuestion: {question}\nAnswer from the documents above.")
```

Sonra süreyi tut ve `prompt_eval_count`'u ham Ollama cevabından oku; token sayısı kronometrenin yanında ekranda dursun.

## Sayılar ne dedi

<div class="measured">

| ne | ölçüm |
| --- | --- |
| prompt'a doldurulan Q3 corpus'u | 28 doküman, 75 KB |
| bunun token karşılığı | kabaca 20-25k (corpus boyutundan aritmetik, ölçüm değil) |
| `qwen2.5:3b`, kısa retrieved context ile | 3/3 doğru, 0,9 s |
| `gemma3:4b`, aynı sorular | 2/3, 1,9 s |
| `qwen3:4b`, aynı sorular | doğru, 11,6 s (reasoning token'ları) |
| `llama3.2:3b`, başlık context'te | doğru cevap EUR 90 iken EUR 70 dedi — eğitimde yasaklı |
| söz konusu sayı | Q2 sayfası EUR 120, Q3 sayfası EUR 90 |

</div>

Çıpa 0,9 saniye: `qwen2.5:3b`'nin kısa bir context'ten cevap verme süresi. Önünde 20-25k token varken kronometre ne gösterirse göstersin, onu bununla karşılaştır. Aradaki fark, prompt doldurmanın sorgu başına maliyeti.

## Daha derine

Uzun prompt iki ayrı yerden pahalı, çünkü prefill ile decode farklı ölçekleniyor. Prefill, tüm dizi üzerinde self-attention — naif formülasyonda kuadratik, flash-attention kernel'leriyle bile eklediğin her token'la birlikte büyüyen bir iş. Decode ise başka: üretilen her token, boyutu prompt uzunluğuyla doğrusal büyüyen bir key-value cache üzerinde attention yapıyor. 25k token'lık bir prompt sadece yavaş bir başlangıç maliyeti değil. Cevabın tamamı boyunca ondan sonraki her token'ı yavaşlatıyor ve bellekte şişiriyor.

Caching argümanının ancak yarısının işlemesinin sebebi de bu. Prefix caching tekrarlanan prefill'i ortadan kaldırıyor. Uzun prefix'in ürettiği KV cache'i kaldırmıyor; o cache bellekte duruyor ve her decode adımında üzerinden geçiliyor. GPU'da istekleri batch'lediğinde eşzamanlı kullanıcı sayısını genelde compute değil KV cache sınırlıyor. Kullanıcı başına 25k token doldurmak, kaç kullanıcının sığdığından doğrudan kesmek demek.

Doğruluk tarafında herkesin başvurduğu mekanizma "lost in the middle": uzun bir prompt'un ortasındaki malzeme, baştaki ve sondaki kadar güvenilir kullanılmıyor. Bunu gerçek ama belirli bir corpus üzerindeki büyüklüğü belirsiz bir etki olarak ele al — bu corpus dahil, çünkü biz ölçmedik. Yine de prompt doldurmalı bir sistem çıkaracaksan ucuz önlem sıralama: en muhtemel dokümanı en sona, sorunun hemen öncesine koy; modelin her yeri eşit taradığına güvenme.

10 milyon dokümanda cevap daha büyük bir window değil. İki aşamalı bir huni: önce ucuz ve geniş seçim, sonra küçük ve pahalı bir modelin küçük bir pencereyi okuması. Soru da "ne kadarı sığıyor"dan çıkıp "seçim aşamasındaki recall'üm ne" haline geliyor; çünkü huninin düşürdüğü şey, generator ne kadar iyi olursa olsun ulaşılamaz. Adını koymaya değer bir orta yol var, çünkü birileri bunu kuracak: *scope'lanmış* bir prompt doldur. Kitabın tamamını değil, tek bir fare family'ye ait olan her şeyi — deterministik bir filtreyle çözerek. Bilet zaten fare basis'in `KSHEU26` olduğunu söylüyor. Bu da retrieval; sadece embedding yerine metadata ile yapılıyor ve tarife gibi yapılı alanlarda çoğu zaman vector search'ü geçiyor.

Bunların hiçbiri fine-tuning'i anlamsız kılmıyor. Fine-tuning modele işin şeklini öğretti: kelime dağarcığını, tonu, fare basis kodu diye bir şey olduğunu. Prompt doldurmak ise sahip olmadığı bilgiyi veriyor. İkisi farklı yönlere doğru bozuluyor; kazanan tasarım şekli ağırlıklara, bilgiyi context'e koyuyor.

## Çıkış cümlesi

> Sığdı. Ama her sorguda hepsinin bedelini ödedim. Doğru parçayı seçmem lazım.
