# Pre-work maili — 4 Ekim'de gönderilecek (eğitimden 3 gün önce)

> Bu bir şablon. Göndermeden önce: linkleri tıkla, komutları temiz bir makinede kopyala-yapıştır
> dene, ve kendi telefonundan maili aç — kod blokları düzgün görünüyor mu bak.
> `UNVERIFIED` ile işaretli iki yer (Windows'ta admin'siz Podman, api imajının boyutu) 2–4 Ekim
> provasında gerçek bir Windows laptopta doğrulanıp düzeltilmeli.

---

**Konu:** RAG eğitimi 7 Ekim — hazırlık: üç kurulum, ~5 GB indirme (lütfen ofiste değil evde)

Merhaba,

7 Ekim Çarşamba günkü RAG eğitimi tamamen **kendi laptopunuzda** çalışacak. API key yok, cloud
hesabı yok, hiçbir yere giriş yok, Python yok. Bunun tek bedeli: üç programı kurmuş ve iki modeli
**önceden** indirmiş olmanız gerekiyor.

**Lütfen bunu evde yapın, ofis ağında değil.** Hepsi bittiğinde diskte yaklaşık 5 GB yer tutuyor:
4.5 GB model, ~650 MB ChromaDB imajı ve lab'ın küçük api imajı. Komutları yazmak on dakika;
indirmelerin bitmesi bağlantınıza göre yarım saati buluyor.

**Eğitim günü salonda model indirmek yok.** Yirmi laptop aynı anda birkaç GB çekmeye kalkarsa
sabahı kaybediyoruz. Aşağıdaki adımlar bunun için var.

## 1. Üç programı kurun — 10 dakika

| | Ne işe yarıyor | Nereden |
|---|---|---|
| **Ollama** | modelleri laptopunuzda çalıştırıyor | https://ollama.com/download |
| **Podman Desktop** | ChromaDB'yi ve lab'ın api'sini container olarak çalıştırıyor | https://podman-desktop.io |
| **Bruno** | gün boyu tıklayarak ilerlediğimiz istek koleksiyonu | https://www.usebruno.com/downloads |

- **Ollama**, Windows'ta kendi kullanıcı klasörünüze kuruluyor, admin sormuyor. Kurulumdan sonra
  açık terminal pencerelerini kapatıp yenisini açın: `ollama` komutu ancak yeni pencerede tanınıyor.
- **Podman Desktop**, Windows'ta altta WSL2 kullanıyor; ilk açılışta "Podman makinesi"ni kurmasını
  isteyin. `UNVERIFIED: WSL2 makinenizde kapalıysa açılması admin hakkı ve bir yeniden başlatma
  isteyebilir.` Admin şifresi sorarsa **durun ve bana yazın** — sabah 09:00'da IT ile tartışmaktan
  iyidir; makinesi yeşil olan biriyle eşleştiririz.
- **Bruno**, her iki sistemde de admin istemeden kuruluyor.

## 2. İki model indirin — 4.5 GB, bağlantınıza göre 15–30 dakika

Terminal (Windows'ta PowerShell) açıp:

```
ollama pull gemma3:4b
ollama pull bge-m3
```

`ollama list` bunları 3.3 GB ve 1.2 GB olarak gösteriyor. İkisi de gerekiyor: birincisi soruları
cevaplıyor, ikincisi metni vektöre çeviriyor. **Başka bir model koymayın** — gün içindeki her
cevap bu ikisiyle üretildi; model değişirse cevaplar da değişir.

## 3. Lab'ı klonlayıp bir kez ayağa kaldırın — 5 dakika + indirme

```
git clone https://github.com/kuthaygumus/amadeus-rag-lab
cd amadeus-rag-lab
podman compose up --build
```

**git yoksa gerek de yok.** Şu ZIP'i indirip çıkarın, sonra çıkardığınız klasöre `cd` yapın:
https://github.com/kuthaygumus/amadeus-rag-lab/archive/refs/heads/main.zip

İlk `podman compose up --build` iki şey yapıyor: ChromaDB imajını indiriyor (~650 MB) ve lab'ın
api imajını oluşturuyor (bir dakika civarı). İkisi de **bir kere** oluyor; gün içinde tekrar
indirme yok. Terminalde `chroma` ve `api` satırları akmaya başladığında bitti demektir.
Bu pencereyi açık bırakın, 4. adıma geçin. İşiniz bitince aynı pencerede `Ctrl+C`, sonra
`podman compose down` diyebilirsiniz; indirilenler diskte kalıyor.

`UNVERIFIED: api imajının oluşması sırasında \`npm ci\` kurumsal proxy arkasında takılabilir.
Evde yapın; imaj bir kez oluştuğunda ofiste yeniden oluşturulmuyor.`

## 4. Yeşil ışık — 1 dakika

Bruno'yu açın → **Open Collection** → klonladığınız klasörün içindeki `bruno/amadeus-rag-lab`
klasörünü seçin → sağ üstten environment olarak **`local`** seçin → soldan **`00-health`** →
**`health`** isteğini açıp **→ (Send)** deyin.

**`"status": "ready"` ve dört `"ok"` görüyorsanız işiniz bitti.** Kapatın, unutun, çarşamba görüşürüz.

Cevap başka bir şey diyorsa eksik olanı ismiyle söylüyor:

| cevapta ne yazıyor | ne yapacaksınız |
|---|---|
| `"ollama": "unreachable at …"` | Ollama açık değil — uygulamayı başlatın (menü çubuğu / sistem tepsisinde ikonu görünmeli) |
| `"chatModel": "missing — run: ollama pull …"` (ya da `embedModel`) | cevabın içinde yazan `ollama pull` komutunu aynen koşturun |
| `"chroma": "unreachable at …"` | 3. adımdaki `podman compose up` çalışmıyor — terminal penceresine bakın |
| Bruno "could not connect" | api container'ı ayakta değil — `podman ps` iki satır göstermeli |

`collections` boş görünüyor; normal — henüz hiçbir şey ingest edilmedi, onu salonda yapıyoruz.

Takılırsanız bana cevabın ekran görüntüsünü atın — sabah 09:10'da yanınıza oturmaktan iyidir.

## İndirme tutmazsa: USB var

Evdeki bağlantı yarıda pes ederse ya da `ollama pull` bir türlü bitmezse **bana cevap yazın**,
size USB hazırlayayım. Yanımda iki stick olacak ama günden önce haber vermeniz, sabah
öğrenmemden iyi.

Model dosyaları diskte sıradan dosyalar:

| İşletim sistemi | Dizin |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

USB'den kopyalarken iki şeye dikkat:

1. **Önce Ollama'yı tamamen kapatın** — macOS'ta menü çubuğundaki ikon → Quit, Windows'ta
   sistem tepsisindeki ikon → Quit Ollama. Windows'ta sunucu oturumla birlikte açılıyor ve o
   dosyaları açık tutuyor.
2. **`blobs/` ve `manifests/` içeriğini birleştirin, klasörü komple değiştirmeyin.** Üstüne
   kopyalamak, makinede zaten olan modellerin kaydını siliyor.

Sonra Ollama'yı açıp `ollama list` deyin.

## Windows'ta bilinen bir pürüz

`00-health` `"ollama": "unreachable at …"` diyor ama terminalde `ollama list` çalışıyorsa: Windows'ta
Ollama bazen sadece `127.0.0.1`'i dinliyor ve container içinden görünmüyor. Kullanıcı ortam
değişkeni olarak `OLLAMA_HOST=0.0.0.0` ekleyip Ollama'yı kapatıp açın, isteği tekrar gönderin.
Admin gerekmiyor.

## Gün hakkında

09:00–15:00, öğle arası var. Günün teori kısımlarını (sinir ağı nasıl öğrenir, fine-tuning)
projektörden ben anlatıyorum — onlar için laptopunuzda hiçbir şey çalışmıyor. Pratik kısımlarda
site size Bruno'da bir klasör ve bir istek adı veriyor; gönderiyorsunuz, cevap dersin kendisi.

Repo, container'lar ve Bruno koleksiyonu sizde kalıyor; pazartesi sabahı da, internetsiz, aynı
laptopta çalışmaya devam ediyor.

Materyal: https://amadeus-rag-training.vercel.app/tr/

Görüşmek üzere,
Kuthay
