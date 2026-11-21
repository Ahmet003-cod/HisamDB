# HisamDB — Güvenilirlik, Dayanıklılık ve Eşzamanlılık Sözleşmesi (`RELIABILITY.md`)

Bu doküman, HisamDB `v0.6.0` sürümünün **gerçekte sağladığı** ve **sağlamadığı** veri bütünlüğü, çökme kurtarma ve eşzamanlılık garantilerini şeffaf biçimde tanımlar.

---

## 1. İşlem (Transaction) ve ACID Garanti Matrisi

HisamDB, **tek-süreçli (embedded / single-writer process), çok iş parçacıklı** bir vektör veritabanı olarak tasarlanmıştır. Aşağıdaki tablo her bir ACID özelliğinin gerçek kapsamını özetler:

| Özellik | Durum | Gerçek Garanti Düzeyi | Sınırlar (Garanti Edilmeyenler) |
|---|---|---|---|
| **Atomicity (Atomiklik)** | ✅ Tek koleksiyon / tek çağrı | Tek bir `add()` batch'inde disk yazımı başarısız olursa bellek tamponu, ID haritası, epoch ve `.mvdbvec` dosya uzunluğu tamamen geri alınır (**rollback**). SQLite yazımları tek işlem bloğundadır. | Çoklu koleksiyonları kapsayan dağıtık 2PC (two-phase commit) işlemleri yoktur. |
| **Consistency (Tutarlılık)** | ✅ Katı doğrulama | `NaN`/`Inf` reddi, boyut kontrolü, batch içi mükerrer ID reddi, path-traversal koruması, CRC32 doğrulaması (`.mvdbvec`, `.wal`, `.mvdbidx`, `.items.mvdbcol`) ve SHA-256 snapshot manifest doğrulaması. | Harici süreçlerin aynı `persist_dir` dizinine eşzamanlı yazması desteklenmez. |
| **Isolation (İzolasyon)** | ✅ Tek süreçte Seri Yazıcı / Çoklu Okuyucu | Yazıcı-öncelikli `RWLock`: Eşzamanlı okuyucular birbirini engellemez; her yazma (`add`, `upsert`, `delete`, `compact`) münhasır (exclusive) kilit altında çalışır. Multi-tenant katmanında `TenantCollection` katı kiracı yalıtımı sağlar. | Çoklu işlem (multi-process) eşzamanlı yazma kilidi veya MVCC snapshot izolasyonu yoktur. |
| **Durability (Dayanıklılık)** | ⚙️ Yapılandırılabilir (`wal_sync`) | Bkz. Bölüm 2 (`wal_sync="normal"` vs `wal_sync="full"`). `checkpoint()`, `compact()` ve `create_snapshot()` her zaman `os.fsync()` + atomik `os.replace()` kullanır. | `EphemeralClient` (bellek içi) kalıcı değildir. |

---

## 2. WAL Dayanıklılık Modları ve Çökme Senaryoları

Her kalıcı koleksiyon, 21-byte başlıklı (`MWAL` + `LSN` + `OpType` + `PayloadLen` + `CRC32`) ikili bir Write-Ahead Log (`.wal`) tutar:

| Mod (`index_params`) | Tekil Silme P50 (10K) | Süreç Çökmesi (`SIGKILL` / OOM) | Ani Elektrik Kesintisi | Kullanım Önerisi |
|---|---:|---|---|---|
| `{"wal_sync": "normal"}` *(varsayılan)* | **~0.40 ms** | ✅ **Kayıpsız** (OS sayfa önbelleğine `flush()` + SQLite `WAL/NORMAL`) | ⚠️ Son `checkpoint()`'tan sonraki son birkaç ms kaybedilebilir | Çoğu RAG, API ve AI Agent üretim iş yükü |
| `{"wal_sync": "full"}` | **~1.97 ms** | ✅ **Kayıpsız** | ✅ **Kayıpsız** (her WAL yazımında `os.fsync()`) | Finansal / denetim kayıtlı kritik iş yükleri |

### Kurtarma (Crash Recovery) Akışı
Veritabanı yeniden açıldığında her koleksiyon için otomatik olarak:
1. **Torn-Write Budama:** `.wal` dosyasının sonundaki yarım/bozuk CRC32 çerçeveleri temizce budanır.
2. **WAL Replay:** Son `CHECKPOINT` / `COMPACT` kaydından sonra gelen ancak çökme nedeniyle SQLite'a yansımamış `OP_DELETE` kayıtları geri oynatılır.
3. **Vektör & Önbellek Senkronizasyonu:** `.items.mvdbcol` önbelleği bozuksa veya epoch uyuşmuyorsa sistem sessizce SQLite doğruluk kaynağına düşer; `.mvdbvec` dosya uzunluğu SQLite slot sayısından fazlaysa fazla kuyruk güvenle budanır.
4. **İndeks Doğrulaması:** Kayıtlı `.hnsw.bin`, `.ivf.mvdbidx`, `.ivfpq.mvdbidx` veya `.disk.meta` dosyalarında CRC32/epoch uyuşmazlığı varsa bozuk indeks yok sayılır ve sağlam vektörlerden otomatik yeniden inşa edilir.

---

## 3. Eşzamanlılık ve Arka Plan Bakım Mimarisi (`v0.6.0`)

- **Okuyucu-Yazıcı Kilidi (`RWLock`):** Çoklu okuyucular (`search`, `batch_search`, `hybrid_search`, `batch_hybrid_search`) paralel çalışır; bekleyen yazıcı varken okuyucu kıtlığı yerine yazıcı önceliği uygulanır.
- **Tembel İndeks Thread-Safety:** `MetadataIndex`, `BM25Index`, `SparseVectorIndex`, `BM25Index._ensure_compiled()` ve `VectorBuffer._ensure_norms()` soğuk açılış sonrası eşzamanlı okuyucu yarışlarını (race condition) engellemek için double-checked `threading.Lock` koruması altındadır.
- **HNSW Eşzamanlı `ef_search` Koruması:** `HNSWIndex` dahili `_ef_lock` RWLock'u ile varsayılan `ef_search` sorgularını tamamen paralel yürütürken, özel `ef_search` değişikliklerini yarışmasız korur.
- **Bloklamayan Arka Plan İndeks İnşası (`rebuild_index_async()`):**
  1. Kısa bir `read_lock` altında vektör anlık görüntüsü alınır.
  2. Ağır K-Means / PQ / HNSW graf eğitimi **tamamen kilit dışında** çalıştırılır (okuyucular kesintisiz hizmet verir).
  3. Mikrosaniyelik `write_lock` altında inşa sırasında gelen yeni vektör deltaları yeni indekse eklenir, tombstone'lar işlenir ve indeks atomik olarak takas edilir.
- **Arka Plan Sıkıştırma (`compact_async()` ve `auto_compact_async=True`):** Tekrarlı sıkıştırma çağrıları tek görevde birleştirilir (coalescing). `index_params={"auto_compact_async": True}` etkinleştirildiğinde yoğun silmeler bile `<1 ms` içinde döner ve fiziksel sıkıştırma arka planda tamamlanır.
- **Temiz Kapatma:** `client.close()` veya `async with AsyncHisamDB(...)` çıkışı tüm arka plan görevlerini bekler, değişen indeksleri ve `.items.mvdbcol` önbelleğini kaydeder ve tüm `mmap`/WAL/SQLite/havuz tutamaçlarını serbest bırakır.

---

## 4. Yedekleme ve Geri Yükleme Kontrol Listesi

```python
# 1. Tutarlı SHA-256 doğrulamalı anlık görüntü (önce otomatik checkpoint yapar)
manifest = db.create_snapshot("./backups/snap_2026_11_21")

# 2. Doğrulamalı geri yükleme (tek bit bile bozulsa DataCorruptionError fırlatır)
HisamDB.restore_snapshot("./backups/snap_2026_11_21", "./restored_db")
```
