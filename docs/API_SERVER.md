# HisamDB Production REST Sunucusu & Python SDK Rehberi (`v0.7.0`)

HisamDB, hem **gömülü (embedded)** Python kütüphanesi hem de **üretim sınıfı FastAPI + Uvicorn REST sunucusu** olarak çalışabilir. Çekirdek kütüphane yalnızca `numpy` gerektirir; sunucu bileşenleri opsiyonel `server` eklentisiyle kurulur.

---

## 1. Kurulum ve Başlatma

```bash
# Sunucu bağımlılıkları ile kurulum:
pip install ".[server]"
# veya HNSW + orjson + REST sunucusu hepsi birden:
pip install ".[all]"

# Komut satırından sunucuyu başlat:
hisamdb-server --host 0.0.0.0 --port 8000 --data-dir ./mvdb_data
# Alternatif:
python -m hisamdb.api.server --host 0.0.0.0 --port 8000 --data-dir ./mvdb_data
```

### Docker & Docker Compose ile Dağıtım

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

---

## 2. Ortam Değişkenleri & Güvenlik

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `HISAM_HOST` / `HISAM_PORT` | `127.0.0.1` / `8000` | Sunucu dinleme adresi ve portu |
| `HISAM_DATA_DIR` | `./hisamdb_server_data` | Kalıcı veri dizini (`:memory:` bellek içi mod) |
| `HISAM_SNAPSHOT_DIR` | `<data_dir>/_snapshots` | Path-traversal korumalı snapshot kök dizini |
| `HISAM_USE_MMAP` | `false` | Koleksiyonları varsayılan olarak `mmap` modunda aç |
| `HISAM_MAX_WORKERS` | Otomatik (≤8) | `AsyncHisamDB` / `BatchQueryExecutor` işçi sayısı |
| `HISAM_REQUIRE_AUTH` | `false` | `true` ise tüm `/v1/*` ve `/metrics` isteklerinde API anahtarı zorunlu |
| `HISAM_API_KEYS` | *(boş)* | `;` ile ayrılmış anahtar listesi (düz veya önceden `sha256:<hex>` hash'lenmiş) |
| `HISAM_RATE_LIMIT_RPS` / `BURST` | `0` / `0` | Kimlik/IP başına saniyelik istek hızı ve ani patlama (burst) sınırı |
| `HISAM_MAX_BATCH_SIZE` | `10000` | Tek istekte izin verilen azami vektör / sorgu sayısı (aşılırsa `413`) |
| `HISAM_CORS_ORIGINS` | *(boş)* | Virgülle ayrılmış izinli CORS origin listesi |

### API Anahtarı, Kapsam (Scope) ve Sunucu Tarafı Multi-Tenant Biçimi

Her anahtar kaydı `<key_veya_sha256>:<key_id>:<kapsamlar>[:<tenant_id>]` biçimindedir:

```bash
export HISAM_REQUIRE_AUTH=true
export HISAM_API_KEYS="sha256:<admin_hash>:admin:read,write,admin;mvdb_acme_secret:acme:read,write:acme_corp;mvdb_ro:reader:read"
```

- **Hash'leme:** Ham anahtarlar başlangıçta SHA-256 ile özetlenir ve düz metin olarak saklanmaz; doğrulama sabit zamanlı `hmac.compare_digest` ile yapılır.
- **Kapsamlar:** `read`, `write`, `admin`.
- **Kiracı Kilidi (`tenant_id`):** Bir anahtara `tenant_id` bağlandığında o anahtarla yapılan **tüm** `add`, `documents`, `upsert`, `get`, `search`, `batch_search`, `sparse_search`, `hybrid_search`, `retrieve_context` ve `delete` çağrıları sunucu tarafında o kiracıya kilitlenir; çapraz kiracı okuma/silme/upsert (`403`) engellenir.

---

## 3. REST API Uç Noktaları Özeti

| Metot & Yol | Kapsam | Açıklama |
|---|---|---|
| `GET /health` · `GET /ready` | Herkese açık | Canlılık (liveness) ve hazırlık (readiness) probları |
| `GET /metrics` | `read` | Prometheus metin formatı (`hisamdb_http_requests_total`, histogram, vektör/tombstone/RAM) |
| `GET /v1/collections` | `read` | Tüm koleksiyon adlarını listele |
| `POST /v1/collections` | `write` *(kiracısız)* | Koleksiyon oluştur (`get_or_create` destekli) |
| `GET /v1/collections/{name}` | `read` | Koleksiyon istatistikleri (`stats()`) |
| `DELETE /v1/collections/{name}` | `admin` | Koleksiyonu diskten güvenle sil |
| `POST /v1/collections/{name}/add` | `write` | Toplu vektör / doküman / metadata / sparse vektör ekle |
| `POST /v1/collections/{name}/documents` | `write` | RAG doküman/chunk ekleme (`doc_ids`, `parent_ids`) |
| `POST /v1/collections/{name}/upsert` | `write` | Ekle veya yerinde güncelle |
| `POST /v1/collections/{name}/get` | `read` | ID veya `where` filtresiyle kayıt getir |
| `POST /v1/collections/{name}/delete` | `write` | `<1 ms` Tombstone silme (`ids` veya `where`) |
| `POST /v1/collections/{name}/search` | `read` | Dense benzerlik araması |
| `POST /v1/collections/{name}/batch_search` | `read` | Çok çekirdekli paralel toplu arama |
| `POST /v1/collections/{name}/sparse_search` | `read` | Okapi BM25 / Sparse vektör araması |
| `POST /v1/collections/{name}/hybrid_search` | `read` | Dense + BM25 + Sparse hibrit arama (`rrf`, `weighted`, `z_score`) |
| `POST /v1/collections/{name}/batch_hybrid_search` | `read` | Toplu heterojen hibrit arama |
| `POST /v1/collections/{name}/retrieve_context` | `read` | Parent-Child komşu parça bağlam penceresi |
| `POST /v1/collections/{name}/explain` | `read` | `QueryOptimizer` yürütme planını incele |
| `POST /v1/collections/{name}/compact` | `admin` | Tombstone sıkıştırması (`?background=true` destekli) |
| `POST /v1/collections/{name}/rebuild` | `admin` | Bloklamayan arka plan indeks inşası (`?wait=false` destekli) |
| `POST /v1/collections/{name}/checkpoint` | `admin` | Tam CRC32 atomik checkpoint |
| `POST /v1/admin/snapshot` | `admin` | SHA-256 doğrulamalı tutarlı snapshot al |

---

## 4. Sıfır-Bağımlılıklı Python HTTP SDK (`RemoteHisamDB`)

Harici `requests` veya `httpx` gerektirmez; standart `http.client` ile **HTTP/1.1 Keep-Alive** bağlantısı kullanır:

```python
from hisamdb import RemoteHisamDB

with RemoteHisamDB("http://127.0.0.1:8000", api_key="mvdb_admin_key") as db:
    coll = db.get_or_create_collection("docs", dimension=128, index_type="hnsw")
    coll.add(ids=["d1"], vectors=[[0.1] * 128], documents=["İstanbul veri merkezi"])
    hits = coll.hybrid_search(query_vector=[0.1] * 128, query_text="İstanbul", limit=5)
    print(hits[0].id, hits[0].score)
```
