"""
microvectordb.ivf_index
-----------------------
Geriye dönük uyumluluk cephesi (facade).
Asıl optimize edilmiş IVF implementasyonu `microvectordb.indexes.ivf` altındadır.
"""

from microvectordb.indexes.ivf import IVFIndex

__all__ = ["IVFIndex"]
