"""
hisamdb.ivf_index
-----------------------
Geriye dönük uyumluluk cephesi (facade).
Asıl optimize edilmiş IVF implementasyonu `hisamdb.indexes.ivf` altındadır.
"""

from hisamdb.indexes.ivf import IVFIndex

__all__ = ["IVFIndex"]
