"""
Tibbiyot NLP Qidiruv Tizimi - Bitta faylda to'liq Django ilovasi
Ishga tushirish:
    pip install -r requirements.txt
    python app.py migrate
    python app.py loaddata    # CSV dan ma'lumot yuklash (bir marta)
    python app.py runserver
"""

import os
import sys
import re
import json
import math
import string
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────
# Django sozlamalari (settings)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

os.environ.setdefault('DJANGO_SETTINGS_MODULE', '__main__')

from django.conf import settings

if not settings.configured:
    settings.configure(
        BASE_DIR=BASE_DIR,
        SECRET_KEY='medcluster-secret-key-change-in-production-2026',
        DEBUG=True,
        ALLOWED_HOSTS=['*', '.onrender.com'],
        INSTALLED_APPS=[
            'django.contrib.admin',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
            'django.contrib.staticfiles',
        ],
        MIDDLEWARE=[
            'django.middleware.security.SecurityMiddleware',
            'django.contrib.sessions.middleware.SessionMiddleware',
            'django.middleware.common.CommonMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.contrib.auth.middleware.AuthenticationMiddleware',
            'django.contrib.messages.middleware.MessageMiddleware',
            'django.middleware.clickjacking.XFrameOptionsMiddleware',
        ],
        ROOT_URLCONF='__main__',
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': False,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
                'loaders': [
                    ('django.template.loaders.locmem.Loader', {}),
                ],
            },
        }],
        WSGI_APPLICATION='__main__.application',
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': BASE_DIR / 'db.sqlite3',
            }
        },
        DEFAULT_AUTO_FIELD='django.db.models.BigAutoField',
        STATIC_URL='/static/',
        STATIC_ROOT=BASE_DIR / 'staticfiles',
        LANGUAGE_CODE='uz',
        TIME_ZONE='Asia/Tashkent',
        USE_I18N=True,
        USE_TZ=True,
    )


# ─────────────────────────────────────────────
# Django ni ishga tushirish
# ─────────────────────────────────────────────
import django
django.setup()

from django.db import models


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class Document(models.Model):
    """Tibbiy hujjat modeli."""
    LABEL_CHOICES = [
        ('lab', 'Laboratory Report'),
        ('clinical', 'Clinical Note'),
        ('anam', 'Anamnesis'),
        ('other', 'Other'),
    ]
    matn = models.TextField(verbose_name="Matn")
    label = models.CharField(max_length=100, blank=True, null=True, verbose_name="Turi")
    manba_fayl = models.CharField(max_length=255, blank=True, null=True, verbose_name="Manba fayl")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'contenttypes'  # Mavjud app ga bog'laymiz
        db_table = 'document'
        verbose_name = "Hujjat"
        verbose_name_plural = "Hujjatlar"

    def __str__(self):
        return f"Hujjat #{self.pk}"


# ─────────────────────────────────────────────
# SEARCH ENGINE - Tokenizer
# ─────────────────────────────────────────────
STOP_WORDS_EN = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these',
    'those', 'it', 'its', 'as', 'not', 'no', 'nor', 'so', 'yet',
}
STOP_WORDS_UZ = {
    'va', 'yoki', 'lekin', 'ammo', 'bilan', 'uchun', 'bu', 'shu',
    'ular', 'biz', 'men', 'sen', 'u', 'ham', 'esa', 'da', 'ga',
    'ni', 'ning', 'dan', 'bir', 'ko', 'bo', 'qil',
}
STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_UZ


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Matnni tokenlarga ajratadi."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s'''\-]", ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def tokenize_medical(text: str) -> List[str]:
    """Tibbiy matnlar uchun maxsus tokenizatsiya."""
    if not text:
        return []
    medical_abbrevs = re.findall(r'\b[A-Z]{2,}\b', text)
    tokens = tokenize(text, remove_stopwords=True)
    for abbrev in medical_abbrevs:
        if abbrev.lower() not in tokens:
            tokens.append(abbrev.lower())
    return list(set(tokens))


# ─────────────────────────────────────────────
# SEARCH ENGINE - Inverted Index
# ─────────────────────────────────────────────
class InvertedIndex:
    """Teskari Indeks klassi."""

    def __init__(self, use_medical_tokenizer: bool = True):
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_lengths: Dict[str, int] = {}
        self.documents: Dict[str, str] = {}
        self.total_docs: int = 0
        self.use_medical = use_medical_tokenizer

    def add_document(self, doc_id: str, text: str) -> None:
        doc_id = str(doc_id)
        tokens = tokenize_medical(text) if self.use_medical else tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.documents[doc_id] = text
        self.total_docs += 1
        for position, token in enumerate(tokens):
            self.index[token][doc_id].append(position)

    def remove_document(self, doc_id: str) -> None:
        doc_id = str(doc_id)
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]
            self.total_docs = max(0, self.total_docs - 1)
        tokens_to_remove = []
        for token, docs in self.index.items():
            if doc_id in docs:
                del docs[doc_id]
            if not docs:
                tokens_to_remove.append(token)
        for token in tokens_to_remove:
            del self.index[token]

    def search_or(self, query: str) -> List[str]:
        tokens = tokenize_medical(query) if self.use_medical else tokenize(query)
        all_docs = set()
        for token in tokens:
            if token in self.index:
                all_docs.update(self.index[token].keys())
        return list(all_docs)

    def get_document_frequency(self, token: str) -> int:
        return len(self.index.get(token, {}))

    def get_term_frequency_in_doc(self, token: str, doc_id: str) -> int:
        return len(self.index.get(token, {}).get(str(doc_id), []))

    def to_dict(self) -> dict:
        return {
            'index': {k: dict(v) for k, v in self.index.items()},
            'doc_lengths': self.doc_lengths,
            'documents': self.documents,
            'total_docs': self.total_docs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InvertedIndex':
        obj = cls()
        obj.doc_lengths = data.get('doc_lengths', {})
        obj.documents = data.get('documents', {})
        obj.total_docs = data.get('total_docs', 0)
        for token, docs in data.get('index', {}).items():
            for doc_id, positions in docs.items():
                obj.index[token][doc_id] = positions
        return obj


# ─────────────────────────────────────────────
# SEARCH ENGINE - Ranking (TF-IDF & BM25)
# ─────────────────────────────────────────────
class TFIDF:
    def __init__(self, index: InvertedIndex):
        self.index = index

    def tf(self, token: str, doc_id: str) -> float:
        freq = self.index.get_term_frequency_in_doc(token, doc_id)
        doc_len = self.index.doc_lengths.get(str(doc_id), 1)
        return freq / doc_len if doc_len > 0 else 0.0

    def idf(self, token: str) -> float:
        N = self.index.total_docs
        df = self.index.get_document_frequency(token)
        if df == 0 or N == 0:
            return 0.0
        return math.log((N + 1) / (df + 1)) + 1

    def score(self, query: str, doc_id: str) -> float:
        tokens = tokenize_medical(query) if self.index.use_medical else tokenize(query)
        return sum(self.tf(t, doc_id) * self.idf(t) for t in tokens)

    def rank_documents(self, query: str, doc_ids: List[str]) -> List[Tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in doc_ids]
        return sorted(scored, key=lambda x: x[1], reverse=True)


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self._avg_doc_length = self._compute_avg_doc_length()

    def _compute_avg_doc_length(self) -> float:
        if not self.index.doc_lengths:
            return 0.0
        return sum(self.index.doc_lengths.values()) / len(self.index.doc_lengths)

    def update_avg_doc_length(self) -> None:
        self._avg_doc_length = self._compute_avg_doc_length()

    def idf(self, token: str) -> float:
        N = self.index.total_docs
        df = self.index.get_document_frequency(token)
        if df == 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: str) -> float:
        tokens = tokenize_medical(query) if self.index.use_medical else tokenize(query)
        doc_id = str(doc_id)
        doc_len = self.index.doc_lengths.get(doc_id, 0)
        avg_dl = self._avg_doc_length if self._avg_doc_length > 0 else 1
        total_score = 0.0
        for token in tokens:
            tf = self.index.get_term_frequency_in_doc(token, doc_id)
            idf = self.idf(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_dl)
            if denominator > 0:
                total_score += idf * (numerator / denominator)
        return total_score

    def rank_documents(self, query: str, doc_ids: List[str]) -> List[Tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in doc_ids]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def search_and_rank(self, query: str) -> List[Tuple[str, float]]:
        candidate_docs = self.index.search_or(query)
        if not candidate_docs:
            return []
        return self.rank_documents(query, candidate_docs)


# ─────────────────────────────────────────────
# Global Indeks (in-memory cache)
# ─────────────────────────────────────────────
_search_index = InvertedIndex(use_medical_tokenizer=True)
_bm25: Optional[BM25] = None
_tfidf: Optional[TFIDF] = None
_is_initialized = False


def initialize_index():
    """Bazadagi barcha hujjatlarni indeksga yuklaydi."""
    global _is_initialized, _search_index, _bm25, _tfidf

    if _is_initialized:
        return

    # Avval search_index.json dan yuklashga harakat qilish
    index_path = BASE_DIR / 'search_index.json'
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _search_index = InvertedIndex.from_dict(data)
            _bm25 = BM25(_search_index)
            _tfidf = TFIDF(_search_index)
            _is_initialized = True
            print(f"[Search] JSON indeksdan yuklandi: {_search_index.total_docs} hujjat")
            return
        except Exception as e:
            print(f"[Search] JSON yuklashda xatolik: {e}")

    # DB dan yuklash
    try:
        docs = Document.objects.all()
        for doc in docs:
            _search_index.add_document(str(doc.id), doc.matn)
        _bm25 = BM25(_search_index)
        _tfidf = TFIDF(_search_index)
        _is_initialized = True
        print(f"[Search] DB dan yuklandi: {len(docs)} hujjat")
    except Exception as e:
        print(f"[Search] Indeks yuklashda xatolik: {e}")


# ─────────────────────────────────────────────
# HTML Template (inline)
# ─────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="uz">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tibbiyot NLP Qidiruv Tizimi</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f0fdf4; color: #064e3b; }
        .glass { background: rgba(255,255,255,0.8); backdrop-filter: blur(12px); border: 1px solid rgba(16,185,129,0.2); }
        .tab-active { background-color: #059669; color: white; box-shadow: 0 4px 12px rgba(5,150,105,0.3); }
        .animate-fade-in { animation: fadeIn 0.4s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
</head>
<body class="min-h-screen flex flex-col">
    <header class="sticky top-0 z-50 w-full glass border-b border-emerald-100 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-emerald-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-emerald-200">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight">Tibbiyot <span class="text-emerald-600">NLP Tizimi</span></h1>
                <p class="text-[10px] uppercase tracking-widest text-emerald-500/70 font-bold">Medical IR Platform - BM25 & TF-IDF</p>
            </div>
        </div>
        <div id="stats-header" class="hidden md:flex items-center gap-6 text-sm font-medium text-emerald-800">
            <div class="flex flex-col items-end">
                <span id="stat-docs" class="text-emerald-600 font-bold">0</span>
                <span class="text-[10px] text-emerald-400 uppercase">Hujjatlar</span>
            </div>
            <div class="w-px h-6 bg-emerald-100"></div>
            <div class="flex flex-col items-end">
                <span id="stat-tokens" class="text-emerald-600 font-bold">0</span>
                <span class="text-[10px] text-emerald-400 uppercase">Tokenlar</span>
            </div>
        </div>
    </header>

    <main class="flex-1 flex flex-col md:flex-row max-w-[1600px] mx-auto w-full p-4 md:p-8 gap-8">
        <aside class="w-full md:w-64 flex flex-col gap-2 shrink-0">
            <button onclick="switchTab('search')" id="tab-btn-search" class="flex items-center gap-3 px-4 py-3 rounded-2xl font-semibold transition-all hover:bg-white hover:shadow-sm tab-active text-left">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                Qidiruv
            </button>
            <button onclick="switchTab('tokens')" id="tab-btn-tokens" class="flex items-center gap-3 px-4 py-3 rounded-2xl font-semibold text-emerald-600 transition-all hover:bg-white hover:shadow-sm text-left">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/><path d="M18 12a2 2 0 0 0 0 4h4v-4Z"/></svg>
                Tibbiy Tokenlar
            </button>
            <button onclick="switchTab('index')" id="tab-btn-index" class="flex items-center gap-3 px-4 py-3 rounded-2xl font-semibold text-emerald-600 transition-all hover:bg-white hover:shadow-sm text-left">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg>
                Teskari Indeks
            </button>
            <button onclick="switchTab('slots')" id="tab-btn-slots" class="flex items-center gap-3 px-4 py-3 rounded-2xl font-semibold text-emerald-600 transition-all hover:bg-white hover:shadow-sm text-left">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
                Slotlar (NLP)
            </button>
        </aside>

        <div class="flex-1 min-w-0">
            <!-- Search Tab -->
            <section id="tab-search" class="tab-content animate-fade-in">
                <div class="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-200 mb-8">
                    <h2 class="text-2xl font-extrabold mb-6">Axborot qidiruvi (IR)</h2>
                    <div class="flex flex-col md:flex-row gap-4 mb-6">
                        <div class="flex-1 relative">
                            <input type="text" id="search-query" placeholder="Masalan: yurak, qon bosimi, tashxis..." class="w-full pl-12 pr-4 py-4 rounded-2xl border border-emerald-100 focus:ring-4 focus:ring-emerald-50 focus:border-emerald-400 outline-none transition-all text-lg font-medium" onkeydown="if(event.key==='Enter') performSearch()">
                            <svg class="absolute left-4 top-1/2 -translate-y-1/2 text-emerald-400" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                        </div>
                        <div class="flex gap-2">
                            <select id="search-method" class="px-4 py-4 rounded-2xl border border-emerald-100 bg-emerald-50/30 font-bold text-emerald-800 outline-none">
                                <option value="bm25">BM25</option>
                                <option value="tfidf">TF-IDF</option>
                            </select>
                            <button onclick="performSearch()" class="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-4 rounded-2xl font-bold shadow-lg shadow-emerald-200 transition-all hover:-translate-y-1 active:translate-y-0">
                                QIDIRISH
                            </button>
                        </div>
                    </div>
                    <div id="search-loading" class="hidden flex justify-center py-12">
                        <div class="w-10 h-10 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin"></div>
                    </div>
                    <div id="search-results" class="space-y-4">
                        <div class="text-center py-12 text-slate-400"><p>Qidiruv natijalari bu yerda chiqadi.</p></div>
                    </div>
                </div>
            </section>

            <!-- Tokens Tab -->
            <section id="tab-tokens" class="tab-content hidden animate-fade-in">
                <div class="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-200 overflow-hidden">
                    <div class="flex justify-between items-center mb-8">
                        <div>
                            <h2 class="text-2xl font-extrabold">Tokenlar chastotasi</h2>
                            <p class="text-slate-400 text-sm mt-1">Eng ko'p ishlatilgan tibbiy terminlar statistikasi.</p>
                        </div>
                        <button onclick="loadTokens()" class="text-emerald-600 font-bold flex items-center gap-2 hover:bg-emerald-50 px-4 py-2 rounded-xl transition-all">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>
                            Yangilash
                        </button>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left border-collapse">
                            <thead>
                                <tr class="border-b border-slate-100 text-[10px] uppercase tracking-widest font-extrabold text-slate-400">
                                    <th class="py-4 px-4">#</th>
                                    <th class="py-4 px-4">Token</th>
                                    <th class="py-4 px-4 text-right">Chastota</th>
                                    <th class="py-4 px-4 text-right">Hujjatlar</th>
                                    <th class="py-4 px-4 text-right">Trend</th>
                                </tr>
                            </thead>
                            <tbody id="tokens-table-body"></tbody>
                        </table>
                    </div>
                </div>
            </section>

            <!-- Inverted Index Tab -->
            <section id="tab-index" class="tab-content hidden animate-fade-in">
                <div class="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-200">
                    <h2 class="text-2xl font-extrabold mb-2">Teskari indeksatsiya</h2>
                    <p class="text-slate-400 text-sm mb-8">Har bir token qaysi hujjatlarda uchrashi haqida ma'lumot.</p>
                    <div id="index-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
                </div>
            </section>

            <!-- Slots Tab -->
            <section id="tab-slots" class="tab-content hidden animate-fade-in">
                <div class="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-200">
                    <h2 class="text-2xl font-extrabold mb-2">Soha bo'yicha slotlarga ajratish</h2>
                    <p class="text-slate-400 text-sm mb-8">Tibbiy gaplarni mantiqiy qismlarga (Subject, Predicate, Object) ajratish.</p>
                    <div class="space-y-6" id="slots-container"></div>
                </div>
            </section>
        </div>
    </main>

    <footer class="p-8 text-center text-slate-400 text-sm">
        &copy; 2026 MedIR System - NLP Medical Search Dashboard. Barcha huquqlar himoyalangan.
    </footer>

    <script>
        const API_BASE = '/api';

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.getElementById('tab-' + tabId).classList.remove('hidden');
            document.querySelectorAll('aside button').forEach(btn => {
                btn.classList.remove('tab-active');
                btn.classList.add('text-emerald-600');
            });
            const btn = document.getElementById('tab-btn-' + tabId);
            btn.classList.add('tab-active');
            btn.classList.remove('text-emerald-600');
            if (tabId === 'tokens') loadTokens();
            if (tabId === 'index') loadIndex();
            if (tabId === 'slots') loadSlots();
        }

        async function loadStats() {
            try {
                const res = await fetch(`${API_BASE}/stats/`);
                const data = await res.json();
                document.getElementById('stat-docs').innerText = data.total_documents.toLocaleString();
                document.getElementById('stat-tokens').innerText = data.unique_tokens.toLocaleString();
                document.getElementById('stats-header').classList.remove('hidden');
            } catch(e) { console.error(e); }
        }

        async function performSearch() {
            const query = document.getElementById('search-query').value;
            const method = document.getElementById('search-method').value;
            if (!query) return;
            const resultsDiv = document.getElementById('search-results');
            const loadingDiv = document.getElementById('search-loading');
            resultsDiv.innerHTML = '';
            loadingDiv.classList.remove('hidden');
            try {
                const res = await fetch(`${API_BASE}/search/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, method, top_k: 15 })
                });
                const data = await res.json();
                loadingDiv.classList.add('hidden');
                if (data.results && data.results.length > 0) {
                    resultsDiv.innerHTML = `<p class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Top ${data.results.length} ta natija (${data.method.toUpperCase()}):</p>`;
                    data.results.forEach(r => {
                        resultsDiv.innerHTML += `
                            <div class="p-5 rounded-2xl border border-slate-100 bg-slate-50 hover:border-emerald-200 transition-all">
                                <div class="flex justify-between items-start mb-2">
                                    <span class="text-[10px] font-bold text-emerald-500 uppercase tracking-tighter">Hujjat #${r.doc_id}</span>
                                    <span class="px-2 py-1 bg-white rounded-lg text-xs font-mono font-bold text-slate-500 border border-slate-100">Score: ${r.score}</span>
                                </div>
                                <p class="text-slate-700 leading-relaxed">${r.content || 'Matn yuklanmoqda...'}</p>
                                ${r.label ? `<span class="mt-2 inline-block text-[10px] bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-bold">${r.label}</span>` : ''}
                            </div>`;
                    });
                } else {
                    resultsDiv.innerHTML = '<div class="text-center py-12 text-slate-400">Hech narsa topilmadi.</div>';
                }
            } catch(e) {
                loadingDiv.classList.add('hidden');
                resultsDiv.innerHTML = '<div class="text-red-500 p-4">Xatolik: ' + e.message + '</div>';
            }
        }

        async function loadTokens() {
            const body = document.getElementById('tokens-table-body');
            body.innerHTML = '<tr><td colspan="5" class="text-center py-12">Yuklanmoqda...</td></tr>';
            try {
                const res = await fetch(`${API_BASE}/tokens/`);
                const data = await res.json();
                body.innerHTML = '';
                data.tokens.forEach((t, i) => {
                    const perc = Math.min(100, (t.frequency / data.tokens[0].frequency) * 100);
                    body.innerHTML += `
                        <tr class="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                            <td class="py-4 px-4 text-xs font-bold text-slate-300">#${i+1}</td>
                            <td class="py-4 px-4 font-bold text-emerald-900">${t.token}</td>
                            <td class="py-4 px-4 text-right font-mono font-bold text-slate-600">${t.frequency.toLocaleString()}</td>
                            <td class="py-4 px-4 text-right text-slate-500">${t.doc_count.toLocaleString()}</td>
                            <td class="py-4 px-4 text-right w-32">
                                <div class="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                    <div class="h-full bg-emerald-500 rounded-full" style="width: ${perc}%"></div>
                                </div>
                            </td>
                        </tr>`;
                });
            } catch(e) { console.error(e); }
        }

        async function loadIndex() {
            const container = document.getElementById('index-container');
            container.innerHTML = '<div class="col-span-full text-center py-12">Yuklanmoqda...</div>';
            try {
                const res = await fetch(`${API_BASE}/inverted-index/`);
                const data = await res.json();
                container.innerHTML = '';
                Object.entries(data.index_sample).forEach(([token, docs]) => {
                    container.innerHTML += `
                        <div class="p-6 rounded-2xl bg-slate-50 border border-slate-100 hover:shadow-md transition-all">
                            <div class="flex items-center gap-2 mb-4">
                                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                                <h3 class="font-bold text-emerald-950">${token}</h3>
                            </div>
                            <div class="flex flex-wrap gap-1.5">
                                ${docs.slice(0, 10).map(id => `<span class="px-2 py-0.5 bg-white rounded-md text-[10px] font-bold text-slate-400 border border-slate-100">#${id}</span>`).join('')}
                                ${docs.length > 10 ? `<span class="text-[10px] text-slate-300 ml-1">+${docs.length - 10} yana</span>` : ''}
                            </div>
                        </div>`;
                });
            } catch(e) { console.error(e); }
        }

        async function loadSlots() {
            const container = document.getElementById('slots-container');
            container.innerHTML = '<div class="text-center py-12">Yuklanmoqda...</div>';
            try {
                const res = await fetch(`${API_BASE}/slots/`);
                const data = await res.json();
                container.innerHTML = '';
                data.results.forEach((item, i) => {
                    container.innerHTML += `
                        <div class="p-6 rounded-3xl bg-slate-50 border border-slate-100">
                            <div class="text-xs font-bold text-slate-300 mb-3 uppercase tracking-widest">Gap #${i+1}</div>
                            <p class="text-slate-800 font-medium mb-5 text-lg">"${item.original}"</p>
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                                <div class="bg-blue-100/50 p-4 rounded-2xl border border-blue-200/50">
                                    <span class="block text-[10px] font-bold text-blue-500 uppercase mb-1">Ega (Subject)</span>
                                    <span class="text-blue-900 font-bold">${item.slots.subject}</span>
                                </div>
                                <div class="bg-purple-100/50 p-4 rounded-2xl border border-purple-200/50">
                                    <span class="block text-[10px] font-bold text-purple-500 uppercase mb-1">Kesim (Predicate)</span>
                                    <span class="text-purple-900 font-bold">${item.slots.predicate || '—'}</span>
                                </div>
                                <div class="bg-amber-100/50 p-4 rounded-2xl border border-amber-200/50">
                                    <span class="block text-[10px] font-bold text-amber-500 uppercase mb-1">To'ldiruvchi (Object/Value)</span>
                                    <span class="text-amber-900 font-bold">${item.slots.object_value || '—'}</span>
                                </div>
                            </div>
                        </div>`;
                });
            } catch(e) { console.error(e); }
        }

        document.addEventListener('DOMContentLoaded', () => {
            loadStats();
        });
    </script>
</body>
</html>"""


# ─────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


def dashboard_view(request):
    """Asosiy dashboard sahifasi."""
    return HttpResponse(DASHBOARD_HTML, content_type='text/html')


@csrf_exempt
@require_http_methods(["POST"])
def search_documents(request):
    """
    Hujjatlarni qidirish API.
    POST /api/search/
    Body: {"query": "yurak", "method": "bm25", "top_k": 10}
    """
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        method = data.get('method', 'bm25').lower()
        top_k = int(data.get('top_k', 10))

        if not query:
            return JsonResponse({'error': "So'rov bo'sh bo'lishi mumkin emas"}, status=400)

        initialize_index()

        if method == 'bm25' and _bm25:
            ranked = _bm25.search_and_rank(query)
        elif method == 'tfidf' and _tfidf:
            candidate_docs = _search_index.search_or(query)
            ranked = _tfidf.rank_documents(query, candidate_docs)
        else:
            return JsonResponse({'error': 'Method noto\'g\'ri. "bm25" yoki "tfidf" tanlang'}, status=400)

        results = []
        for doc_id, score in ranked[:top_k]:
            if score > 0:
                try:
                    db_id = int(float(doc_id))
                    doc_obj = Document.objects.get(pk=db_id)
                    results.append({
                        'doc_id': db_id,
                        'score': round(score, 4),
                        'content': doc_obj.matn[:400] + "..." if len(doc_obj.matn) > 400 else doc_obj.matn,
                        'label': doc_obj.label or "Tibbiy hujjat"
                    })
                except Exception:
                    content = _search_index.documents.get(str(doc_id), "")
                    results.append({
                        'doc_id': doc_id,
                        'score': round(score, 4),
                        'content': content[:400] + "..." if len(content) > 400 else content
                    })

        return JsonResponse({
            'query': query,
            'method': method,
            'total_found': len(results),
            'results': results,
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': "JSON formati noto'g'ri"}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_index_stats(request):
    """Indeks statistikasi. GET /api/stats/"""
    initialize_index()
    return JsonResponse({
        'total_documents': _search_index.total_docs,
        'unique_tokens': len(_search_index.index),
        'index_size_tokens': sum(len(docs) for docs in _search_index.index.values()),
    })


def get_tokens_data(request):
    """Tokenlar chastotasi. GET /api/tokens/"""
    initialize_index()
    token_freqs = []
    for token, docs in _search_index.index.items():
        total_count = sum(len(positions) for positions in docs.values())
        token_freqs.append({
            'token': token,
            'frequency': total_count,
            'doc_count': len(docs)
        })
    token_freqs.sort(key=lambda x: x['frequency'], reverse=True)
    return JsonResponse({
        'total_unique_tokens': len(token_freqs),
        'tokens': token_freqs[:500]
    })


def get_inverted_index_data(request):
    """Teskari indeks namunasi. GET /api/inverted-index/"""
    initialize_index()
    limit = int(request.GET.get('limit', 100))
    sample_index = {}
    for i, (token, docs) in enumerate(_search_index.index.items()):
        if i >= limit:
            break
        sample_index[token] = list(docs.keys())
    return JsonResponse({
        'index_sample': sample_index,
        'total_tokens': len(_search_index.index)
    })


def get_medical_slots(request):
    """Tibbiy gaplarni slotlarga ajratish. GET /api/slots/"""
    initialize_index()
    # Indeksdan hujjatlarni olish (birinchi 50 ta)
    doc_items = list(_search_index.documents.items())[:50]
    results = []
    for doc_id, text in doc_items:
        words = text.split()
        if len(words) >= 3:
            results.append({
                'original': text[:200],
                'slots': {
                    'subject': words[0],
                    'predicate': words[1],
                    'object_value': " ".join(words[2:min(10, len(words))])
                }
            })
        elif words:
            results.append({
                'original': text[:200],
                'slots': {'subject': text[:50], 'predicate': '', 'object_value': ''}
            })
    return JsonResponse({'results': results})


@csrf_exempt
@require_http_methods(["DELETE"])
def remove_from_index(request, doc_id: str):
    """Hujjatni indeksdan o'chirish. DELETE /api/index/<doc_id>/"""
    try:
        _search_index.remove_document(doc_id)
        return JsonResponse({'success': True, 'doc_id': doc_id})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ─────────────────────────────────────────────
# URL PATTERNS
# ─────────────────────────────────────────────
from django.urls import path, include

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('api/search/', search_documents, name='search'),
    path('api/stats/', get_index_stats, name='stats'),
    path('api/tokens/', get_tokens_data, name='tokens'),
    path('api/inverted-index/', get_inverted_index_data, name='inverted-index'),
    path('api/slots/', get_medical_slots, name='slots'),
    path('api/index/<str:doc_id>/', remove_from_index, name='remove-index'),
]


# WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()


# ─────────────────────────────────────────────
# MANAGEMENT COMMANDS (manage.py vazifasini bajaradi)
# ─────────────────────────────────────────────
def run_migrate():
    """Ma'lumotlar bazasini yaratadi."""
    from django.core.management import call_command
    # Document modelini yaratish
    from django.db import connection
    with connection.schema_editor() as schema_editor:
        try:
            schema_editor.create_model(Document)
            print("[DB] 'document' jadvali yaratildi.")
        except Exception as e:
            if 'already exists' in str(e).lower():
                print("[DB] 'document' jadvali allaqachon mavjud.")
            else:
                print(f"[DB] Jadval yaratishda xatolik: {e}")
    # Django auth tables uchun
    call_command('migrate', '--run-syncdb', verbosity=0)
    print("[DB] Migratsiya tugadi.")


def run_loaddata():
    """CSV fayldan ma'lumot yuklaydi."""
    csv_path = BASE_DIR / 'Tibbiyot.csv'
    if not csv_path.exists():
        print(f"[Data] XATO: {csv_path} topilmadi!")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[Data] pandas o'rnatilmagan. pip install pandas")
        return

    print(f"[Data] {csv_path} yuklanmoqda...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Kerakli ustunlar
    matn_col = 'Matn' if 'Matn' in df.columns else df.columns[2]
    label_col = 'Label' if 'Label' in df.columns else None
    manba_col = 'manba_fayl' if 'manba_fayl' in df.columns else None

    count = 0
    batch = []
    for _, row in df.iterrows():
        matn = str(row.get(matn_col, '')).strip()
        if not matn or matn == 'nan':
            continue
        label = str(row.get(label_col, '')) if label_col else ''
        manba = str(row.get(manba_col, '')) if manba_col else ''
        batch.append(Document(matn=matn, label=label[:100], manba_fayl=manba[:255]))
        count += 1
        if len(batch) >= 1000:
            Document.objects.bulk_create(batch, ignore_conflicts=True)
            batch = []
            print(f"  ... {count} ta hujjat yuklandi")

    if batch:
        Document.objects.bulk_create(batch, ignore_conflicts=True)

    print(f"[Data] Jami {count} ta hujjat bazaga yuklandi.")

    # Indeks yaratish va saqlash
    print("[Index] Indeks qurilmoqda...")
    idx = InvertedIndex(use_medical_tokenizer=True)
    for doc in Document.objects.all():
        idx.add_document(str(doc.id), doc.matn)

    index_path = BASE_DIR / 'search_index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(idx.to_dict(), f, ensure_ascii=False)
    print(f"[Index] {index_path} ga saqlandi. {idx.total_docs} hujjat, {len(idx.index)} token.")


def run_server(host='0.0.0.0', port=8000):
    """Development serverini ishga tushiradi."""
    from django.core.management import call_command
    initialize_index()
    call_command('runserver', f'{host}:{port}')


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
Tibbiyot NLP Qidiruv Tizimi

Buyruqlar:
  python app.py migrate      - Ma'lumotlar bazasini yaratish
  python app.py loaddata     - CSV dan ma'lumot yuklash va indeks qurish
  python app.py runserver    - Serverni ishga tushirish (http://localhost:8000)
  python app.py runserver 0.0.0.0:8080  - Boshqa portda ishga tushirish
        """)
        sys.exit(0)

    command = sys.argv[1]

    if command == 'migrate':
        run_migrate()
    elif command == 'loaddata':
        run_migrate()
        run_loaddata()
    elif command == 'runserver':
        host_port = sys.argv[2] if len(sys.argv) > 2 else '0.0.0.0:8000'
        if ':' in host_port:
            h, p = host_port.split(':', 1)
            run_server(h, int(p))
        else:
            run_server(port=int(host_port))
    else:
        # Django management command sifatida o'tkazish
        from django.core.management import execute_from_command_line
        execute_from_command_line(sys.argv)
