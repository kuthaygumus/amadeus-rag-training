// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { satteri } from '@astrojs/markdown-satteri';

/** Module spine — mirrors the GATE chain. Each entry is one stage of the day. */
const modules = [
  ['00-setup',     '0. Setup — Before You Arrive',        '0. Kurulum — Gelmeden Önce'],
  ['01-bare-llm',  '1. The Bare LLM Wall',                '1. Çıplak LLM Duvarı'],
  ['02-neural-net','2. How a Neural Network Learns',      '2. Sinir Ağı Nasıl Öğrenir'],
  ['03-finetune',  '3. Fine-Tuning: Your Own Model',      '3. Fine-Tuning: Kendi Modelin'],
  ['04-stuff-prompt','4. Just Stuff the Prompt',          "4. Hepsini Prompt'a Doldur"],
  ['05-naive-rag', '5. Keyword Search to Naive RAG',      "5. Keyword Search'ten Naive RAG'e"],
  ['06-embeddings','6. The Embedding Bake-Off',           '6. Embedding Bake-Off'],
  ['07-chunking',  '7. Chunking, Noise and Measurement',  '7. Chunking, Gürültü ve Ölçüm'],
  ['08-chromadb',  '8. ChromaDB: Embedded vs Server',     '8. ChromaDB: Embedded ve Server'],
  ['09-hybrid',    '9. Hybrid, Rerank and Contextual',    '9. Hybrid, Rerank ve Contextual'],
  ['10-agentic',   '10. Agentic RAG — The Finale',        '10. Agentic RAG — Finale'],
];

const PRESENTER_NOTE_OPEN_TAG = '<div class="presenter-note">';

/**
 * Keep presenter notes out of the Pagefind search index.
 *
 * `.presenter-note { display: none }` only hides the notes visually. Pagefind reads the
 * built HTML, so without this the stage directions — the polls, the planted hand counts,
 * the result a module is not meant to reveal yet — come back as search results to anyone
 * in the room. `data-pagefind-ignore="all"` drops the element from the index and from the
 * result excerpt.
 *
 * The notes are written as raw HTML blocks in the Markdown, so Sätteri hands them to the
 * pipeline as `raw` nodes rather than parsed elements; the opening tag is rewritten in
 * place. Doing it here rather than in the pages keeps the attribute out of 74 hand-edited
 * spots and applies to any note added later.
 */
const excludePresenterNotesFromSearch = {
  name: 'exclude-presenter-notes-from-search',
  raw(node) {
    const value = String(node.value);
    if (!value.includes(PRESENTER_NOTE_OPEN_TAG)) return;
    return {
      ...node,
      value: value.replaceAll(
        PRESENTER_NOTE_OPEN_TAG,
        '<div class="presenter-note" data-pagefind-ignore="all">',
      ),
    };
  },
};

/**
 * Presenter mode. Off by default, so the room only ever sees the page.
 *
 * Kral turns it on with `?presenter=1` in the address bar or with Alt+Shift+P, and the
 * choice is remembered in localStorage for the rest of the day. `?presenter=0` or a second
 * Alt+Shift+P turns it back off. The state is set on <html> before the body renders, so the
 * notes never flash. There is no visible control: nothing in the page invites a participant
 * to discover the mechanism.
 */
const presenterModeScript = `
(function () {
  var KEY = 'rag-training-presenter';
  var root = document.documentElement;
  var on = false;

  function persist(value) {
    try {
      if (value) localStorage.setItem(KEY, 'on');
      else localStorage.removeItem(KEY);
    } catch (error) { /* private browsing: presenter mode stays per-page */ }
  }

  function stored() {
    try { return localStorage.getItem(KEY) === 'on'; } catch (error) { return false; }
  }

  // The badge exists only while presenter mode is on, so a participant who opens the
  // inspector on a normal page finds nothing about it in the DOM.
  function syncBadge() {
    if (!document.body) return;
    var badge = document.getElementById('presenter-mode-badge');
    if (on && !badge) {
      badge = document.createElement('div');
      badge.id = 'presenter-mode-badge';
      badge.className = 'presenter-badge';
      badge.textContent = 'PRESENTER MODE · Alt+Shift+P';
      document.body.appendChild(badge);
    } else if (!on && badge) {
      badge.remove();
    }
  }

  function apply(value) {
    on = value;
    if (value) root.dataset.presenter = 'true';
    else delete root.dataset.presenter;
    syncBadge();
  }

  var param = new URLSearchParams(location.search).get('presenter');
  if (param === null) {
    apply(stored());
  } else {
    apply(param === '1' || param === 'on' || param === 'true');
    persist(on);
  }

  document.addEventListener('keydown', function (event) {
    if (!event.altKey || !event.shiftKey || event.ctrlKey || event.metaKey) return;
    if (event.code !== 'KeyP') return;
    event.preventDefault();
    apply(!on);
    persist(on);
  });

  document.addEventListener('DOMContentLoaded', syncBadge);
})();
`.trim();

export default defineConfig({
  site: 'https://amadeus-rag-training.vercel.app',
  markdown: {
    processor: satteri({ hastPlugins: [excludePresenterNotesFromSearch] }),
  },
  integrations: [
    starlight({
      title: 'RAG Training Day',
      description:
        'A one-day, measurement-driven RAG course. Runs fully offline on a locked-down corporate laptop.',
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        tr: { label: 'Türkçe', lang: 'tr' },
      },
      customCss: ['./src/styles/custom.css'],
      head: [{ tag: 'script', content: presenterModeScript }],
      // The fictional-airline notice sits once at the bottom of every page instead of once at
      // the top of all 26 of them. src/components/Footer.astro wraps Starlight's own footer.
      components: { Footer: './src/components/Footer.astro' },
      // The 404 response is served by public/404.html, which covers both locales.
      disable404Route: true,
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/kuthaygumus/amadeus-rag-training' },
      ],
      sidebar: [
        {
          label: 'Start Here',
          translations: { tr: 'Buradan Başla' },
          items: [{ label: 'Overview', slug: 'index', translations: { tr: 'Genel Bakış' } }],
        },
        {
          label: 'The Day',
          translations: { tr: 'Gün' },
          items: modules.map(([slug, en, tr]) => ({
            label: en,
            slug: `modules/${slug}`,
            translations: { tr },
          })),
        },
        {
          label: 'Reference',
          translations: { tr: 'Referans' },
          items: [
            { label: 'Glossary', slug: 'reference/glossary', translations: { tr: 'Sözlük' } },
          ],
        },
      ],
    }),
  ],
});
