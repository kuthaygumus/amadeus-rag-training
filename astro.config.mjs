// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

/** Module spine — mirrors the GATE chain. Each entry is one stage of the day. */
const modules = [
  ['00-setup',     '0. Setup — Before You Arrive',        '0. Kurulum — Gelmeden Önce'],
  ['01-bare-llm',  '1. The Bare LLM Wall',                '1. Çıplak LLM Duvarı'],
  ['02-neural-net','2. How a Neural Network Learns',      '2. Sinir Ağı Nasıl Öğrenir'],
  ['03-finetune',  '3. Fine-Tuning: Your Own Model',      '3. Fine-Tuning: Kendi Modelin'],
  ['04-stuff-prompt','4. Just Stuff the Prompt',          '4. Hepsini Prompt’a Doldur'],
  ['05-naive-rag', '5. Keyword Search to Naive RAG',      '5. Keyword Aramadan Naive RAG’a'],
  ['06-embeddings','6. The Embedding Bake-Off',           '6. Embedding Bake-Off'],
  ['07-chunking',  '7. Chunking, Noise and Measurement',  '7. Chunking, Gürültü ve Ölçüm'],
  ['08-chromadb',  '8. ChromaDB: Embedded vs Server',     '8. ChromaDB: Embedded vs Server'],
  ['09-hybrid',    '9. Hybrid, Rerank and Contextual',    '9. Hybrid, Rerank ve Contextual'],
  ['10-agentic',   '10. Agentic RAG — The Finale',        '10. Agentic RAG — Final'],
];

export default defineConfig({
  site: 'https://amadeus-rag-training.vercel.app',
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
