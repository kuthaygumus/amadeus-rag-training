import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Only `docs` is declared. Starlight reads an `i18n` collection unconditionally to resolve
// its UI strings and warns when it finds none, but this site uses Starlight's bundled English
// and Turkish strings unchanged, so there is nothing to put in that collection. Declaring it
// empty is worse than leaving it out: the loader then warns about the missing directory too.
// Measured on this repo — no declaration: 1 warning; declaration, no directory: 2 warnings;
// declaration plus any file under src/content/i18n/: 0 warnings.
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
