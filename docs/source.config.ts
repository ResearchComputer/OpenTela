import { defineDocs, defineConfig, frontmatterSchema } from 'fumadocs-mdx/config';
import { z } from 'zod';

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: frontmatterSchema.extend({
      experimental: z.boolean().optional(),
      tldr: z.string().optional(),
      date: z.string().optional(),
    }),
  },
});

export default defineConfig();
