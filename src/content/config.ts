import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
	loader: glob({
		pattern: '**/*.md',
		base: 'content/blog',
	}),
	schema: z.object({
		title: z.string().min(1),
		description: z.string().min(1),
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		heroImage: z.string().optional(),
		pinned: z.boolean().optional(),
		tags: z.array(
			z.string()
				.min(1)
				.transform((tag) => tag.toLowerCase())
		).min(1),
		draft: z.boolean().optional(),
	}),
});

export const collections = { blog };
