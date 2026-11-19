# Editing this site from a phone

Every page is a text file in this repo. Committing a change rebuilds the site in about a minute.

1. Open `nnamu-cl/namusanga.com` in the GitHub mobile app (or github.com in the phone browser) and browse the code.
2. New post: in `_posts/` create a file named `YYYY-MM-DD-my-slug.md`. The slug becomes the address, `/blog/my-slug`.
3. Start the file with the block below, then write markdown under it. `categories` must use names from `_data/categories.yml`.
4. Images: upload to `assets/img/`, then write `{% include figure.html src="my-figure.png" alt="What it shows" label="Figure 1" caption="Optional caption" %}`.
5. Formulas go between `$$` and `$$` (set `math: true`). Side notes and code blocks: copy one from an existing post.
6. New experience entry: edit `_data/experience.yml`, copy an existing entry to the top and change it. Keep the two-space indentation.
7. Same pattern for `_data/certifications.yml`, `_data/projects.yml` and `_data/skills.yml`. Home hero text is at the top of `index.html`; sidebar text, email and links are in `_config.yml`.
8. Links from a post to another page start with `../`, for example `[about](../about)`. No em or en dashes anywhere on the site.
9. Commit. If the change does not show up, open the Actions tab: a failed "pages build and deployment" run names the file and line.
10. Not set up, an option for later: a browser editor such as Decap CMS on top of these same files. It needs an OAuth proxy, so plain commits are the CMS for now.

```yaml
---
title: My title
date: 2026-12-01 09:00:00 +0300
categories: [CUDA, Profiling]
cover: my-cover.png          # file in assets/img/, optional
description: One sentence for search results and link previews.
summary: One line shown under the title on the home page.
math: true                   # only when the post has formulas
---
```
