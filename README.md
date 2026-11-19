# namusanga.com

Portfolio site of Nicholas Namusanga. Plain Jekyll, built by GitHub Pages straight from `main`:
no theme, no plugins, no build step, no Actions workflow to maintain.

To add or change content, read [CONTENT.md](CONTENT.md).

## Layout

| Path | What it holds |
|---|---|
| `_posts/` | Writing, one markdown file per post, served at `/blog/<slug>` |
| `_data/experience.yml`, `certifications.yml`, `projects.yml`, `skills.yml` | The lists on the home, experience, certifications and projects pages |
| `_data/categories.yml`, `categories/*.md` | Post categories and their pages at `/categories/<slug>/` |
| `index.html`, `about.md`, `contact.md`, `experience.html`, ... | The pages |
| `_layouts/`, `_includes/`, `assets/css/main.css` | Templates and the one stylesheet |
| `assets/img/` | Every image on the site |
| `assets/vendor/katex/` | KaTeX 0.16.22, loaded only on pages with `math: true` |
| `resume/` | The resume PDF, linked from every page |

## How links work

Internal links are relative to the current page (`_includes/base.html` works out the prefix), so
the same build is correct at `https://namusanga.com/` and at
`https://nnamu-cl.github.io/namusanga.com/`. `baseurl` stays empty.

## Local preview

```
docker run --rm -p 4000:4000 -v "$PWD":/site -w /site ruby:3.3 \
  sh -c "gem install github-pages webrick && jekyll serve --host 0.0.0.0"
```

## Going live on namusanga.com

`main` carries no `CNAME` file on purpose. GitHub couples that file to the custom domain setting,
and once it is set the `nnamu-cl.github.io/namusanga.com/` preview redirects to `namusanga.com`.
Pull request #1 ("Go live") adds the file and lists the DNS records; merge it on switch day.
