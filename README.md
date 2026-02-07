# AI Learning Platform

A structured learning platform for Artificial Intelligence, migrated from Google Blogger to GitHub Pages with Jekyll.

## Features

### Core Features
- **Three Learning Paths**: Beginner, Intermediate, and Advanced
- **Topic Categories**: Machine Learning, Deep Learning, NLP, Computer Vision, Generative AI, and AI Ethics
- **GitHub Pages**: Free, fast, reliable hosting
- **Jekyll Powered**: Static site generator for performance

### New Features
- **Blog Index Page**: Dedicated page for browsing all articles with category filtering
- **Search Functionality**: Client-side search to find articles quickly
- **Enhanced Navigation**: Sticky header with improved navigation and breadcrumbs
- **Related Posts**: Automatic related article suggestions
- **Author Info**: Author box on blog posts
- **Custom 404 Page**: Friendly error page with popular article suggestions
- **Responsive Design**: Mobile-friendly layout
- **RSS Feed**: Automatic feed generation at `/feed.xml`
- **SEO Optimized**: Meta tags and sitemap generation

## Learning Paths

| Path | Description | Icon |
|------|-------------|------|
| Beginner | Start with AI fundamentals | 🌱 |
| Intermediate | Build core ML knowledge | 🌿 |
| Advanced | Master specialized techniques | 🌳 |

## Topics

- Machine Learning
- Deep Learning
- Natural Language Processing
- Computer Vision
- Generative AI
- AI Ethics

## Quick Start

1. Choose a [Learning Path](/learning_paths) based on your level
2. Browse [Topics](/topics) to explore specific areas
3. Read posts and practice what you learn
4. Use [Search](/search) to find specific content

## Development

This site uses Jekyll and GitHub Pages.

### Prerequisites
- Ruby 2.7 or higher
- Bundler
- Git

### Local Setup

```bash
# Install dependencies
bundle install

# Start local server
bundle exec jekyll serve --baseurl "/rootForAI"

# Open in browser
# http://localhost:4000/rootForAI/
```

## Creating Content

### New Blog Post

Create a file in `_posts/` with format `YYYY-MM-DD-title.md`:

```markdown
---
layout: post
title: "Your Title"
date: 2025-12-01
path_type: beginner  # beginner, intermediate, or advanced
categories:
  - machine-learning
  - fundamentals
read_time: 10        # minutes
---

Your content in Markdown...
```

### Available Path Types
- `beginner` - Introductory content
- `intermediate` - Building on basics
- `advanced` - Specialized topics

### Available Categories
- `machine-learning`, `deep-learning`, `nlp`
- `computer-vision`, `generative-ai`, `ethics`
- `fundamentals`, `python`, `frameworks`, `transformers`

## Migration from Blogger

This platform was created to migrate content from [rootfortutorials.blogspot.com](https://rootfortutorials.blogspot.com/2025/12/).

To add your Blogger posts:
1. Export your Blogger content as XML
2. Convert posts to Jekyll Markdown format in `_posts/`
3. Add appropriate `path_type` and `categories` frontmatter
4. Update the date format to `YYYY-MM-DD`

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions.

## Deployment

This site automatically deploys to GitHub Pages when you push to the `main` branch.

Your site will be available at: `https://msmariswamy.github.io/rootForAI/`

## Site Structure

```
ai-website/
├── _config.yml              # Main configuration
├── _config_github.yml       # GitHub Pages configuration
├── _layouts/                # Layout templates
├── _posts/                  # Blog posts
├── assets/
│   └── css/                 # Stylesheets
├── learning_paths/          # Learning path pages
├── topics/                  # Topic pages
├── blog.md                  # Blog index
├── search.md                # Search page
├── 404.md                   # Custom 404
├── index.md                 # Homepage
└── about.md                 # About page
```

## Configuration

Edit `_config.yml` to customize:
- Site title and description
- Learning paths
- Topics and colors
- Social links
- Pagination settings

## Documentation

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

## License

MIT License

## Contact

For issues or questions, visit the [GitHub repository](https://github.com/msmariswamy/ai_website).