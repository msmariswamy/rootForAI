# AI Learning Platform - Usage Guide

## Overview

This is a Jekyll-based GitHub Pages site for AI learning, structured with blog posts, learning paths, and topics.

## What's New

### 1. **Blog Index Page** (`/blog`)
- Dedicated page listing all blog posts
- Category filtering by topic
- Post cards with metadata
- Reading time display

### 2. **Search Functionality** (`/search`)
- Client-side search for blog posts
- Search by title, content, and categories
- Quick filter buttons for popular topics
- Real-time results

### 3. **Enhanced Navigation**
- Sticky header with clear navigation
- Links to: Home, Paths, Topics, Blog, Search, About
- Smooth hover effects
- Breadcrumb navigation on post pages

### 4. **Improved Blog Posts**
- Post pages with full metadata
- Author information box
- Related posts section
- Category tags
- Back to blog button

### 5. **Custom 404 Page**
- Friendly error page
- Popular article suggestions
- Quick navigation back to main sections

### 6. **Enhanced Styling**
- Modern card-based design
- Responsive layout
- Hover effects and animations
- Clean typography
- Color-coded categories

## Site Structure

```
ai-website/
├── _config.yml              # Main Jekyll configuration
├── _config_github.yml       # GitHub Pages configuration
├── _layouts/
│   ├── default.html         # Base layout
│   └── post.html            # Blog post layout
├── _posts/                  # Blog posts (YYYY-MM-DD-title.md)
├── assets/
│   └── css/
│       └── style.scss       # Site styling
├── learning_paths/          # Learning path pages
├── topics/                  # Topic pages
├── blog.md                  # Blog index page
├── search.md                # Search page
├── 404.md                   # Custom 404 page
├── index.md                 # Homepage
└── about.md                 # About page
```

## Creating New Blog Posts

### Post Format

Create a new file in `_posts/` with the format `YYYY-MM-DD-title.md`:

```markdown
---
layout: post
title: "Your Post Title"
date: 2025-12-01
path_type: beginner  # beginner, intermediate, or advanced
categories:
  - machine-learning
  - deep-learning
read_time: 10        # Estimated reading time in minutes
---

Your post content goes here in Markdown format.

## Section 1

Content...

## Section 2

Content...

---
```

### Available Path Types
- `beginner` - For introductory content
- `intermediate` - For more advanced topics
- `advanced` - For expert-level content

### Available Categories
- `machine-learning` - ML fundamentals
- `deep-learning` - Neural networks and deep learning
- `nlp` - Natural Language Processing
- `computer-vision` - Image processing
- `generative-ai` - AI generation tasks
- `ethics` - AI ethics and responsibility
- `fundamentals` - Basic concepts
- `python` - Python programming
- `frameworks` - ML/AI frameworks
- `transformers` - Transformer models

## Local Development

### Prerequisites
- Ruby (version 2.7 or higher)
- Bundler
- Git

### Setup

1. Install Jekyll and Bundler:
```bash
gem install jekyll bundler
```

2. Install dependencies:
```bash
bundle install
```

3. Start local server:
```bash
bundle exec jekyll serve --baseurl "/rootForAI"
```

4. Open in browser:
```
http://localhost:4000/rootForAI/
```

### GitHub Pages Deployment

1. Push changes to your repository
2. GitHub will automatically build and deploy
3. Site available at: `https://msmariswamy.github.io/rootForAI/`

## Configuration

### Site Settings (`_config.yml`)

```yaml
title: AI Learning Platform
description: Your pathway to mastering Artificial Intelligence
url: "https://msmariswamy.github.io"
baseurl: "/rootForAI"

# Social links
social:
  github: msmariswamy
  twitter: ""

# Pagination
paginate: 6
```

### Customizing Learning Paths

Edit the `learning_paths` section in `_config.yml`:

```yaml
learning_paths:
  - id: beginner
    title: Beginner Path
    description: Start your AI journey with fundamentals
    icon: 🌱
  - id: intermediate
    title: Intermediate Path
    description: Build on your knowledge with core concepts
    icon: 🌿
  - id: advanced
    title: Advanced Path
    description: Master specialized AI techniques and applications
    icon: 🌳
```

### Customizing Topics

Edit the `topics` section in `_config.yml`:

```yaml
topics:
  - id: machine-learning
    title: Machine Learning
    color: "#3498db"
  - id: deep-learning
    title: Deep Learning
    color: "#2ecc71"
```

## Features

### 1. Responsive Design
The site is fully responsive and works on all screen sizes.

### 2. Code Highlighting
Code blocks are automatically highlighted using Rouge syntax highlighter.

### 3. RSS Feed
An RSS feed is automatically generated at `/feed.xml`.

### 4. SEO Optimization
The site includes SEO meta tags and sitemap generation.

### 5. Search
Client-side search functionality for finding blog posts.

### 6. Category Filtering
Filter blog posts by category on the blog index page.

## Tips for Good Content

1. **Use Clear Titles**: Make your post titles descriptive
2. **Add Excerpts**: Write a good opening paragraph for post previews
3. **Use Headings**: Structure your content with proper headings
4. **Include Code Examples**: Add working code examples where relevant
5. **Set Reading Time**: Estimate reading time for each post
6. **Categorize Properly**: Use relevant categories for discoverability
7. **Link to Related Content**: Connect posts using the categories system

## Troubleshooting

### Site Not Building
- Check for syntax errors in Markdown files
- Verify front matter in posts is correct
- Ensure all required plugins are installed

### Images Not Loading
- Check image paths are correct
- Ensure images are in the `assets/images/` directory
- Use `{{ '/rootForAI/assets/images/image.png' | relative_url }}`

### Search Not Working
- Ensure JavaScript is enabled in browser
- Check that blog posts have proper front matter
- Verify the search.md file exists

## Next Steps

1. **Add More Content**: Create more blog posts following the format
2. **Customize Styling**: Modify `assets/css/style.scss` for custom styling
3. **Add Images**: Add images to enhance your posts
4. **Create Learning Paths**: Expand the learning_paths/ directory
5. **Build Topics**: Add more topic-specific content
6. **Set Up Domain**: Configure your custom domain via CNAME file

## Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Guide](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)

## Support

For issues or questions, check the [GitHub repository](https://github.com/msmariswamy/ai_website).

---

*Happy Learning!* 🚀