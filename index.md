---
layout: default
title: AI Learning Platform
description: Your pathway to mastering Artificial Intelligence with structured learning paths
hero: true
---

<div class="container">

  <div style="text-align: center; margin-bottom: 3rem;">
    <h2 style="margin-bottom: 1rem;">Welcome to AI Learning Platform</h2>
    <p style="color: #666; max-width: 600px; margin: 0 auto;">Learn artificial intelligence through structured paths and practical examples</p>
  </div>

  <div id="recent-posts" style="margin-bottom: 3rem;">
    <h2 style="margin-bottom: 1.5rem; text-align: center;">Latest Articles</h2>
    <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
      {% for post in site.posts limit: 5 %}
      <article style="margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #eee;">
        <h3 style="margin-bottom: 0.5rem; font-size: 1.2rem;">
          <a href="{{ post.url | relative_url }}" style="color: #2c3e50; text-decoration: none;">{{ post.title }}</a>
        </h3>
        <div style="color: #666; font-size: 0.85rem; margin-bottom: 0.5rem;">{{ post.date | date: "%B %-d, %Y" }}</div>
        <p style="margin: 0; color: #555;">{{ post.excerpt | strip_html | truncate: 150 }}</p>
      </article>
      {% endfor %}
    </div>
    <div style="text-align: center; margin-top: 1.5rem;">
      <a href="{{ '/blog' | relative_url }}" style="color: #3498db; text-decoration: none;">View all articles &rarr;</a>
    </div>
  </div>

  <div style="text-align: center; margin-top: 3rem;">
    <h2 style="margin-bottom: 1.5rem;">Learning Paths</h2>
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
      {% for p in site.learning_paths %}
      <a href="{{ '/learning_paths/' | relative_url }}{{ p.id }}"
         style="padding: 1rem 1.5rem; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-decoration: none; color: #2c3e50; min-width: 200px;">
        <span style="font-size: 2rem; display: block; margin-bottom: 0.5rem;">{{ p.icon }}</span>
        <h3 style="margin-bottom: 0.5rem;">{{ p.title }}</h3>
        <p style="font-size: 0.9rem; color: #666;">{{ p.description }}</p>
      </a>
      {% endfor %}
    </div>
  </div>

</div>