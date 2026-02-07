---
layout: default
title: Blog
description: All AI learning articles and tutorials
hero: true
---

<div class="container" style="margin-top: 2rem;">
  <div style="margin-bottom: 2rem;">
    <h1 style="margin-bottom: 1rem; font-size: 2rem;">AI Learning Blog</h1>
    <p style="color: #666; font-size: 1rem;">Explore our collection of articles on Artificial Intelligence</p>
  </div>

  <!-- Posts List -->
  <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
    {% for post in site.posts %}
    <article style="margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #eee;">
      <h2 style="margin-bottom: 0.5rem; font-size: 1.3rem;">
        <a href="{{ post.url | relative_url }}" style="color: #2c3e50; text-decoration: none;">{{ post.title }}</a>
      </h2>
      <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.75rem;">
        <span>{{ post.date | date: "%B %-d, %Y" }}</span>
        {% if post.path_type %}
        <span style="margin-left: 0.75rem; padding: 0.1rem 0.5rem; background: #f1f8ff; border-radius: 10px; font-size: 0.7rem;">{{ post.path_type | capitalize }}</span>
        {% endif %}
      </div>
      <p style="margin: 0; color: #555; line-height: 1.5;">{{ post.excerpt | strip_html | truncate: 200 }}</p>
    </article>
    {% endfor %}
  </div>
</div>