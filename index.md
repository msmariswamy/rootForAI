---
layout: default
title: AI Learning Platform
description: Your pathway to mastering Artificial Intelligence with structured learning paths
hero: true
---

<div class="path-selector" id="paths">
  <h2 style="margin: 0 0 1.5rem;">Choose Your Learning Path</h2>
  <div>
    {% for p in site.learning_paths %}
    <a href="{{ '/rootForAI/learning_paths/' | relative_url }}{{ p.id }}" class="path-card">
      <span class="icon">{{ p.icon }}</span>
      <h3>{{ p.title }}</h3>
      <p>{{ p.description }}</p>
    </a>
    {% endfor %}
  </div>
</div>

<div class="topics-grid" id="topics" style="margin-top: 3rem;">
  <h2 style="margin-bottom: 1.5rem;">Explore Topics</h2>
  {% for topic in site.topics %}
  <a href="{{ '/rootForAI/topics/' | relative_url }}{{ topic.id }}" class="topic-card" style="border-left-color: {{ topic.color }};">
    <h3 style="color: {{ topic.color }};">{{ topic.title }}</h3>
    <p>View all {{ topic.title }} articles</p>
  </a>
  {% endfor %}
</div>

<div id="recent-posts" style="margin-top: 3rem;">
  <h2 style="margin-bottom: 1.5rem;">Recent Posts</h2>
  <div class="posts-list">
    {% for post in site.posts limit: 6 %}
    <a href="{{ post.url | relative_url }}" class="post-card">
      <h3>{{ post.title }}</h3>
      <div class="meta">{{ post.date | date: "%b %-d, %Y" }}</div>
      <p>{{ post.excerpt | strip_html | truncate: 150 }}</p>
      <div class="tags">
        {% if post.path_type %}
        <span class="tag tag-{{ post.path_type }}">{{ post.path_type | capitalize }}</span>
        {% endif %}
        {% for cat in post.categories limit: 2 %}
        <span class="tag">{{ cat }}</span>
        {% endfor %}
      </div>
    </a>
    {% endfor %}
  </div>
  <a href="{{ '/rootForAI/posts' | relative_url }}" style="display: inline-block; margin-top: 1.5rem; color: #3498db; text-decoration: none;">View all posts &rarr;</a>
</div>
