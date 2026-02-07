---
layout: default
title: Blog
description: All AI learning articles and tutorials
hero: true
---

<div class="container" style="margin-top: 2rem;">
  <div style="margin-bottom: 2rem;">
    <h1 style="margin-bottom: 1rem; font-size: 2.5rem;">AI Learning Blog</h1>
    <p style="color: #666; font-size: 1.1rem;">Explore our collection of articles, tutorials, and resources on Artificial Intelligence</p>
  </div>

  <!-- Filters -->
  <div class="filters" style="margin-bottom: 2rem;">
    <h3 style="margin-bottom: 1rem;">Filter by Category</h3>
    <div class="filter-buttons">
      <button class="filter-btn active" data-filter="all">All Posts</button>
      {% for topic in site.topics %}
      <button class="filter-btn" data-filter="{{ topic.id }}" style="border-left: 3px solid {{ topic.color }};">{{ topic.title }}</button>
      {% endfor %}
    </div>
  </div>

  <!-- Posts Grid -->
  <div class="blog-posts-grid">
    {% for post in site.posts %}
    <article class="blog-post-card" data-categories="{% for cat in post.categories %}{{ cat }} {% endfor %}">
      <div class="post-header">
        <span class="post-date">{{ post.date | date: "%B %-d, %Y" }}</span>
        {% if post.path_type %}
        <span class="post-path-badge path-badge-{{ post.path_type }}">{{ post.path_type | capitalize }}</span>
        {% endif %}
      </div>
      <h2 class="post-title">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h2>
      <p class="post-excerpt">{{ post.excerpt | strip_html | truncate: 160 }}</p>
      <div class="post-footer">
        <div class="post-categories">
          {% for category in post.categories limit: 3 %}
          <span class="category-tag">{{ category }}</span>
          {% endfor %}
        </div>
        {% if post.read_time %}
        <span class="read-time">{{ post.read_time }} min read</span>
        {% endif %}
      </div>
    </article>
    {% endfor %}
  </div>
</div>

<script>
// Filter functionality
document.addEventListener('DOMContentLoaded', function() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  const postCards = document.querySelectorAll('.blog-post-card');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', function() {
      // Update active state
      filterBtns.forEach(b => b.classList.remove('active'));
      this.classList.add('active');

      const filter = this.dataset.filter;

      postCards.forEach(card => {
        if (filter === 'all') {
          card.style.display = 'block';
        } else {
          const categories = card.dataset.categories;
          if (categories.includes(filter)) {
            card.style.display = 'block';
          } else {
            card.style.display = 'none';
          }
        }
      });
    });
  });
});
</script>