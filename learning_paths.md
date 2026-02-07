---
layout: default
title: Learning Paths
description: Choose your AI learning journey
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
